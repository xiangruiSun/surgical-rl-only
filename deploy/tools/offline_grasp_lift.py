#!/usr/bin/env python3
"""Replay a full approach -> close -> observe -> lift episode. No ROS, no robot.

The arm is a kinematic mock that reaches each commanded pose (optionally with
lag and noise), and the jaw is a first-order model with an optional physical
stop standing in for a needle between the fingers.  This is the optimistic
case: a sequence that will not finish here will not finish on hardware.

Example, on the pose echoed from lcsr-dvrk-15:

    python3 tools/offline_grasp_lift.py \
      --start-pos  -0.05639860616831881 0.03366166453830251 0.024455994074878362 \
      --start-quat  0.23319925218484056 0.4267863636861243 -0.23588767438897446 0.841325450478807 \
      --grasp-pos  -0.050726357 0.015332369 0.049514053 \
      --controller d2 --lift-sign -1 --grasp-gate evidence \
      --jaw-stops-at-deg 3.0

Rehearsals worth running before you touch the robot:

===============================================  ====================================
``--empty-gripper``                              the evidence gate must refuse to lift
``--jaw-stops-at-deg 3``                         a 0.5 mm needle held near the pivot
``--jaw-stops-at-deg -14.5``                     cable stretch hides nearly all of it
``--jaw-stops-at-deg -14.5 --no-jaw-effort``     and with no effort field it is lost
``--drop-at-step N``                             the needle slips mid-lift
``--lift-sign +1``                               the precheck warns about direction
``--lag 0.6 --noise-mm 0.3``                     a sloppy, noisy arm
===============================================  ====================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.controllers import D2Controller, RLController, ResidualController
from surgicai_rl_deploy.feasibility import precheck
from surgicai_rl_deploy.frames import Pose
from surgicai_rl_deploy.jaw import JawBaseline, JawCalibration
from surgicai_rl_deploy.loop import SafetyLimits
from surgicai_rl_deploy.mock import MockArm, MockJaw
from surgicai_rl_deploy.plan import LiftSpec, build_plan
from surgicai_rl_deploy.sequence import (
    GraspLiftSequencer,
    SequenceConfig,
    PHASE_DONE,
)


def build_controller(args):
    if args.controller == "d2":
        return D2Controller(staged=True)
    from surgicai_rl_deploy.policy import ApproachPolicy

    if not args.model:
        raise SystemExit("--model is required for the rl/residual controllers")
    policy = ApproachPolicy.load(
        args.model, device=args.device, verify=not args.allow_unknown_model
    )
    if args.controller == "rl":
        return RLController(policy)
    return ResidualController(
        policy, policy_weight=args.policy_weight, servo_weight=args.servo_weight
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start-pos", nargs=3, type=float, required=True)
    ap.add_argument("--start-quat", nargs=4, type=float, required=True)
    ap.add_argument("--grasp-pos", nargs=3, type=float, required=True)
    ap.add_argument("--grasp-quat", nargs=4, type=float, default=None)
    ap.add_argument(
        "--goal-orientation",
        choices=["hold", "trained_relative", "explicit"],
        default=None,
    )
    ap.add_argument("--controller", choices=["rl", "d2", "residual"], default="d2")
    ap.add_argument("--model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-unknown-model", action="store_true")
    ap.add_argument("--policy-weight", type=float, default=0.50)
    ap.add_argument("--servo-weight", type=float, default=0.75)
    ap.add_argument("--frame-mode", choices=["rebase", "translate", "identity"],
                    default="rebase")

    ap.add_argument("--lift-axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--lift-sign", type=int, choices=[-1, 1], default=None)
    ap.add_argument("--lift-distance-cm", type=float, default=1.5)
    ap.add_argument("--lift-frame", choices=["robot", "tool"], default="robot")

    ap.add_argument("--jaw-open-deg", type=float, default=60.0)
    ap.add_argument("--jaw-closed-deg", type=float, default=0.0)
    ap.add_argument("--jaw-grip-deg", type=float, default=-15.0)
    ap.add_argument("--jaw-approach-open-deg", type=float, default=40.0)

    ap.add_argument("--grasp-gate", choices=["manual", "evidence", "always", "never"],
                    default="evidence",
                    help="offline default is 'evidence'; there is no operator here")
    ap.add_argument("--on-slip", choices=["abort", "continue", "lower"], default="abort")
    ap.add_argument("--jaw-baseline", help="JSON from tools/calibrate_jaw.py")
    ap.add_argument("--settle-steps", type=int, default=5)
    ap.add_argument("--settle-translation-tol-mm", type=float, default=0.5)
    ap.add_argument("--settle-timeout-steps", type=int, default=60)
    ap.add_argument("--residual-margin-deg", type=float, default=1.0)

    # mock arm
    ap.add_argument("--jaw-stops-at-deg", "--needle-blocks-jaw-deg",
                    dest="jaw_stops_at_deg", type=float, default=2.0,
                    help="the ANGLE the jaw stops at because a needle is between "
                         "the fingers -- not the needle's thickness. A 0.5 mm "
                         "needle roughly 10 mm from the pivot stops the jaw near "
                         "+3 deg. Use a value just above the empty-jaw close "
                         "angle to rehearse the hard case where cable stretch "
                         "hides most of the block. Pass the empty-close angle "
                         "itself for an empty gripper.")
    ap.add_argument("--empty-gripper", action="store_true",
                    help="nothing between the fingers: the jaw closes to the "
                         "command, exactly like the calibration run")
    ap.add_argument("--drop-at-step", type=int, default=None)
    ap.add_argument("--no-jaw-effort", action="store_true",
                    help="pretend jaw/measured_js carries no effort field, as on "
                         "some arms; the evidence then rests on the jaw angle "
                         "alone, which a thin needle may not move far enough")
    ap.add_argument("--lag", type=float, default=0.0)
    ap.add_argument("--noise-mm", type=float, default=0.0)
    ap.add_argument("--jaw-noise-deg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-cycles", type=int, default=800)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    np.set_printoptions(precision=3, suppress=True)

    jaw_cal = JawCalibration(
        open_rad=float(np.deg2rad(args.jaw_open_deg)),
        closed_rad=float(np.deg2rad(args.jaw_closed_deg)),
        grip_rad=float(np.deg2rad(args.jaw_grip_deg)),
        approach_open_rad=float(np.deg2rad(args.jaw_approach_open_deg)),
    )

    baseline = None
    if args.jaw_baseline:
        baseline = JawBaseline.from_dict(json.loads(Path(args.jaw_baseline).read_text()))
    elif args.grasp_gate == "evidence":
        # Offline, the empty-jaw behaviour of the mock is known exactly, so a
        # synthetic baseline is legitimate -- and is labelled as synthetic.
        baseline = JawBaseline(
            empty_close_rad=jaw_cal.grip_rad,
            empty_close_effort=MockJaw(angle_rad=0.0).effort_floor,
            empty_close_rad_noise=max(float(np.deg2rad(args.jaw_noise_deg)),
                                      float(np.deg2rad(0.1))),
            empty_close_effort_noise=0.005,
            source="synthetic (offline mock, not a real arm)",
        )

    lift = LiftSpec(
        axis=args.lift_axis,
        sign=int(args.lift_sign) if args.lift_sign is not None else 1,
        distance_m=args.lift_distance_cm / 100.0,
        frame=args.lift_frame,
        explicit=args.lift_sign is not None,
    )

    start = Pose.from_pos_quat(
        args.start_pos, args.start_quat, jaw_cal.normalise(jaw_cal.approach_open_rad)
    )
    plan = build_plan(
        start,
        args.grasp_pos,
        goal_orientation=args.goal_orientation
        or ("explicit" if args.grasp_quat else "hold"),
        goal_quat_xyzw=tuple(args.grasp_quat) if args.grasp_quat else None,
        lift=lift,
        jaw=jaw_cal,
    )

    cfg = SequenceConfig(
        frame_mode=args.frame_mode,
        grasp_gate=args.grasp_gate,
        on_slip=args.on_slip,
        settle_steps=args.settle_steps,
        settle_translation_tol_mm=args.settle_translation_tol_mm,
        settle_timeout_steps=args.settle_timeout_steps,
        residual_margin_deg=args.residual_margin_deg,
    )
    limits = SafetyLimits()

    report = precheck(
        plan,
        controller=args.controller,
        execute=False,
        grasp_gate=args.grasp_gate,
        jaw_baseline=baseline,
        approach_max_steps=cfg.approach_max_steps,
        lift_max_steps=cfg.lift_max_steps,
        success_trans_cm=cfg.lift_success_trans_cm,
        strict=args.strict,
    )
    print(report.render())
    print()
    if not report.ok:
        return 2

    controller = build_controller(args)
    sequencer = GraspLiftSequencer(plan, controller, cfg, limits, baseline)

    block = (
        None if args.empty_gripper else float(np.deg2rad(args.jaw_stops_at_deg))
    )
    arm = MockArm(
        start,
        MockJaw(
            angle_rad=jaw_cal.approach_open_rad,
            block_at_rad=block,
            noise_rad=float(np.deg2rad(args.jaw_noise_deg)),
            drop_at_step=args.drop_at_step,
        ),
        lag=args.lag,
        noise_mm=args.noise_mm,
        seed=args.seed,
        jaw_calibration=jaw_cal,
    )
    arm.prime_jaw()
    if args.no_jaw_effort:
        arm.jaw.effort_gain = 0.0
        arm.jaw.effort_floor = 0.0
        arm._last_jaw_effort = None
        original_apply = arm.apply

        def apply_without_effort(command):
            original_apply(command)
            arm._last_jaw_effort = None

        arm.apply = apply_without_effort
        if baseline is not None:
            baseline = JawBaseline(
                empty_close_rad=baseline.empty_close_rad,
                empty_close_effort=None,
                empty_close_rad_noise=baseline.empty_close_rad_noise,
                empty_close_effort_noise=None,
                source=baseline.source + " (effort channel disabled)",
            )
            sequencer.baseline = baseline

    print(f"controller  : {controller.describe()}")
    print(f"plan        : approach {plan.approach_travel_cm:.2f} cm, "
          f"lift {plan.lift_spec.describe()}")
    print(f"{jaw_cal.describe()}")
    print()

    approach_report = sequencer.begin(arm.state())
    if approach_report["out_of_distribution"]:
        print("approach is OUTSIDE the R6 demonstration support:")
        for line in approach_report["out_of_distribution"]:
            print(f"  - {line}")
        print()

    trace = []
    phase = None
    step = None
    for _ in range(args.max_cycles):
        step = sequencer.step(arm.state())
        trace.append(step.as_dict())
        if step.phase != phase:
            phase = step.phase
            print(f"--- phase: {phase}")
        if args.verbose or step.events or step.done:
            jaw = step.jaw_evidence
            jaw_txt = ""
            if jaw is not None:
                jaw_txt = (
                    f"  jaw cmd {np.degrees(jaw.commanded_rad):+6.1f} meas "
                    f"{np.degrees(jaw.measured_rad):+6.1f} blocked={jaw.jaw_blocked}"
                    if jaw.measured_rad is not None
                    else "  jaw: no feedback"
                )
            print(
                f"  {step.index:4d} {step.phase:<14s} err "
                f"{step.trans_err_cm:6.2f} cm / {step.rot_err_deg:6.2f} deg{jaw_txt}"
            )
            for event in step.events:
                print(f"       * {json.dumps(event, default=str)}")
        if step.done:
            break
        arm.apply(step.command)

    summary = sequencer.summary()
    print()
    print(f"final phase : {summary['phase']}")
    print(f"reason      : {summary['reason']}")
    print(f"cycles      : {summary['steps']}")
    lifted_cm = float(np.linalg.norm(arm.pose.p - plan.grasp.p) * 100.0)
    print(f"height above the grasp pose at the end: {lifted_cm:.2f} cm "
          f"(target {plan.lift_travel_cm:.2f} cm)")
    print("grasp verified: NO - this hardware has no grasp sensor")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "precheck": report.as_dict(),
                    "plan": plan.as_dict(),
                    "controller": controller.describe(),
                    "approach_report": {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in approach_report.items()
                    },
                    "summary": summary,
                    "final_height_cm": lifted_cm,
                    "trace": trace,
                },
                indent=2,
                default=str,
            )
        )
        print(f"wrote {args.json_out}")

    return 0 if summary["phase"] == PHASE_DONE and summary["reason"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
