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

from surgicai_rl_deploy.calib import cli as calib_cli  # noqa: E402

from surgicai_rl_deploy.controllers import D2Controller, RLController, ResidualController
from surgicai_rl_deploy.feasibility import precheck
from surgicai_rl_deploy.frames import Pose
from surgicai_rl_deploy.jaw import JawBaseline, JawCalibration
from surgicai_rl_deploy.loop import SafetyLimits
from surgicai_rl_deploy.mock import MockArm, MockJaw
from surgicai_rl_deploy.contract import CONTRACTS
from surgicai_rl_deploy.plan import LiftSpec, TransportSpec, build_plan

CONTRACT_NAMES = tuple(CONTRACTS)
from surgicai_rl_deploy.sequence import (
    GraspLiftSequencer,
    SequenceConfig,
    PHASE_DONE,
)


def load_policy(path, device, verify, named_contract=None):
    """Return ``(policy, contract)``, resolving the contract from the digest."""
    from surgicai_rl_deploy.contract import contract_for_digest
    from surgicai_rl_deploy.policy import ApproachPolicy

    policy = ApproachPolicy.load(path, device=device, verify=verify)
    if named_contract:
        return policy, CONTRACTS[named_contract]
    contract = contract_for_digest(policy.sha256)
    if contract is None:
        print(
            f"WARNING: {Path(path).name} has no registered contract "
            f"(sha256 {policy.sha256[:12]}), so its action scale, episode budget "
            "and tolerances are guesses. Settle them with "
            "tools/replay_demos.py --compare, then pass --contract."
        )
    return policy, contract


def build_controller(args):
    """Return ``(controller, contract)`` for the approach leg."""
    if args.controller == "d2":
        # A named contract still applies: staging into a demonstrated support
        # is geometry, and comparing the servo against the same support is the
        # whole point of having both controllers.
        named = getattr(args, "contract", None)
        return D2Controller(staged=True), (CONTRACTS[named] if named else None)
    if not args.model:
        raise SystemExit("--model is required for the rl/residual controllers")
    policy, contract = load_policy(
        args.model, args.device, not args.allow_unknown_model, args.contract
    )
    if args.controller == "rl":
        return RLController(policy), contract
    return ResidualController(
        policy, policy_weight=args.policy_weight, servo_weight=args.servo_weight
    ), contract


def build_shadow(args):
    """Return ``(controller, contract)`` for the logged-only transport policy."""
    if not args.shadow_model:
        return None, None
    policy, contract = load_policy(
        args.shadow_model, args.device, not args.allow_unknown_model,
        args.shadow_contract,
    )
    return RLController(policy), contract


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start-pos", nargs=3, type=float, required=True)
    ap.add_argument("--start-quat", nargs=4, type=float, required=True)
    ap.add_argument("--grasp-pos", nargs=3, type=float, default=None)
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
    ap.add_argument("--contract", choices=sorted(CONTRACT_NAMES),
                    help="force a checkpoint contract instead of resolving it "
                         "from the model's SHA256")

    ap.add_argument("--grasp-standoff-mm", type=float, default=None,
                    help="how far short of the grasp pose the approach policy "
                         "aims, along the tool axis. Default: the checkpoint "
                         "contract's value (7 mm for the Approach policies, "
                         "matching needle_goal_evaluator's lift_height). The "
                         "remaining distance is a separate slow descent. 0 "
                         "disables it and aims straight at the needle.")
    ap.add_argument("--descend-step-mm", type=float, default=0.5)
    ap.add_argument("--stage-success-trans-cm", type=float, default=0.2)
    ap.add_argument("--descend-success-trans-cm", type=float, default=0.05,
                    help="how close the jaws must get to the grasp pose before "
                         "closing. Cannot be finer than the arm's deadband; the "
                         "precheck refuses the combination.")
    ap.add_argument("--transport-success-trans-cm", type=float, default=0.3)
    ap.add_argument("--place-success-trans-cm", type=float, default=0.2)
    ap.add_argument("--descend-max-steps", type=int, default=200)
    ap.add_argument("--suture-pos", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="TOOL position at the suturing point, metres. This is "
                         "where measured_cp should read when the needle is "
                         "placed, not where the needle is.")
    ap.add_argument("--suture-quat", nargs=4, type=float, default=None,
                    metavar=("QX", "QY", "QZ", "QW"),
                    help="tool orientation at the suturing point -- the needle "
                         "angle, expressed as a gripper pose")
    ap.add_argument("--suture-confirmed", action="store_true",
                    help="a human has checked this pose against the scene")
    ap.add_argument("--taught-grasp-pos", nargs=3, type=float, default=None,
                    help="the grasp pose the SUTURING pose was taught with, if "
                         "different from --grasp-pos. The suturing pose encodes "
                         "how the needle sat in the jaws at that moment.")
    ap.add_argument("--taught-grasp-quat", nargs=4, type=float, default=None)
    ap.add_argument("--compensate-suture", choices=["apply", "report", "off"],
                    default="apply",
                    help="correct the suturing pose for where the jaws actually "
                         "closed. Exact, and needs neither the needle pose nor "
                         "the entry pose; assumes the needle did not move.")
    ap.add_argument("--max-suture-compensation-mm", type=float, default=10.0)
    ap.add_argument("--transport-via", choices=["lift_height", "direct"],
                    default="lift_height")
    ap.add_argument("--transport-clearance-cm", type=float, default=None,
                    help="height of the via point above the suturing pose; "
                         "default is the lift distance")
    ap.add_argument("--on-approach-failure", choices=["hold", "servo"],
                    default="hold")

    ap.add_argument("--stage", action="store_true",
                    help="insert a staging move so the approach policy starts "
                         "inside its own demonstrated support")
    ap.add_argument("--stage-offset-tool-cm", nargs=3, type=float, default=None,
                    help="where to sit inside the demonstration box; default is "
                         "its mean, which maximises margin")
    ap.add_argument("--stage-rotation-deg", type=float, default=None)

    ap.add_argument("--shadow-model",
                    help="a second checkpoint run alongside the transport and "
                         "place legs, logged and never published")
    ap.add_argument("--shadow-contract", choices=sorted(CONTRACT_NAMES))

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
    ap.add_argument("--min-command-mm", type=float, default=0.0)
    ap.add_argument("--deadband-mm", type=float, default=0.0,
                    help="mock arm ignores commanded displacements below this, "
                         "like a real PSM does. Use it to reproduce a stall.")
    ap.add_argument("--lag", type=float, default=0.0)
    ap.add_argument("--noise-mm", type=float, default=0.0)
    ap.add_argument("--jaw-noise-deg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-cycles", type=int, default=800)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out")
    calib_cli.add_arguments(ap)
    parsed = ap.parse_args(argv)
    if parsed.grasp_pos is None and not parsed.needle_pose:
        ap.error("one of --grasp-pos or --needle-pose is required")
    return parsed


def main(argv=None) -> int:
    args = parse_args(argv)

    # A needle observation, if one was given, becomes --grasp-pos before
    # anything else looks at it -- exactly as it does on the real node, so this
    # rehearsal exercises the same path.
    target, messages = calib_cli.apply_to_args(args, strict=getattr(args, "strict", False))
    for line in messages:
        print(line)
    if target is not None:
        print(target.report.render())
        if not target.report.ok:
            return 3
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
    controller, contract = build_controller(args)
    shadow_controller, shadow_contract = build_shadow(args)

    # The policy's goal sits short of the needle; the contract knows by how much.
    standoff = (
        args.grasp_standoff_mm / 1000.0 if args.grasp_standoff_mm is not None
        else (contract.grasp_standoff_m if contract is not None else 0.0)
    )

    if args.suture_pos is not None and args.suture_quat is None:
        raise SystemExit(
            "--suture-pos needs --suture-quat: the point of the leg is to "
            "present the needle at an angle, so the orientation is the payload"
        )
    if args.stage and contract is None:
        raise SystemExit(
            "--stage needs a checkpoint contract to stage INTO. Pass --model "
            "with --controller rl, or name one with --contract."
        )

    taught_grasp = None
    if args.taught_grasp_pos is not None:
        if args.taught_grasp_quat is None:
            raise SystemExit("--taught-grasp-pos needs --taught-grasp-quat")
        taught_grasp = Pose.from_pos_quat(
            args.taught_grasp_pos, args.taught_grasp_quat, 0.0
        )

    transport = None
    if args.suture_pos is not None:
        transport = TransportSpec(
            via=args.transport_via,
            via_clearance_m=(
                None if args.transport_clearance_cm is None
                else args.transport_clearance_cm / 100.0
            ),
        )

    plan = build_plan(
        start,
        args.grasp_pos,
        goal_orientation=args.goal_orientation
        or ("explicit" if args.grasp_quat else "hold"),
        goal_quat_xyzw=tuple(args.grasp_quat) if args.grasp_quat else None,
        lift=lift,
        jaw=jaw_cal,
        grasp_standoff_m=standoff,
        suture_position_m=args.suture_pos,
        suture_quat_xyzw=tuple(args.suture_quat) if args.suture_quat else None,
        transport=transport,
        taught_grasp_pose=taught_grasp,
        stage_contract=contract if args.stage else None,
        stage_offset_tool_cm=args.stage_offset_tool_cm,
        stage_rotation_deg=args.stage_rotation_deg,
    )

    cfg = SequenceConfig(
        frame_mode=args.frame_mode,
        approach_contract=contract,
        descend_max_steps=args.descend_max_steps,
        descend_step_mm=args.descend_step_mm,
        descend_success_trans_cm=args.descend_success_trans_cm,
        stage_success_trans_cm=args.stage_success_trans_cm,
        transport_success_trans_cm=args.transport_success_trans_cm,
        place_success_trans_cm=args.place_success_trans_cm,
        on_approach_failure=args.on_approach_failure,
        grasp_gate=args.grasp_gate,
        on_slip=args.on_slip,
        settle_steps=args.settle_steps,
        settle_translation_tol_mm=args.settle_translation_tol_mm,
        settle_timeout_steps=args.settle_timeout_steps,
        residual_margin_deg=args.residual_margin_deg,
    )
    limits = SafetyLimits(min_command_mm=args.min_command_mm)

    report = precheck(
        plan,
        controller=args.controller,
        execute=False,
        grasp_gate=args.grasp_gate,
        jaw_baseline=baseline,
        approach_max_steps=cfg.approach_max_steps,
        lift_max_steps=cfg.lift_max_steps,
        descend_max_steps=cfg.descend_max_steps,
        descend_step_mm=cfg.descend_step_mm,
        min_command_mm=args.min_command_mm,
        tolerances_mm={
            'stage': cfg.stage_success_trans_cm * 10.0,
            'approach': cfg.approach_success_trans_cm * 10.0,
            'descend': cfg.descend_success_trans_cm * 10.0,
            'lift': cfg.lift_success_trans_cm * 10.0,
            **({} if plan.suture is None else {
                'transport': cfg.transport_success_trans_cm * 10.0,
                'place': cfg.place_success_trans_cm * 10.0}),
        },
        transport_max_steps=cfg.transport_max_steps,
        place_max_steps=cfg.place_max_steps,
        success_trans_cm=cfg.lift_success_trans_cm,
        suture_confirmed=args.suture_confirmed,
        stage_contract=contract,
        strict=args.strict,
    )
    print(report.render())
    print()
    if not report.ok:
        return 2

    sequencer = GraspLiftSequencer(
        plan, controller, cfg, limits, baseline,
        shadow_controller=shadow_controller, shadow_contract=shadow_contract,
    )

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
        deadband_mm=args.deadband_mm,
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
    if contract is not None:
        print(f"contract    : {contract.describe()}")
    if shadow_controller is not None:
        print(f"shadow      : {shadow_controller.describe()} (logged, never published)")
    print("plan        : " + " -> ".join(n for n, _ in plan.waypoints))
    print(f"              approach {plan.approach_travel_cm:.2f} cm, "
          f"lift {plan.lift_spec.describe()}")
    if plan.suture is not None:
        print(f"              transport {plan.transport_travel_cm:.2f} cm / "
              f"{plan.transport_rotation_deg:.1f} deg, "
              f"{(plan.transport_spec).describe()}")
    print(f"{jaw_cal.describe()}")
    print()

    approach_report = sequencer.begin(arm.state())
    for line in (approach_report or {}).get("out_of_distribution") or []:
        print(f"  - approach OUTSIDE the demonstration support: {line}")

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
    if plan.suture is not None:
        err = float(np.linalg.norm(arm.pose.p - plan.suture.p) * 100.0)
        print(f"distance from the suturing pose at the end: {err:.3f} cm "
              f"(reached={summary['reached_suture_pose']})")
    if summary.get("shadow"):
        sh = summary["shadow"]
        print(f"shadow      : {sh['cycles']} cycles of one-step advice; "
              f"divergence from what was commanded: median "
              f"{(sh['median_divergence_mm'] or 0.0):.2f} mm / "
              f"{(sh['median_divergence_deg'] or 0.0):.2f} deg, max "
              f"{(sh['max_divergence_mm'] or 0.0):.2f} mm; "
              f"{sh['cycles_clamped']} cycles clamped")
        if sh.get("support") and not sh["support"].get("in_distribution"):
            for line in sh["support"]["out_of_distribution"]:
                print(f"              out of distribution: {line}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "precheck": report.as_dict(),
                    "plan": plan.as_dict(),
                    "controller": controller.describe(),
                    "approach_report": {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in (approach_report or {}).items()
                    },
                    "shadow_log": sequencer.shadow_log,
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
