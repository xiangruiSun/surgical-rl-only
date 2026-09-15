#!/usr/bin/env python3
"""Replay the approach loop against a kinematic mock of the arm.

No ROS, no robot.  The mock assumes the PSM reaches each commanded pose
exactly (optionally with lag and noise), which is the *optimistic* case: if the
controller cannot converge here, it will not converge on hardware.

Example (the pose you echoed on lcsr-dvrk-15):

    python tools/offline_check.py \
      --model ~/surgicai-rl-only/models/rl/r6_unified_single_goal_yaw15_seed1_final.zip \
      --start-pos -0.05639860616831881 0.03366166453830251 0.024455994074878362 \
      --start-quat 0.23319925218484056 0.4267863636861243 -0.23588767438897446 0.841325450478807 \
      --goal-pos -0.050726357 0.015332369 0.049514053 \
      --controller rl --frame-mode rebase
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.controllers import D2Controller, RLController, ResidualController
from surgicai_rl_deploy.frames import Pose
from surgicai_rl_deploy.loop import ApproachLoop, LoopConfig, SafetyLimits


def build_controller(args):
    if args.controller == "d2":
        return D2Controller(staged=args.staged)
    from surgicai_rl_deploy.policy import ApproachPolicy

    if not args.model:
        raise SystemExit("--model is required for the rl/residual controllers")
    policy = ApproachPolicy.load(args.model, device=args.device, verify=not args.allow_unknown_model)
    if args.controller == "rl":
        return RLController(policy)
    return ResidualController(policy, policy_weight=args.policy_weight,
                              servo_weight=args.servo_weight, staged=args.staged)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--controller", choices=["rl", "d2", "residual"], default="rl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-unknown-model", action="store_true")
    ap.add_argument("--start-pos", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--start-quat", nargs=4, type=float, required=True,
                    metavar=("QX", "QY", "QZ", "QW"))
    ap.add_argument("--goal-pos", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--goal-quat", nargs=4, type=float, default=None)
    ap.add_argument("--goal-orientation", choices=["hold", "trained_relative", "explicit"],
                    default=None, help="default: hold, or explicit if --goal-quat is given")
    ap.add_argument("--success-trans-cm", type=float, default=1.0)
    ap.add_argument("--success-rot-deg", type=float, default=10.0)
    ap.add_argument("--start-jaw", type=float, default=0.0,
                    help="normalised 0..1. The R6 demonstrations started at "
                         "0.76, and the jaw occupies 3 of the 21 observation "
                         "dimensions, so 0.0 is itself off-distribution.")
    ap.add_argument("--goal-jaw", default="hold",
                    help="hold | closed | open | <float>. Training drove the "
                         "jaw to 0.0 (closed) during the approach; 'hold' "
                         "leaves it at --start-jaw, which makes the jaw error "
                         "identically zero.")
    ap.add_argument("--frame-mode", choices=["rebase", "translate", "identity"], default="rebase")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--staged", action="store_true", default=True)
    ap.add_argument("--policy-weight", type=float, default=0.50)
    ap.add_argument("--servo-weight", type=float, default=0.75)
    ap.add_argument("--lag", type=float, default=0.0,
                    help="0 = arm reaches the command exactly; 0.3 = 30%% short each step")
    ap.add_argument("--noise-mm", type=float, default=0.0, help="per-step measurement noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    controller = build_controller(args)

    cfg = LoopConfig(
        frame_mode=args.frame_mode,
        goal_orientation=args.goal_orientation or ("explicit" if args.goal_quat else "hold"),
        goal_quat_xyzw=tuple(args.goal_quat) if args.goal_quat else None,
        goal_jaw=args.goal_jaw,
        max_steps=args.max_steps,
        success_trans_cm=args.success_trans_cm,
        success_rot_rad=float(np.deg2rad(args.success_rot_deg)),
    )
    loop = ApproachLoop(controller, cfg, SafetyLimits())

    start = Pose.from_pos_quat(args.start_pos, args.start_quat, args.start_jaw)
    report = loop.begin(start, args.goal_pos)

    np.set_printoptions(precision=3, suppress=True)
    print(f"controller      : {controller.describe()}")
    print(f"frame mode      : {report['frame_mode']}")
    print(f"path length     : {report['translation_cm']:.2f} cm")
    print(f"start->goal rot : {report['start_to_goal_rotation_deg']:.1f} deg")
    print(f"start offset in tool frame (cm): {report['start_offset_tool_cm']}")
    if report["in_distribution"]:
        print("training support: INSIDE the R6 demonstration support")
    else:
        print("training support: OUTSIDE the R6 demonstration support")
        for line in report["out_of_distribution"]:
            print(f"  - {line}")
    print()

    measured = start
    trace = []
    result = None
    for _ in range(args.max_steps):
        result = loop.step(measured)
        trace.append(
            {
                "i": result.index,
                "trans_err_cm": result.trans_err_cm,
                "rot_err_deg": result.rot_err_deg,
                "action": result.action.round(3).tolist(),
                "clamps": [c["kind"] for c in result.clamps],
            }
        )
        if args.verbose or result.index % 20 == 0 or result.done:
            print(
                f"  step {result.index:3d}  err {result.trans_err_cm:6.2f} cm / "
                f"{result.rot_err_deg:6.2f} deg   action {result.action.round(2)}"
                + (f"   [{','.join(c['kind'] for c in result.clamps)}]" if result.clamps else "")
            )
        if result.done:
            break
        # kinematic mock of the arm following the command
        target = result.command
        p = measured.p + (target.p - measured.p) * (1.0 - args.lag)
        if args.noise_mm:
            p = p + rng.normal(0.0, args.noise_mm / 1000.0, 3)
        from scipy.spatial.transform import Rotation

        rel = Rotation.from_matrix(measured.R.T @ target.R).as_rotvec() * (1.0 - args.lag)
        R = measured.R @ Rotation.from_rotvec(rel).as_matrix()
        measured = Pose(p, R, target.jaw)

    print()
    print(f"outcome         : {result.reason}")
    print(f"steps           : {result.index}")
    print(f"final error     : {result.trans_err_cm:.2f} cm / {result.rot_err_deg:.2f} deg")
    best = min(t["trans_err_cm"] for t in trace)
    print(f"closest approach: {best:.2f} cm")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "controller": controller.describe(),
                    "report": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in report.items()},
                    "outcome": result.reason,
                    "steps": result.index,
                    "final_trans_err_cm": result.trans_err_cm,
                    "final_rot_err_deg": result.rot_err_deg,
                    "closest_trans_err_cm": best,
                    "trace": trace,
                },
                indent=2,
            )
        )
        print(f"wrote {args.json_out}")
    return 0 if result.reason == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
