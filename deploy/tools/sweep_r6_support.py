#!/usr/bin/env python3
"""Does the R6 policy work *anywhere* inside its own training support?

A single in-support start pose that fails proves little -- it could be an
unlucky corner.  This samples the support systematically, runs the policy
against a perfect kinematic arm from each start, and reports how many converge.
The same sweep runs the geometric servo for comparison, so the two are measured
on identical geometry rather than on anecdotes.

    python3 tools/sweep_r6_support.py \\
      --model ../models/rl/r6_unified_single_goal_yaw15_seed1_final.zip \\
      --grasp-pos  <x y z> --grasp-quat <qx qy qz qw> \\
      --grid 3 --rotations 25.7 57.1 100.2

Every sampled start is verified in-support before it is used, so a failure
cannot be blamed on distribution.  If the success rate is zero across the box,
the policy does not work inside the region it was trained on, and no amount of
start-pose engineering will change that -- which is a result, not a dead end.

The jaw is swept too.  The demonstrations began at 0.76 normalised and closed
to 0.0 during the approach, and the jaw is 3 of the 21 observation dimensions,
so running with a jaw that never moves is itself off-distribution.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_r6_start import solve_start_pose, support_report  # noqa: E402

from surgicai_rl_deploy.contract import (  # noqa: E402
    R6_START_OFFSET_TOOL_MAX,
    R6_START_OFFSET_TOOL_MIN,
    R6_START_TO_GOAL_ROTVEC_MEAN,
)
from surgicai_rl_deploy.controllers import D2Controller, RLController  # noqa: E402
from surgicai_rl_deploy.frames import Pose  # noqa: E402
from surgicai_rl_deploy.loop import ApproachLoop, LoopConfig, SafetyLimits  # noqa: E402


def run_episode(controller, start: Pose, grasp: Pose, *, goal_jaw, max_steps,
                success_trans_cm, success_rot_deg, frame_mode):
    loop = ApproachLoop(
        controller,
        LoopConfig(
            frame_mode=frame_mode,
            goal_orientation="explicit",
            goal_quat_xyzw=tuple(grasp.quat_xyzw()),
            goal_jaw=goal_jaw,
            max_steps=max_steps,
            success_trans_cm=success_trans_cm,
            success_rot_rad=float(np.deg2rad(success_rot_deg)),
        ),
        SafetyLimits(),
    )
    loop.begin(start, grasp.p)

    measured = start
    closest = np.inf
    result = None
    for _ in range(max_steps):
        result = loop.step(measured)
        closest = min(closest, result.trans_err_cm)
        if result.done:
            break
        target = result.command
        rel = Rotation.from_matrix(measured.R.T @ target.R).as_rotvec()
        measured = Pose(
            target.p, measured.R @ Rotation.from_rotvec(rel).as_matrix(), target.jaw
        )
    return {
        "outcome": result.reason,
        "steps": int(result.index),
        "final_trans_cm": float(result.trans_err_cm),
        "final_rot_deg": float(result.rot_err_deg),
        "closest_trans_cm": float(closest),
        "success": result.reason == "success",
    }


def sample_offsets(grid: int):
    """A grid over the support box, inset so samples sit strictly inside."""
    if grid < 2:
        from surgicai_rl_deploy.contract import R6_START_OFFSET_TOOL_MEAN

        return [np.asarray(R6_START_OFFSET_TOOL_MEAN, dtype=np.float64)]
    axes = []
    for lo, hi in zip(R6_START_OFFSET_TOOL_MIN, R6_START_OFFSET_TOOL_MAX):
        pad = 0.1 * (hi - lo)
        axes.append(np.linspace(lo + pad, hi - pad, grid))
    return [np.array(p) for p in itertools.product(*axes)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-unknown-model", action="store_true")
    ap.add_argument("--grasp-pos", nargs=3, type=float, required=True)
    ap.add_argument("--grasp-quat", nargs=4, type=float, required=True)

    ap.add_argument("--grid", type=int, default=3,
                    help="samples per axis of the offset box (3 -> 27 starts)")
    ap.add_argument("--rotations", nargs="*", type=float,
                    default=[25.7, 57.1, 100.2],
                    help="start->goal rotations to try, degrees")
    ap.add_argument("--start-jaw", nargs="*", type=float, default=[0.76, 0.0],
                    help="normalised start jaw values to try; 0.76 is what the "
                         "demonstrations began at")
    ap.add_argument("--goal-jaw", nargs="*", default=["closed", "hold"],
                    help="goal jaw specs to try; training closed the jaw")

    ap.add_argument("--frame-mode", choices=["rebase", "translate", "identity"],
                    default="rebase")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--success-trans-cm", type=float, default=1.0)
    ap.add_argument("--success-rot-deg", type=float, default=10.0)
    ap.add_argument("--skip-servo", action="store_true")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    from surgicai_rl_deploy.policy import ApproachPolicy

    policy = ApproachPolicy.load(
        args.model, device=args.device, verify=not args.allow_unknown_model
    )
    rl = RLController(policy)
    servo = D2Controller(staged=True)

    grasp = Pose.from_pos_quat(args.grasp_pos, args.grasp_quat, 0.0)
    offsets = sample_offsets(args.grid)

    unit = np.asarray(R6_START_TO_GOAL_ROTVEC_MEAN, dtype=np.float64)
    unit = unit / np.linalg.norm(unit)

    rows = []
    print(f"model      : {policy.describe()}")
    print(f"frame mode : {args.frame_mode}")
    print(f"starts     : {len(offsets)} offsets x {len(args.rotations)} rotations "
          f"x {len(args.start_jaw)} start jaws x {len(args.goal_jaw)} goal jaws")
    print()
    header = (f"{'offset (cm)':<22}{'rot':>7}{'jaw':>12}  "
              f"{'RL':>22}   {'servo':>22}")
    print(header)
    print("-" * len(header))

    for offset in offsets:
        for rot_deg in args.rotations:
            rotvec = unit * float(np.deg2rad(rot_deg))
            start_base = solve_start_pose(grasp, offset, rotvec)
            support = support_report(start_base, grasp)
            for start_jaw in args.start_jaw:
                for goal_jaw in args.goal_jaw:
                    start = Pose(start_base.p, start_base.R, float(start_jaw))
                    common = dict(
                        goal_jaw=goal_jaw, max_steps=args.max_steps,
                        success_trans_cm=args.success_trans_cm,
                        success_rot_deg=args.success_rot_deg,
                        frame_mode=args.frame_mode,
                    )
                    rl_res = run_episode(rl, start, grasp, **common)
                    servo_res = (
                        None if args.skip_servo
                        else run_episode(servo, start, grasp, **common)
                    )
                    rows.append({
                        "offset_cm": offset.tolist(),
                        "rotation_deg": float(rot_deg),
                        "start_jaw": float(start_jaw),
                        "goal_jaw": goal_jaw,
                        "in_support": support["in_support"],
                        "rl": rl_res,
                        "servo": servo_res,
                    })
                    def fmt(res):
                        if res is None:
                            return f"{'-':>22}"
                        mark = "OK " if res["success"] else "   "
                        return (f"{mark}{res['outcome'][:9]:<10}"
                                f"{res['closest_trans_cm']:>5.2f}cm"
                                f"{res['steps']:>4}")
                    print(f"{str(np.round(offset, 2)):<22}{rot_deg:>7.1f}"
                          f"{start_jaw:>6.2f}/{goal_jaw:<5}  "
                          f"{fmt(rl_res)}   {fmt(servo_res)}"
                          + ("" if support["in_support"] else "  [OUT OF SUPPORT]"))

    rl_ok = sum(1 for r in rows if r["rl"]["success"])
    servo_ok = sum(1 for r in rows if r["servo"] and r["servo"]["success"])
    in_support = sum(1 for r in rows if r["in_support"])
    best_rl = min((r["rl"]["closest_trans_cm"] for r in rows), default=float("nan"))

    print()
    print(f"episodes            : {len(rows)}  ({in_support} verified in-support)")
    print(f"RL converged        : {rl_ok}/{len(rows)}")
    if not args.skip_servo:
        print(f"servo converged     : {servo_ok}/{len(rows)}")
    print(f"best RL approach    : {best_rl:.2f} cm")
    print()
    if rl_ok == 0:
        print("The policy did not converge from ANY sampled start inside its own")
        print("training support. The out-of-distribution geometry was not what was")
        print("wrong with it, and choosing a better start pose cannot fix it.")
    elif rl_ok < len(rows):
        print("The policy converges from part of its support only. The successful")
        print("region is worth reading off the table above before trusting it.")
    else:
        print("The policy converged from every sampled in-support start.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "model": policy.describe(),
            "frame_mode": args.frame_mode,
            "grasp_pos_m": grasp.p.tolist(),
            "grasp_quat_xyzw": list(map(float, args.grasp_quat)),
            "rl_success": rl_ok,
            "servo_success": servo_ok,
            "episodes": len(rows),
            "rows": rows,
        }, indent=2))
        print(f"\nwrote {args.json_out}")

    return 0 if rl_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
