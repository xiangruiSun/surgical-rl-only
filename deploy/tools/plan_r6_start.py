#!/usr/bin/env python3
"""Solve for a start pose that puts the R6 policy *inside* its training support.

Why this can be done at all
---------------------------
The two support constraints are

    tool offset   R_start^T (p_grasp - p_start)     in the demonstration box
    rotation      geodesic(R_start, R_grasp)        in [25.7, 100.2] deg

and both are invariant under the frame bridge: for any rigid ``X``,

    R_start_policy^T (p_goal_policy - p_start_policy)
        = (X.R R_start)^T X.R (p_goal - p_start)
        = R_start^T (p_goal - p_start)

and a geodesic between two rotations is unchanged by a common pre-rotation.
So the support can be satisfied by choosing the start pose in the robot's own
frame, with no reference to the training frame at all, and ``--frame-mode
rebase`` then lines the absolute goal up on top of the trained one.

Given a fixed grasp pose the solution is direct:

    R_start = R_grasp * Rel^-1        Rel = the mean start->goal rotation
    p_start = p_grasp - R_start * offset

What this does NOT tell you
---------------------------
That the policy will work.  It removes the out-of-distribution excuse; whether
the R6 checkpoint actually converges from an in-distribution start is an
empirical question, and the answer comes from running
``tools/offline_check.py --controller rl --model ...`` against the pose this
prints.  Do that before putting the arm anywhere.

Nor does it check reachability.  The pose is geometry; the arm may not be able
to adopt it, or may collide on the way.  ``--current-pos`` reports how far the
arm has to travel so you can judge that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.contract import (
    R6_START_OFFSET_TOOL_MAX,
    R6_START_OFFSET_TOOL_MEAN,
    R6_START_OFFSET_TOOL_MIN,
    R6_START_ROT_DEG_MAX,
    R6_START_ROT_DEG_MIN,
    R6_START_TO_GOAL_ROTVEC_MEAN,
    SUPPORT_EPS_CM,
)
from surgicai_rl_deploy.frames import Pose, rotation_error_rad
from surgicai_rl_deploy.staging import solve_start_pose as _solve_start_pose
from surgicai_rl_deploy.staging import support_report as _support_report


# The solver moved into the package so the node, the sweep and the pipeline
# all use one implementation; these names stay for callers and tests.
solve_start_pose = _solve_start_pose
support_report = _support_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grasp-pos", nargs=3, type=float, required=True,
                    metavar=("X", "Y", "Z"), help="metres, frame of measured_cp")
    ap.add_argument("--grasp-quat", nargs=4, type=float, required=True,
                    metavar=("QX", "QY", "QZ", "QW"),
                    help="the gripper orientation you want AT the grasp")
    ap.add_argument("--current-pos", nargs=3, type=float, default=None,
                    help="where the arm is now, to report how far it must move")
    ap.add_argument("--current-quat", nargs=4, type=float, default=None)

    ap.add_argument("--offset-tool-cm", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="where to sit inside the demonstration box; default is "
                         "the demonstrations' mean, which maximises margin")
    ap.add_argument("--rotation-deg", type=float, default=None,
                    help="scale the mean start->goal rotation to this angle "
                         "(default: leave it at its measured 57.1 deg)")
    ap.add_argument("--model", default="models/rl/r6_unified_single_goal_yaw15_seed1_final.zip",
                    help="only used to render the follow-up commands")
    ap.add_argument("--arm", default="/PSM1")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    np.set_printoptions(precision=4, suppress=True)

    grasp = Pose.from_pos_quat(args.grasp_pos, args.grasp_quat, 0.0)
    offset = (
        np.asarray(args.offset_tool_cm, dtype=np.float64)
        if args.offset_tool_cm is not None
        else np.asarray(R6_START_OFFSET_TOOL_MEAN, dtype=np.float64)
    )

    rotvec = np.asarray(R6_START_TO_GOAL_ROTVEC_MEAN, dtype=np.float64)
    if args.rotation_deg is not None:
        magnitude = float(np.linalg.norm(rotvec))
        rotvec = rotvec / magnitude * float(np.deg2rad(args.rotation_deg))

    start = solve_start_pose(grasp, offset, rotvec)
    report = support_report(start, grasp)
    quat = start.quat_xyzw()

    print("R6 demonstration support")
    print(f"  tool offset box : {R6_START_OFFSET_TOOL_MIN} .. {R6_START_OFFSET_TOOL_MAX} cm")
    print(f"  rotation        : {R6_START_ROT_DEG_MIN} .. {R6_START_ROT_DEG_MAX} deg")
    print()
    print("Solved start pose (same frame as measured_cp)")
    print(f"  position  m : {start.p}")
    print(f"  quat  xyzw  : {quat}")
    print()
    print("Resulting geometry")
    print(f"  tool offset : {report['offset_tool_cm']} cm   "
          f"(margin to box faces {report['box_margin_cm']} cm)")
    print(f"  rotation    : {report['rotation_deg']:.2f} deg   "
          f"(margin {report['rotation_margin_deg']:.2f} deg)")
    print(f"  travel      : {report['travel_cm']:.2f} cm")
    print(f"  IN SUPPORT  : {report['in_support']}")
    if not report["in_support"]:
        print("  -> the requested offset or rotation is itself outside the box")

    if args.current_pos is not None:
        move_cm = float(np.linalg.norm(start.p - np.asarray(args.current_pos)) * 100.0)
        print()
        print(f"The arm must move {move_cm:.2f} cm to reach this start pose.")
        if args.current_quat is not None:
            here = Pose.from_pos_quat(args.current_pos, args.current_quat, 0.0)
            turn = float(np.degrees(rotation_error_rad(here, start)))
            print(f"and turn the wrist {turn:.2f} deg.")

    pos_s = " ".join(f"{v:.9f}" for v in start.p)
    quat_s = " ".join(f"{v:.9f}" for v in quat)
    grasp_pos_s = " ".join(f"{v:.9f}" for v in grasp.p)
    grasp_quat_s = " ".join(f"{v:.9f}" for v in np.asarray(args.grasp_quat))

    print()
    print("=" * 74)
    print("1. CHECK OFFLINE FIRST -- does the policy converge from this start?")
    print("=" * 74)
    print(f"""python3 tools/offline_check.py \\
  --model {args.model} \\
  --start-pos  {pos_s} \\
  --start-quat {quat_s} \\
  --goal-pos   {grasp_pos_s} \\
  --goal-quat  {grasp_quat_s} \\
  --goal-orientation explicit \\
  --controller rl --frame-mode rebase --verbose""")
    print()
    print("Compare against --controller d2 on the same pair. If the policy still")
    print("does not converge here, the geometry was never the problem.")
    print()
    print("=" * 74)
    print("2. MOVE THE ARM THERE (geometric servo, tight tolerance)")
    print("=" * 74)
    print(f"""python3 run_approach.py \\
  --arm {args.arm} \\
  --goal-pos  {pos_s} \\
  --goal-quat {quat_s} \\
  --goal-orientation explicit \\
  --controller d2 --interface move_cp --rate 2 \\
  --success-trans-cm 0.2 --success-rot-deg 2.0 \\
  --trace goto_r6_start.jsonl --execute""")
    print()
    print("=" * 74)
    print("3. RUN THE POLICY FROM THERE")
    print("=" * 74)
    print(f"""python3 run_grasp_lift.py \\
  --arm {args.arm} \\
  --grasp-pos  {grasp_pos_s} \\
  --grasp-quat {grasp_quat_s} \\
  --controller rl --model {args.model} --frame-mode rebase \\
  --jaw-approach-open-deg 45.6 \\
  --lift-sign -1 --max-path-radius-cm 12 \\
  --jaw-baseline jaw_baseline.json \\
  --grasp-gate manual --interface move_cp --rate 2 \\
  --trace live_rl_grasp_lift.jsonl --execute""")
    print()
    print("45.6 deg is the jaw angle the demonstrations started from (0.76")
    print("normalised); the jaw is part of the observation, so it is worth")
    print("matching. Close and lift stay on the servo either way.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "start_pos_m": start.p.tolist(),
            "start_quat_xyzw": quat.tolist(),
            "grasp_pos_m": grasp.p.tolist(),
            "grasp_quat_xyzw": list(map(float, args.grasp_quat)),
            "offset_tool_cm": offset.tolist(),
            "rotvec": rotvec.tolist(),
            "report": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in report.items()
            },
        }, indent=2))
        print(f"\nwrote {args.json_out}")

    return 0 if report["in_support"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
