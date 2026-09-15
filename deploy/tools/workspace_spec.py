#!/usr/bin/env python3
"""What workspace does this task actually need, and what limits it?

"Extend the workspace" is not one question, and answering the wrong one wastes
a retraining run.  A goal-conditioned policy's support, as measured from its
demonstrations, is expressed entirely in **relative** quantities:

    tool offset   R_start^T (p_goal - p_start)
    rotation      geodesic(R_start, R_goal)

Both are invariant under any rigid transform (see :mod:`..staging`).  So for
the approach leg, *any* grasp pose can be brought inside the support by
choosing the start pose -- which is what ``--stage`` does -- and the trained
workspace is not the binding constraint at all.  What binds instead is whether
that staging pose is reachable, collision-free and inside the operator's box.

This tool measures that.  Give it the real task geometry and the envelope of
needle placements you expect, and it reports, over that envelope:

  * how much of it is in support **without** staging (usually: none)
  * the staging pose each placement needs, and the box those poses span
  * how far the arm must travel to reach them, so reachability can be judged
  * what a retrain would have to cover if you refuse to stage

    python3 tools/workspace_spec.py \\
        --grasp-pos <x y z> --grasp-quat <qx qy qz qw> \\
        --current-pos <x y z> --current-quat <qx qy qz qw> \\
        --needle-xy-mm 3 3 --needle-yaw-deg 30 --contract approach_upstream

The needle envelope defaults match ``src/SurgicAI/RL/needle_reset_ranges.py``
in this repository, where the bound is documented as a *perception* limit --
the pose audit was reliable at 20 degrees of needle yaw and failed near 40 --
not a policy limit.  Retraining a policy over a wider needle yaw than
perception can estimate buys nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.contract import CONTRACTS  # noqa: E402
from surgicai_rl_deploy.frames import Pose  # noqa: E402
from surgicai_rl_deploy.staging import stage_pose_for, support_report  # noqa: E402


def needle_envelope(grasp: Pose, xy_mm, yaw_deg, samples: int, seed: int = 0):
    """Grasp poses implied by the needle sitting anywhere in its envelope.

    The needle is assumed to move in the plane it lies in: two translations and
    a yaw about the axis normal to that plane.  The gripper follows it rigidly,
    because the grasp is defined relative to the needle, not to the world.
    """
    rng = np.random.default_rng(seed)
    dx, dy = np.asarray(xy_mm, dtype=np.float64) / 1000.0
    yaw = float(np.deg2rad(yaw_deg))
    out = []
    for _ in range(samples):
        offset = np.array([rng.uniform(-dx, dx), rng.uniform(-dy, dy), 0.0])
        spin = Rotation.from_rotvec([0.0, 0.0, rng.uniform(-yaw, yaw)]).as_matrix()
        out.append(Pose(grasp.p + offset, spin @ grasp.R, grasp.jaw))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grasp-pos", nargs=3, type=float, required=True)
    ap.add_argument("--grasp-quat", nargs=4, type=float, required=True)
    ap.add_argument("--current-pos", nargs=3, type=float, required=True,
                    help="where the arm parks; staging travel is measured from here")
    ap.add_argument("--current-quat", nargs=4, type=float, required=True)
    ap.add_argument("--contract", default="approach_upstream", choices=sorted(CONTRACTS))
    ap.add_argument("--needle-xy-mm", nargs=2, type=float, default=[3.0, 3.0])
    ap.add_argument("--needle-yaw-deg", type=float, default=30.0)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-path-radius-cm", type=float, default=8.0)
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    contract = CONTRACTS[args.contract]
    grasp = Pose.from_pos_quat(args.grasp_pos, args.grasp_quat, 0.0)
    current = Pose.from_pos_quat(args.current_pos, args.current_quat, 0.0)

    grasps = needle_envelope(
        grasp, args.needle_xy_mm, args.needle_yaw_deg, args.samples, args.seed
    )

    unstaged_ok = 0
    staged_ok = 0
    staged_points = []
    travels = []
    needed_offsets = []
    needed_rotations = []

    for g in grasps:
        # (a) no staging: the arm starts where it is
        rep = support_report(current, g, contract)
        unstaged_ok += bool(rep["in_support"])
        needed_offsets.append(rep["offset_tool_cm"])
        needed_rotations.append(rep["rotation_deg"])

        # (b) staged: solve a start pose in the middle of the support
        staged = stage_pose_for(g, contract)
        staged_ok += bool(support_report(staged, g, contract)["in_support"])
        staged_points.append(staged.p)
        travels.append(float(np.linalg.norm(staged.p - current.p) * 100.0))

    staged_points = np.array(staged_points)
    travels = np.array(travels)
    needed_offsets = np.array(needed_offsets)
    needed_rotations = np.array(needed_rotations)
    n = len(grasps)

    print(f"contract        : {contract.name}")
    print(f"needle envelope : +-{args.needle_xy_mm[0]:.1f} x "
          f"+-{args.needle_xy_mm[1]:.1f} mm, +-{args.needle_yaw_deg:.0f} deg yaw"
          f"   ({n} samples)")
    print()
    print("IN SUPPORT")
    print(f"  without staging : {unstaged_ok}/{n}  ({100*unstaged_ok/n:.1f}%)")
    print(f"  with staging    : {staged_ok}/{n}  ({100*staged_ok/n:.1f}%)")
    print()
    print("STAGING POSES REQUIRED  (this is what actually has to be reachable)")
    lo, hi = staged_points.min(axis=0) * 100.0, staged_points.max(axis=0) * 100.0
    print(f"  box, cm         : x {lo[0]:+.2f}..{hi[0]:+.2f}  "
          f"y {lo[1]:+.2f}..{hi[1]:+.2f}  z {lo[2]:+.2f}..{hi[2]:+.2f}")
    print(f"  span, cm        : {np.round(hi - lo, 2).tolist()}")
    print(f"  travel from the parked pose, cm: median {np.median(travels):.2f}, "
          f"max {travels.max():.2f}")
    over = int((travels > args.max_path_radius_cm).sum())
    print(f"  beyond the {args.max_path_radius_cm:.0f} cm path-radius guard: "
          f"{over}/{n}"
          + ("" if over == 0 else "   <-- these need a planned motion, not a servo"))
    print()
    print("IF YOU REFUSE TO STAGE, a retrain would have to cover")
    print(f"  tool offset x   : {needed_offsets[:,0].min():+.2f} .. "
          f"{needed_offsets[:,0].max():+.2f} cm   "
          f"(trained {contract.start_offset_tool_min[0]:+.2f} .. "
          f"{contract.start_offset_tool_max[0]:+.2f})")
    print(f"  tool offset y   : {needed_offsets[:,1].min():+.2f} .. "
          f"{needed_offsets[:,1].max():+.2f} cm   "
          f"(trained {contract.start_offset_tool_min[1]:+.2f} .. "
          f"{contract.start_offset_tool_max[1]:+.2f})")
    print(f"  tool offset z   : {needed_offsets[:,2].min():+.2f} .. "
          f"{needed_offsets[:,2].max():+.2f} cm   "
          f"(trained {contract.start_offset_tool_min[2]:+.2f} .. "
          f"{contract.start_offset_tool_max[2]:+.2f})")
    print(f"  rotation        : {needed_rotations.min():.1f} .. "
          f"{needed_rotations.max():.1f} deg   "
          f"(trained {contract.start_rot_deg_min:.1f} .. "
          f"{contract.start_rot_deg_max:.1f})")
    print()

    if staged_ok == n and over == 0:
        print("Staging covers the whole envelope and every staging pose is inside")
        print("the reachability guard. The trained workspace is NOT what limits")
        print("this task; perception and reachability are. Retraining for a wider")
        print("support would not change what this deployment can do.")
    elif staged_ok == n:
        print("Staging covers the whole envelope, but some staging poses are")
        print("further than the path-radius guard allows. Either park the arm")
        print("closer, raise the guard once you have checked the arm covers it,")
        print("or plan that motion properly instead of servoing it.")
    else:
        print("Staging does not cover the whole envelope, which should not happen")
        print("for a support defined purely by relative geometry. Check that the")
        print("contract's rotation range contains the mean rotation vector's own")
        print("magnitude, which is what stage_pose_for() adopts by default.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "contract": contract.name,
            "samples": n,
            "in_support_unstaged": unstaged_ok,
            "in_support_staged": staged_ok,
            "staging_box_cm": {"min": (lo).tolist(), "max": (hi).tolist()},
            "travel_cm": {"median": float(np.median(travels)),
                          "max": float(travels.max())},
            "beyond_path_radius": over,
            "required_offset_cm": {
                "min": needed_offsets.min(axis=0).tolist(),
                "max": needed_offsets.max(axis=0).tolist()},
            "required_rotation_deg": [float(needed_rotations.min()),
                                      float(needed_rotations.max())],
        }, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
