"""Putting a policy leg inside its own training support, in the robot's frame.

Why this is possible at all
---------------------------
A policy's support, as measured from its demonstrations, is expressed in two
quantities:

    tool offset   R_start^T (p_goal - p_start)      inside the demonstration box
    rotation      geodesic(R_start, R_goal)         inside the observed range

and **both are invariant under the frame bridge**.  For any rigid ``X``,

    R_start_policy^T (p_goal_policy - p_start_policy)
        = (X.R R_start)^T X.R (p_goal - p_start)
        = R_start^T (p_goal - p_start)

and a geodesic between two rotations is unchanged by a common pre-rotation.
So the support can be satisfied by choosing the start pose in the robot's own
frame, with no reference to the training frame at all, and ``frame_mode
"rebase"`` then lines the absolute goal up on top of the trained one.

Given a fixed goal pose the solution is direct, not a search:

    R_start = R_goal * Rel^-1        Rel = the start->goal rotation to adopt
    p_start = p_goal - R_start * offset

What this does not tell you
---------------------------
That the policy will work, that the arm can adopt the pose, or that it can get
there without hitting anything.  It removes one excuse -- being out of
distribution -- and nothing else.  :mod:`.feasibility` checks the rest, and
``tools/replay_demos.py`` says whether the policy converges at all.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import APPROACH_R6, SUPPORT_EPS_CM, SubtaskContract
from .frames import Pose, rotation_error_rad


def solve_start_pose(goal: Pose, offset_tool_cm, rotvec) -> Pose:
    """The start pose whose tool offset and start->goal rotation are as given."""
    rel = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    # R_goal = R_start @ rel   ->   R_start = R_goal @ rel^-1
    R_start = goal.R @ rel.T
    p_start = goal.p - R_start @ (np.asarray(offset_tool_cm, dtype=np.float64) / 100.0)
    return Pose(p_start, R_start, goal.jaw)


def rotvec_for(contract: SubtaskContract, rotation_deg: Optional[float] = None):
    """The mean start->goal rotation, optionally rescaled to a given angle."""
    rotvec = np.asarray(contract.start_to_goal_rotvec_mean, dtype=np.float64)
    if rotation_deg is None:
        return rotvec
    norm = float(np.linalg.norm(rotvec))
    if norm <= 1e-12:
        raise ValueError(f"{contract.name} has no measured start->goal rotation")
    return rotvec / norm * float(np.deg2rad(rotation_deg))


def stage_pose_for(
    goal: Pose,
    contract: SubtaskContract,
    *,
    offset_tool_cm=None,
    rotation_deg: Optional[float] = None,
    jaw: Optional[float] = None,
) -> Pose:
    """The staging pose for one policy leg: middle of the box by default."""
    offset = (
        contract.start_offset_tool_mean if offset_tool_cm is None else offset_tool_cm
    )
    pose = solve_start_pose(goal, offset, rotvec_for(contract, rotation_deg))
    if jaw is None:
        jaw = float(contract.demo_start_jaw)
    return Pose(pose.p, pose.R, float(jaw))


def support_report(start: Pose, goal: Pose,
                   contract: SubtaskContract = APPROACH_R6) -> dict:
    """How far inside (or outside) the contract's support this pair sits."""
    offset = start.R.T @ ((goal.p - start.p) * 100.0)
    rot_deg = float(np.degrees(rotation_error_rad(start, goal)))
    lo = np.asarray(contract.start_offset_tool_min, dtype=np.float64)
    hi = np.asarray(contract.start_offset_tool_max, dtype=np.float64)
    inside_box = bool(
        np.all(offset >= lo - SUPPORT_EPS_CM) and np.all(offset <= hi + SUPPORT_EPS_CM)
    )
    inside_rot = bool(
        contract.start_rot_deg_min <= rot_deg <= contract.start_rot_deg_max
    )
    return {
        "contract": contract.name,
        "offset_tool_cm": offset,
        "rotation_deg": rot_deg,
        "travel_cm": float(np.linalg.norm(goal.p - start.p) * 100.0),
        "inside_box": inside_box,
        "inside_rotation": inside_rot,
        "in_support": inside_box and inside_rot,
        "box_margin_cm": np.minimum(offset - lo, hi - offset),
        "rotation_margin_deg": float(
            min(rot_deg - contract.start_rot_deg_min,
                contract.start_rot_deg_max - rot_deg)
        ),
        "reasons": contract.in_support(offset, rot_deg),
    }
