"""Builds the exact Dict observation the checkpoints expect.

Mirrors ``RL/utils/gym_manager.py :: update_observation`` +
``normalize_observation`` with no behavioural change.
"""

from __future__ import annotations

import numpy as np

from .contract import GOAL_SCALE
from .frames import Pose


def build_observation(
    current_vec7_raw,
    goal_vec7_raw,
    *,
    align_rpy_branch: bool = False,
    wrap_rpy_delta: bool = False,
) -> dict:
    """Return ``{"observation", "achieved_goal", "desired_goal"}``.

    Parameters
    ----------
    current_vec7_raw, goal_vec7_raw
        RAW 7-vectors, ``[x_m, y_m, z_m, roll, pitch, yaw, jaw_norm]``, both
        already expressed in the *policy* frame.
    align_rpy_branch
        Snap the achieved RPY onto the same 2*pi branch as the desired RPY
        (env flag ``align_obs_rpy_branch``).  Off by default, matching the
        training default.
    wrap_rpy_delta
        Wrap only the RPY delta block into [-pi, pi] (env flag
        ``wrap_obs_rpy_delta``).  Off by default.
    """
    current = np.asarray(current_vec7_raw, dtype=np.float32).reshape(7).copy()
    goal = np.asarray(goal_vec7_raw, dtype=np.float32).reshape(7).copy()

    if align_rpy_branch:
        current[3:6] = current[3:6] + 2.0 * np.pi * np.round(
            (goal[3:6] - current[3:6]) / (2.0 * np.pi)
        )
        delta = goal - current
    elif wrap_rpy_delta:
        delta = goal - current
        delta[3:6] = (delta[3:6] + np.pi) % (2.0 * np.pi) - np.pi
    else:
        delta = goal - current

    achieved = current * GOAL_SCALE
    desired = goal * GOAL_SCALE
    observation = np.concatenate([achieved, desired, delta * GOAL_SCALE]).astype(
        np.float32
    )
    return {
        "observation": observation,
        "achieved_goal": achieved.astype(np.float32),
        "desired_goal": desired.astype(np.float32),
    }


def observation_from_poses(current: Pose, goal: Pose, **kwargs) -> dict:
    return build_observation(current.to_vec7(), goal.to_vec7(), **kwargs)
