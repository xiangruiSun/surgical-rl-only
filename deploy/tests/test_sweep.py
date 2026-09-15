"""The support sweep: sampling, and the episode runner it scores with.

The RL half needs the checkpoint and is not exercised here. What is exercised
is everything that decides whether a failure can be blamed on the policy: that
every sampled start really is in support, and that the servo converges from all
of them, so the geometry is never the confound.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from plan_r6_start import solve_start_pose, support_report  # noqa: E402
from sweep_r6_support import run_episode, sample_offsets  # noqa: E402

from surgicai_rl_deploy.contract import (  # noqa: E402
    R6_START_OFFSET_TOOL_MAX,
    R6_START_OFFSET_TOOL_MIN,
    R6_START_TO_GOAL_ROTVEC_MEAN,
)
from surgicai_rl_deploy.controllers import D2Controller  # noqa: E402
from surgicai_rl_deploy.frames import Pose  # noqa: E402

GRASP = Pose.from_pos_quat(
    [0.152755565, 0.062276244, 0.076004606],
    [-0.184750288, 0.331699971, -0.575078498, 0.724656595],
    0.0,
)
UNIT = np.asarray(R6_START_TO_GOAL_ROTVEC_MEAN) / np.linalg.norm(
    R6_START_TO_GOAL_ROTVEC_MEAN
)


@pytest.mark.parametrize("grid", [1, 2, 3])
def test_sample_count(grid):
    assert len(sample_offsets(grid)) == max(1, grid) ** 3 if grid > 1 else 1


def test_every_sample_is_strictly_inside_the_box():
    """Samples are inset, so none sits on a face where membership is arguable."""
    for offset in sample_offsets(3):
        assert np.all(offset > R6_START_OFFSET_TOOL_MIN)
        assert np.all(offset < R6_START_OFFSET_TOOL_MAX)


@pytest.mark.parametrize("rot_deg", [25.7, 57.1, 100.2])
def test_every_sampled_start_is_in_support(rot_deg):
    for offset in sample_offsets(2):
        start = solve_start_pose(GRASP, offset, UNIT * np.deg2rad(rot_deg))
        assert support_report(start, GRASP)["in_support"] is True


def test_the_servo_converges_from_every_sampled_start():
    """If the servo failed here the sweep would be measuring the geometry, not
    the policy."""
    failures = []
    for offset in sample_offsets(2):
        for rot_deg in (25.7, 57.1, 100.2):
            start = solve_start_pose(GRASP, offset, UNIT * np.deg2rad(rot_deg))
            result = run_episode(
                D2Controller(staged=True), start, GRASP,
                goal_jaw="closed", max_steps=200,
                success_trans_cm=1.0, success_rot_deg=10.0, frame_mode="rebase",
            )
            if not result["success"]:
                failures.append((offset.tolist(), rot_deg, result))
    assert not failures, failures


def test_run_episode_reports_the_closest_approach():
    start = solve_start_pose(GRASP, [0.96, 2.445, 2.661], UNIT * np.deg2rad(57.1))
    result = run_episode(
        D2Controller(staged=True), start, GRASP,
        goal_jaw="closed", max_steps=200,
        success_trans_cm=1.0, success_rot_deg=10.0, frame_mode="rebase",
    )
    assert result["closest_trans_cm"] <= result["final_trans_cm"] + 1e-9
    assert result["steps"] >= 1


def test_a_hopeless_step_budget_reports_max_steps():
    start = solve_start_pose(GRASP, [0.96, 2.445, 2.661], UNIT * np.deg2rad(57.1))
    result = run_episode(
        D2Controller(staged=True), start, GRASP,
        goal_jaw="closed", max_steps=3,
        success_trans_cm=0.01, success_rot_deg=0.1, frame_mode="rebase",
    )
    assert result["success"] is False
    assert result["outcome"] == "max_steps"
