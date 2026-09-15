"""Solving for a start pose inside the R6 demonstration support.

The support constraints are invariant under the frame bridge, so they can be
satisfied in the robot's own frame. These tests pin that invariance, because it
is the whole reason the solve is legitimate: if the tool offset or the rotation
depended on the bridge, a pose that looks in-support here would not be
in-support once the policy saw it.
"""

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from plan_r6_start import solve_start_pose, support_report  # noqa: E402

from surgicai_rl_deploy.contract import (  # noqa: E402
    R6_START_OFFSET_TOOL_MAX,
    R6_START_OFFSET_TOOL_MEAN,
    R6_START_OFFSET_TOOL_MIN,
    R6_START_ROT_DEG_MAX,
    R6_START_ROT_DEG_MIN,
    R6_START_TO_GOAL_ROTVEC_MEAN,
)
from surgicai_rl_deploy.frames import FrameBridge, Pose  # noqa: E402
from surgicai_rl_deploy.loop import ApproachLoop, LoopConfig, SafetyLimits  # noqa: E402

from conftest import REAL_GOAL_POS  # noqa: E402

GRASP_POS = [0.15275556516262584, 0.06227624380447951, 0.07600460590461247]
GRASP_QUAT = [
    -0.1847502875667516, 0.3316999709489106,
    -0.5750784982987905, 0.7246565954373855,
]


@pytest.fixture
def grasp():
    return Pose.from_pos_quat(GRASP_POS, GRASP_QUAT, 0.0)


@pytest.fixture
def solved(grasp):
    return solve_start_pose(
        grasp, R6_START_OFFSET_TOOL_MEAN, R6_START_TO_GOAL_ROTVEC_MEAN
    )


def test_the_solution_lands_in_support(solved, grasp):
    report = support_report(solved, grasp)
    assert report["in_support"] is True
    assert report["inside_box"] and report["inside_rotation"]


def test_the_offset_is_exactly_what_was_asked_for(solved, grasp):
    report = support_report(solved, grasp)
    np.testing.assert_allclose(
        report["offset_tool_cm"], R6_START_OFFSET_TOOL_MEAN, atol=1e-9
    )


def test_the_rotation_is_the_demonstrations_mean(solved, grasp):
    expected = float(np.degrees(np.linalg.norm(R6_START_TO_GOAL_ROTVEC_MEAN)))
    assert support_report(solved, grasp)["rotation_deg"] == pytest.approx(
        expected, abs=1e-6
    )
    assert R6_START_ROT_DEG_MIN < expected < R6_START_ROT_DEG_MAX


def test_the_mean_offset_sits_comfortably_inside_the_box(solved, grasp):
    """Not just inside -- far enough from every face that a millimetre of
    positioning error does not push the episode out of support."""
    margin = support_report(solved, grasp)["box_margin_cm"]
    assert np.all(margin > 1.0)


def test_travel_resembles_the_demonstrations(solved, grasp):
    # demonstration median travel is 4.12 cm
    assert 3.0 < support_report(solved, grasp)["travel_cm"] < 5.0


# --- the invariance the solve relies on -----------------------------------
@pytest.mark.parametrize("mode", ["rebase", "translate", "identity"])
def test_support_metrics_survive_the_frame_bridge(solved, grasp, mode):
    """A rigid transform cancels in both the tool offset and the geodesic, so
    what is in-support in the ECM frame is in-support in the policy frame."""
    from surgicai_rl_deploy.contract import R6_TRAINED_GOAL_VEC7

    bridge = FrameBridge.build(mode, grasp, Pose.from_vec7(R6_TRAINED_GOAL_VEC7))
    robot = support_report(solved, grasp)
    policy = support_report(bridge.to_policy(solved), bridge.to_policy(grasp))
    np.testing.assert_allclose(
        robot["offset_tool_cm"], policy["offset_tool_cm"], atol=1e-9
    )
    assert robot["rotation_deg"] == pytest.approx(policy["rotation_deg"], abs=1e-9)


def test_the_loop_agrees_that_it_is_in_distribution(solved, grasp):
    """The solve must agree with ApproachLoop.distribution_report, which is
    what actually gates the run."""
    loop = ApproachLoop(
        _NullController(),
        LoopConfig(
            frame_mode="rebase",
            goal_orientation="explicit",
            goal_quat_xyzw=tuple(GRASP_QUAT),
        ),
        SafetyLimits(),
    )
    report = loop.begin(solved, grasp.p)
    assert report["in_distribution"] is True
    assert report["out_of_distribution"] == []


def test_the_original_start_pose_was_not_in_distribution():
    """Sanity: the pose the arm was actually at fails, which is why the solve
    is needed at all."""
    start = Pose.from_pos_quat(
        [0.08691037127952378, 0.08040837470034838, 0.035698267165496275],
        [-0.18525206922173656, 0.3186118737966743,
         -0.5784831033942791, 0.7276849894096765],
        0.0,
    )
    grasp = Pose.from_pos_quat(GRASP_POS, GRASP_QUAT, 0.0)
    assert support_report(start, grasp)["in_support"] is False


# --- knobs ----------------------------------------------------------------
@pytest.mark.parametrize("angle", [26.0, 57.13, 100.0])
def test_a_rescaled_rotation_stays_in_support(grasp, angle):
    rotvec = np.asarray(R6_START_TO_GOAL_ROTVEC_MEAN, dtype=np.float64)
    rotvec = rotvec / np.linalg.norm(rotvec) * np.deg2rad(angle)
    report = support_report(solve_start_pose(grasp, R6_START_OFFSET_TOOL_MEAN, rotvec), grasp)
    assert report["rotation_deg"] == pytest.approx(angle, abs=1e-6)
    assert report["in_support"] is True


@pytest.mark.parametrize("angle", [10.0, 120.0])
def test_a_rotation_outside_the_range_is_reported_as_such(grasp, angle):
    rotvec = np.asarray(R6_START_TO_GOAL_ROTVEC_MEAN, dtype=np.float64)
    rotvec = rotvec / np.linalg.norm(rotvec) * np.deg2rad(angle)
    report = support_report(solve_start_pose(grasp, R6_START_OFFSET_TOOL_MEAN, rotvec), grasp)
    assert report["inside_rotation"] is False
    assert report["in_support"] is False


def test_an_offset_outside_the_box_is_reported_as_such(grasp):
    outside = np.array([10.0, 10.0, 10.0])
    report = support_report(solve_start_pose(grasp, outside, R6_START_TO_GOAL_ROTVEC_MEAN), grasp)
    assert report["inside_box"] is False
    assert report["in_support"] is False


def test_the_box_corners_are_reachable_by_the_solve(grasp):
    for corner in (R6_START_OFFSET_TOOL_MIN, R6_START_OFFSET_TOOL_MAX):
        report = support_report(
            solve_start_pose(grasp, corner, R6_START_TO_GOAL_ROTVEC_MEAN), grasp
        )
        np.testing.assert_allclose(report["offset_tool_cm"], corner, atol=1e-9)
        assert report["inside_box"] is True


def test_the_solve_is_exact_for_any_grasp_orientation():
    """No dependence on where the grasp happens to point."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        quat = Rotation.random(random_state=int(rng.integers(1 << 30))).as_quat()
        grasp = Pose.from_pos_quat(rng.normal(0, 0.1, 3), quat, 0.0)
        start = solve_start_pose(
            grasp, R6_START_OFFSET_TOOL_MEAN, R6_START_TO_GOAL_ROTVEC_MEAN
        )
        assert support_report(start, grasp)["in_support"] is True


class _NullController:
    name = "null"

    def act(self, obs):
        return np.zeros(7)

    def describe(self):
        return "null"
