"""Episode geometry: the grasp pose, and the 1.5 cm lift off it."""

import numpy as np
import pytest

from surgicai_rl_deploy.frames import rotation_error_rad
from surgicai_rl_deploy.plan import LiftSpec, build_plan

from conftest import REAL_GOAL_POS


def test_lift_defaults_to_the_deployment_contract():
    spec = LiftSpec()
    assert spec.distance_m == pytest.approx(0.015)
    assert spec.axis == "z"
    assert spec.frame == "robot"
    assert spec.explicit is False  # nobody has confirmed the sign yet


@pytest.mark.parametrize("bad", [0.0, -0.01, float("nan")])
def test_lift_distance_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError):
        LiftSpec(distance_m=bad)


def test_lift_distance_ceiling():
    with pytest.raises(ValueError, match="ceiling"):
        LiftSpec(distance_m=0.10)


@pytest.mark.parametrize("bad", ["w", "Z", ""])
def test_lift_axis_validated(bad):
    with pytest.raises(ValueError):
        LiftSpec(axis=bad)


def test_lift_sign_validated():
    with pytest.raises(ValueError):
        LiftSpec(sign=0)


def test_robot_frame_lift_is_axis_aligned(plan):
    """1.5 cm along -z of the robot frame, and nothing else moves."""
    delta = plan.lifted.p - plan.grasp.p
    assert delta[0] == pytest.approx(0.0, abs=1e-12)
    assert delta[1] == pytest.approx(0.0, abs=1e-12)
    assert delta[2] == pytest.approx(-0.015)
    assert plan.lift_travel_cm == pytest.approx(1.5)


def test_lift_preserves_the_grasp_orientation(plan):
    assert rotation_error_rad(plan.grasp, plan.lifted) == pytest.approx(0.0, abs=1e-12)


def test_tool_frame_lift_follows_the_gripper(start_pose):
    spec = LiftSpec(axis="z", sign=-1, frame="tool", explicit=True)
    plan = build_plan(start_pose, REAL_GOAL_POS, lift=spec)
    expected = plan.grasp.R @ np.array([0.0, 0.0, -0.015])
    np.testing.assert_allclose(plan.lifted.p - plan.grasp.p, expected, atol=1e-12)
    # and it is genuinely different from the robot-frame lift
    assert not np.allclose(expected, [0.0, 0.0, -0.015], atol=1e-6)


def test_hold_keeps_the_start_orientation(start_pose):
    plan = build_plan(start_pose, REAL_GOAL_POS, goal_orientation="hold")
    assert rotation_error_rad(start_pose, plan.grasp) == pytest.approx(0.0, abs=1e-12)
    assert plan.approach_rotation_deg == pytest.approx(0.0, abs=1e-9)


def test_explicit_orientation_is_used(start_pose):
    quat = (0.0, 0.0, 0.0, 1.0)
    plan = build_plan(
        start_pose, REAL_GOAL_POS, goal_orientation="explicit", goal_quat_xyzw=quat
    )
    np.testing.assert_allclose(plan.grasp.R, np.eye(3), atol=1e-12)


def test_explicit_orientation_needs_a_quaternion(start_pose):
    with pytest.raises(ValueError, match="needs goal_quat_xyzw"):
        build_plan(start_pose, REAL_GOAL_POS, goal_orientation="explicit")


def test_trained_relative_rotates_the_wrist(start_pose):
    """The only mode that puts the policy's rotation channels back in support."""
    plan = build_plan(start_pose, REAL_GOAL_POS, goal_orientation="trained_relative")
    assert 25.0 < plan.approach_rotation_deg < 101.0


def test_approach_travel_matches_the_recorded_task(plan):
    assert plan.approach_travel_cm == pytest.approx(3.16, abs=0.02)


def test_path_radius_covers_every_waypoint(plan):
    assert plan.path_radius_cm() >= plan.approach_travel_cm


def test_bounding_box_contains_all_waypoints(plan):
    low, high = plan.bounding_box_m(pad_cm=0.0)
    for p in (plan.start.p, plan.grasp.p, plan.lifted.p):
        assert np.all(p >= low - 1e-12) and np.all(p <= high + 1e-12)


def test_plan_dict_records_whether_a_human_confirmed_the_sign(start_pose):
    unconfirmed = build_plan(start_pose, REAL_GOAL_POS, lift=LiftSpec())
    assert unconfirmed.as_dict()["lift"]["operator_confirmed_sign"] is False


def test_grasp_position_must_be_finite(start_pose):
    with pytest.raises(ValueError, match="finite"):
        build_plan(start_pose, [0.0, float("nan"), 0.0])


def test_jaw_rides_along_as_the_normalised_value(plan, jaw_cal):
    assert plan.grasp.jaw == pytest.approx(jaw_cal.normalise(jaw_cal.approach_open_rad))
    assert plan.lifted.jaw == pytest.approx(jaw_cal.normalise(jaw_cal.grip_rad))
