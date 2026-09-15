"""The 2*pi branch of the RPY coordinates.

A rotation matrix is branch-invariant; the RPY triple describing it is not.
scipy's as_euler("xyz") returns roll and yaw in [-pi, pi], while the SurgicAI
environments integrate RPY as free state and never re-canonicalize. In the
upstream Approach checkpoint 100% of desired-goal rolls and 85% of achieved
rolls lie OUTSIDE [-pi, pi], so re-deriving RPY from a matrix moved three of
the twenty-one observation dimensions onto a branch no policy was trained on.

That defect made both released checkpoints diverge to ~130 deg of orientation
error while the geometric servo was unaffected -- the servo only uses relative
rotation, where a common 2*pi offset cancels. These tests exist so it cannot
come back.
"""

import numpy as np
import pytest

from surgicai_rl_deploy.frames import Pose, unwrap_rpy_to
from surgicai_rl_deploy.loop import ApproachLoop, LoopConfig, SafetyLimits

# the upstream Approach checkpoint's own trained goal, roll outside +-pi
UPSTREAM_GOAL_RPY = np.array([-3.774, 0.497, 1.317])


def test_a_matrix_round_trip_moves_roll_by_two_pi():
    """The defect itself, pinned as a fact about scipy, not an opinion."""
    canonical = Pose.from_vec7(np.r_[0, 0, 0, UPSTREAM_GOAL_RPY, 0]).to_vec7()[3:6]
    assert not np.allclose(canonical, UPSTREAM_GOAL_RPY)
    assert canonical[0] == pytest.approx(UPSTREAM_GOAL_RPY[0] + 2 * np.pi)


def test_unwrap_recovers_the_training_branch():
    canonical = Pose.from_vec7(np.r_[0, 0, 0, UPSTREAM_GOAL_RPY, 0]).to_vec7()[3:6]
    np.testing.assert_allclose(
        unwrap_rpy_to(canonical, UPSTREAM_GOAL_RPY), UPSTREAM_GOAL_RPY, atol=1e-12
    )


def test_unwrap_preserves_the_rotation_itself():
    """Unwrapping changes coordinates, never the physical orientation."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        rpy = rng.uniform(-np.pi, np.pi, 3)
        reference = rpy + 2 * np.pi * rng.integers(-2, 3, 3)
        moved = unwrap_rpy_to(rpy, reference)
        a = Pose.from_vec7(np.r_[0, 0, 0, rpy, 0]).R
        b = Pose.from_vec7(np.r_[0, 0, 0, moved, 0]).R
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_unwrap_is_idempotent():
    out = unwrap_rpy_to(UPSTREAM_GOAL_RPY, UPSTREAM_GOAL_RPY)
    np.testing.assert_allclose(out, UPSTREAM_GOAL_RPY, atol=1e-12)


@pytest.mark.parametrize("k", [-2, -1, 1, 2])
def test_any_branch_is_pulled_back(k):
    shifted = UPSTREAM_GOAL_RPY + 2 * np.pi * k * np.array([1, 0, 1])
    np.testing.assert_allclose(
        unwrap_rpy_to(shifted, UPSTREAM_GOAL_RPY), UPSTREAM_GOAL_RPY, atol=1e-9
    )


class _Recorder:
    """Captures the observation the network would actually receive."""

    name = "recorder"

    def __init__(self):
        self.seen = []

    def act(self, obs):
        self.seen.append({k: np.array(v) for k, v in obs.items()})
        return np.zeros(7)

    def describe(self):
        return "recorder"


def _run_one(unwrap, goal_rpy):
    goal_vec7 = np.r_[-0.03253, 0.01311, -0.11986, goal_rpy, 0.0]
    start_vec7 = goal_vec7.copy()
    start_vec7[:3] += [0.005, 0.01, 0.02]
    start_vec7[3:6] += [0.2, -0.1, 0.3]

    recorder = _Recorder()
    loop = ApproachLoop(
        recorder,
        LoopConfig(
            frame_mode="identity", goal_orientation="explicit",
            goal_quat_xyzw=tuple(Pose.from_vec7(np.r_[goal_vec7[:6], 0.0]).quat_xyzw()),
            goal_rpy_train=tuple(goal_rpy), unwrap_rpy=unwrap,
        ),
        SafetyLimits(),
    )
    loop.begin(Pose.from_vec7(start_vec7), goal_vec7[:3])
    loop.step(Pose.from_vec7(start_vec7))
    return recorder.seen[0], start_vec7


def test_the_network_sees_the_training_branch_when_unwrapping():
    obs, start = _run_one(True, UPSTREAM_GOAL_RPY)
    # desired_goal roll must be the training value, not its canonical alias
    assert obs["desired_goal"][3] == pytest.approx(UPSTREAM_GOAL_RPY[0], abs=1e-5)
    # achieved roll must sit on the same branch, i.e. near it, not 2*pi away
    assert abs(obs["achieved_goal"][3] - UPSTREAM_GOAL_RPY[0]) < np.pi


def test_without_unwrapping_the_network_sees_the_wrong_branch():
    obs, _ = _run_one(False, UPSTREAM_GOAL_RPY)
    assert abs(obs["desired_goal"][3] - UPSTREAM_GOAL_RPY[0]) == pytest.approx(
        2 * np.pi, abs=1e-4
    )


def test_the_delta_block_is_small_when_unwrapped():
    """The roll delta should reflect the real error, not a 2*pi artefact."""
    obs, _ = _run_one(True, UPSTREAM_GOAL_RPY)
    assert abs(obs["observation"][17]) < np.pi
    stale, _ = _run_one(False, UPSTREAM_GOAL_RPY)
    assert abs(stale["observation"][17]) < np.pi  # delta cancels...
    # ...but both absolute blocks are on the wrong branch, which is what breaks
    assert abs(stale["achieved_goal"][3] - UPSTREAM_GOAL_RPY[0]) > np.pi


def test_a_canonical_goal_is_unaffected():
    """Checkpoints whose goal already sits inside +-pi must behave identically."""
    canonical_goal = np.array([-3.100, 0.902, 2.397])  # R6's trained goal
    on, _ = _run_one(True, canonical_goal)
    off, _ = _run_one(False, canonical_goal)
    np.testing.assert_allclose(
        on["desired_goal"], off["desired_goal"], atol=1e-5
    )


def test_unwrapping_is_on_by_default():
    assert LoopConfig().unwrap_rpy is True
