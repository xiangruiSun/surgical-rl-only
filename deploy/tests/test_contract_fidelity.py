"""The three contract defects, pinned so none of them can come back.

Each was found by reading SurgicAI's own sources rather than by guessing, and
each is measurable: replaying the upstream Approach checkpoint from its own
demonstration starts, in the training frame, through this very loop.

1. roll branch      canonical RPY  ->   0/25      bound to (-2pi, 0] -> 25/25
2. action scale     0.5 mm / 2 deg -> 7/20 (35%)  1.0 mm / 3 deg -> 19/20 (95%)
3. observation      measured pose, arm tracking 0.3 -> 6/25 (24%)
                    integrated command, same arm   -> 25/25 (100%)

The numbers above come from tools/replay_demos.py, which needs the checkpoints.
These tests need nothing: they pin the *mechanisms* those measurements found.
"""

import numpy as np
import pytest

from surgicai_rl_deploy.contract import (
    APPROACH_R6,
    APPROACH_UPSTREAM,
    CHECKPOINT_CONTRACTS,
    CONTRACTS,
    PLACE_UPSTREAM,
    contract_for_digest,
)
from surgicai_rl_deploy.frames import Pose, bound_roll, vec7_bound
from surgicai_rl_deploy.loop import ApproachLoop, LoopConfig, SafetyLimits


# ======================================================================
# 1. the roll branch
# ======================================================================
def surgicai_reference(roll):
    """Transcribed from RL/subtask_env.py :: Frame2Vec(bound=True)."""
    if roll <= np.deg2rad(-360):
        return roll + 2 * np.pi
    elif roll > np.deg2rad(0):
        return roll - 2 * np.pi
    return roll


@pytest.mark.parametrize(
    "roll",
    [0.0, -0.001, 0.001, np.pi, -np.pi, 3.0, -3.0, -6.0, -2 * np.pi, 1e-9, -1e-9],
)
def test_bound_roll_is_the_surgicai_rule(roll):
    got = bound_roll(np.array([roll, 0.3, -0.4]))
    assert got[0] == pytest.approx(surgicai_reference(roll), abs=1e-12)
    # pitch and yaw are SurgicAI's business, not ours: untouched
    assert got[1] == pytest.approx(0.3)
    assert got[2] == pytest.approx(-0.4)


def test_bound_roll_lands_in_the_training_interval():
    rng = np.random.default_rng(3)
    rolls = rng.uniform(-np.pi, np.pi, 500)
    out = bound_roll(np.stack([rolls, np.zeros(500), np.zeros(500)], axis=1))[:, 0]
    assert np.all(out > -2 * np.pi - 1e-12)
    assert np.all(out <= 0.0 + 1e-12)


def test_bound_roll_never_changes_the_orientation():
    rng = np.random.default_rng(4)
    for _ in range(100):
        rpy = rng.uniform(-np.pi, np.pi, 3)
        a = Pose.from_vec7(np.r_[0, 0, 0, rpy, 0]).R
        b = Pose.from_vec7(np.r_[0, 0, 0, bound_roll(rpy), 0]).R
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_bound_roll_recovers_a_stored_training_vector():
    """The upstream Approach goal, whose roll sits outside +-pi."""
    stored = np.array([-0.032556, 0.011749, -0.118730, -3.775371, 0.493273, 1.311746])
    round_tripped = Pose.from_vec7(np.r_[stored, 0.0]).to_vec7()[3:6]
    assert round_tripped[0] == pytest.approx(stored[3] + 2 * np.pi, abs=1e-9)
    np.testing.assert_allclose(bound_roll(round_tripped), stored[3:6], atol=1e-6)


def test_vec7_bound_keeps_position_and_jaw():
    pose = Pose.from_vec7([0.1, -0.2, 0.3, 2.0, 0.4, -1.0, 0.75])
    out = vec7_bound(pose)
    np.testing.assert_allclose(out[:3], [0.1, -0.2, 0.3], atol=1e-12)
    assert out[6] == pytest.approx(0.75)
    assert out[3] <= 0.0


def test_bound_roll_is_idempotent():
    rng = np.random.default_rng(5)
    rpy = rng.uniform(-np.pi, np.pi, (50, 3))
    once = bound_roll(rpy)
    np.testing.assert_allclose(bound_roll(once), once, atol=1e-15)


# ======================================================================
# 2. the two action scales
# ======================================================================
def test_the_acting_scale_is_not_the_demonstration_scale():
    """The distinction that cost this project a fortnight of wrong results."""
    for contract in (APPROACH_UPSTREAM, PLACE_UPSTREAM):
        assert contract.step_size[0] == pytest.approx(1.0e-3)
        assert np.degrees(contract.step_size[3]) == pytest.approx(3.0)
        assert contract.demo_step_size[0] == pytest.approx(0.5e-3)
        assert np.degrees(contract.demo_step_size[3]) == pytest.approx(2.0)
        assert not np.allclose(contract.step_size, contract.demo_step_size)


def test_upstream_budget_matches_the_training_scripts():
    # RL_training_online.py and Model_evaluation.py both say max_episode_steps=300
    assert APPROACH_UPSTREAM.max_steps == 300
    assert PLACE_UPSTREAM.max_steps == 300


def test_certified_tolerance_is_the_env_class_default():
    """0.5 cm / 30 deg is what the published success rates were measured at."""
    for contract in (APPROACH_UPSTREAM, PLACE_UPSTREAM):
        assert contract.success_trans_cm == pytest.approx(0.5)
        assert np.degrees(contract.success_rot_rad) == pytest.approx(30.0)
        # and the tighter Env_info pair is kept, because 30 deg of needle angle
        # is not a placement
        assert np.degrees(contract.env_info_rot_rad) == pytest.approx(10.0)


def test_r6_is_flagged_unverified():
    assert APPROACH_R6.step_size_verified is False
    assert APPROACH_UPSTREAM.step_size_verified is True
    assert "UNVERIFIED" in APPROACH_R6.describe()


def test_checkpoints_resolve_to_contracts():
    for digest, key in CHECKPOINT_CONTRACTS.items():
        assert key in CONTRACTS
        assert contract_for_digest(digest) is CONTRACTS[key]
    assert contract_for_digest("not a digest") is None


def test_loop_config_from_contract_carries_the_acting_scale():
    cfg = LoopConfig.from_contract(APPROACH_UPSTREAM)
    np.testing.assert_allclose(cfg.step_size, APPROACH_UPSTREAM.step_size)
    assert cfg.max_steps == 300
    assert cfg.policy_jaw_start == pytest.approx(0.80)
    assert cfg.goal_jaw == "0.0"


def test_in_support_reports_each_offending_axis():
    reasons = APPROACH_UPSTREAM.in_support([-9.0, 2.0, 3.0], 70.0)
    assert len(reasons) == 1 and "tool-x" in reasons[0]
    assert APPROACH_UPSTREAM.in_support(APPROACH_UPSTREAM.start_offset_tool_mean, 70.0) == []
    rot = APPROACH_UPSTREAM.in_support(APPROACH_UPSTREAM.start_offset_tool_mean, 5.0)
    assert len(rot) == 1 and "rotation" in rot[0]


# ======================================================================
# 3. the open-loop observation
# ======================================================================
class ConstantAction:
    """Returns a fixed action and records every observation it was given."""

    name = "constant"

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float64)
        self.seen = []

    def act(self, obs):
        self.seen.append({k: np.array(v) for k, v in obs.items()})
        return self.action

    def describe(self):
        return "constant"


PERMISSIVE = SafetyLimits(
    workspace_pad_cm=1000.0, max_step_translation_mm=1000.0,
    max_step_rotation_deg=360.0, max_tracking_error_cm=1000.0,
    max_consecutive_clamps=0,
)

START = np.array([-0.0330, 0.0167, -0.0841, -2.9896, 0.1754, 1.5404, 0.60])
GOAL = np.array([-0.0274, 0.0219, -0.1161, -3.4686, 0.8872, 1.9073, 0.0])


def _loop(controller, **overrides):
    kwargs = dict(
        frame_mode="identity", goal_orientation="explicit",
        goal_quat_xyzw=tuple(Pose.from_vec7(GOAL).quat_xyzw()),
        goal_jaw="0.0", use_policy_jaw=False, max_steps=50,
        success_trans_cm=0.0, success_rot_rad=0.0,
        step_size=np.asarray(APPROACH_UPSTREAM.step_size),
        goal_rpy_train=tuple(GOAL[3:6]),
    )
    kwargs.update(overrides)
    cfg = LoopConfig(**kwargs)
    loop = ApproachLoop(controller, cfg, PERMISSIVE, contract=APPROACH_UPSTREAM)
    loop.begin(Pose.from_vec7(START), GOAL[:3])
    return loop


def test_command_source_integrates_exactly_like_subtask_env():
    """state[t+1] = state[t] + action * step_size, with nothing read back."""
    action = np.array([0.5, -0.25, 1.0, 0.1, -0.2, 0.3, 0.0])
    ctrl = ConstantAction(action)
    loop = _loop(ctrl)
    step = np.asarray(APPROACH_UPSTREAM.step_size)

    expected = loop._state_vec7.copy()
    # the arm is frozen at the start pose and must not matter at all
    frozen = Pose.from_vec7(START)
    for _ in range(10):
        expected = expected + action * step
        result = loop.step(frozen)
        np.testing.assert_allclose(result.state_vec7, expected, atol=1e-12)


def test_a_stuck_arm_cannot_move_the_policys_belief():
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    frozen = Pose.from_vec7(START)

    a = ConstantAction(action)
    loop_a = _loop(a, observation_source="command")
    for _ in range(8):
        loop_a.step(frozen)

    b = ConstantAction(action)
    loop_b = _loop(b, observation_source="measured")
    for _ in range(8):
        loop_b.step(frozen)

    # open loop: the achieved block has marched 8 steps
    travelled = a.seen[-1]["achieved_goal"][0] - a.seen[0]["achieved_goal"][0]
    assert travelled == pytest.approx(7 * 1.0e-3 * 100.0, abs=1e-6)
    # closed loop on a stuck arm: it has not moved at all, which is exactly the
    # off-distribution state that collapsed the policy to 24% under lag
    stuck = b.seen[-1]["achieved_goal"][0] - b.seen[0]["achieved_goal"][0]
    assert stuck == pytest.approx(0.0, abs=1e-9)


def test_the_jaw_channel_integrates_even_though_the_gripper_does_not():
    """Training closed the jaw during the approach; we deliberately do not."""
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    ctrl = ConstantAction(action)
    loop = _loop(ctrl, policy_jaw_start=0.80)
    frozen = Pose.from_vec7(START)
    for _ in range(4):
        result = loop.step(frozen)
    # 0.80 - 4*0.05
    assert result.state_vec7[6] == pytest.approx(0.60, abs=1e-9)
    # but the published command keeps the measured jaw, so the gripper stays put
    assert result.command.jaw == pytest.approx(START[6])


def test_the_policy_jaw_can_be_frozen():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    ctrl = ConstantAction(action)
    loop = _loop(ctrl, policy_jaw_start=0.80, integrate_policy_jaw=False)
    for _ in range(4):
        result = loop.step(Pose.from_vec7(START))
    assert result.state_vec7[6] == pytest.approx(0.80)


def test_sustained_clamping_aborts():
    """A policy that keeps pushing against a clamp has diverged from the arm."""
    ctrl = ConstantAction(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    cfg = LoopConfig(
        frame_mode="identity", goal_orientation="explicit",
        goal_quat_xyzw=tuple(Pose.from_vec7(GOAL).quat_xyzw()),
        goal_jaw="0.0", use_policy_jaw=False, max_steps=200,
        success_trans_cm=0.0, success_rot_rad=0.0,
        step_size=np.asarray(APPROACH_UPSTREAM.step_size),
        goal_rpy_train=tuple(GOAL[3:6]),
    )
    limits = SafetyLimits(max_step_translation_mm=0.01, max_consecutive_clamps=5,
                          max_tracking_error_cm=1000.0)
    loop = ApproachLoop(ctrl, cfg, limits, contract=APPROACH_UPSTREAM)
    loop.begin(Pose.from_vec7(START), GOAL[:3])
    frozen = Pose.from_vec7(START)
    for _ in range(20):
        result = loop.step(frozen)
        if result.done:
            break
    assert result.done
    assert "safety clamp active" in result.reason
    assert result.index == 5


def test_clamp_streak_resets_when_the_clamp_stops_firing():
    ctrl = ConstantAction(np.zeros(7))
    limits = SafetyLimits(max_consecutive_clamps=3, max_tracking_error_cm=1000.0)
    cfg = LoopConfig(
        frame_mode="identity", goal_orientation="explicit",
        goal_quat_xyzw=tuple(Pose.from_vec7(GOAL).quat_xyzw()),
        goal_jaw="0.0", use_policy_jaw=False, max_steps=20,
        success_trans_cm=0.0, success_rot_rad=0.0,
        goal_rpy_train=tuple(GOAL[3:6]),
    )
    loop = ApproachLoop(ctrl, cfg, limits, contract=APPROACH_UPSTREAM)
    loop.begin(Pose.from_vec7(START), GOAL[:3])
    for _ in range(10):
        result = loop.step(Pose.from_vec7(START))
        assert "safety clamp" not in result.reason


# ======================================================================
# the two success metrics, which must never be mixed
# ======================================================================
def test_rpy_norm_metric_judges_the_integrator():
    ctrl = ConstantAction(np.zeros(7))
    loop = _loop(ctrl, rot_metric="rpy_norm", success_trans_cm=100.0,
                 success_rot_rad=100.0)
    result = loop.step(Pose.from_vec7(START))
    expected = float(np.linalg.norm(result.state_vec7[3:6] - GOAL[3:6]))
    assert result.rpy_norm_err_rad == pytest.approx(expected, abs=1e-9)


def test_geodesic_metric_judges_the_arm():
    ctrl = ConstantAction(np.zeros(7))
    loop = _loop(ctrl, success_trans_cm=100.0, success_rot_rad=100.0)
    far = Pose.from_vec7(np.r_[START[:3], GOAL[3:6], 0.0])
    result = loop.step(far)
    assert result.rot_err_deg == pytest.approx(0.0, abs=1e-6)


def test_defaults_are_the_faithful_ones():
    cfg = LoopConfig()
    assert cfg.observation_source == "command"
    assert cfg.rpy_convention == "surgicai_bound"
    assert cfg.integrate_policy_jaw is True


@pytest.mark.parametrize(
    "field,value",
    [("observation_source", "sometimes"), ("rpy_convention", "nearly"),
     ("rot_metric", "vibes")],
)
def test_bad_fidelity_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        LoopConfig(**{field: value})


# ======================================================================
# the per-step cap measures the command, not the gap to the arm
# ======================================================================
def test_the_step_cap_is_measured_against_the_previous_command():
    """A lagging arm must not drag the command back toward itself.

    With an open-loop observation the command legitimately runs ahead of the
    arm.  Capping each step against the *measured* pose then fires every cycle
    and pulls the command back -- which is the closed loop the open-loop
    contract exists to remove, reintroduced through the safety layer.  On a
    mock arm closing half the gap per cycle this alone aborted a policy run
    that otherwise succeeds.
    """
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    frozen = Pose.from_vec7(START)
    limits = SafetyLimits(max_step_translation_mm=1.0, max_consecutive_clamps=0,
                          max_tracking_error_cm=1000.0, workspace_pad_cm=1000.0)

    loop = _loop(ConstantAction(action))
    loop.limits = limits
    positions = []
    for _ in range(6):
        positions.append(loop.step(frozen).command.p[0])
    # each command advances by the full 1 mm cap, away from the stuck arm
    steps = np.diff(positions) * 1000.0
    np.testing.assert_allclose(steps, 1.0, atol=1e-6)


def test_the_measured_reference_is_still_available():
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    frozen = Pose.from_vec7(START)
    limits = SafetyLimits(max_step_translation_mm=1.0, max_consecutive_clamps=0,
                          max_tracking_error_cm=1000.0, workspace_pad_cm=1000.0,
                          step_reference="measured")
    loop = _loop(ConstantAction(action))
    loop.limits = limits
    positions = [loop.step(frozen).command.p[0] for _ in range(6)]
    # pinned 1 mm from the stuck arm, for ever
    np.testing.assert_allclose(np.diff(positions), 0.0, atol=1e-9)


def test_command_reference_is_the_default():
    assert SafetyLimits().step_reference == "command"
