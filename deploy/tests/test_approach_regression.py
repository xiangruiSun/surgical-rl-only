"""Guards on the parts that were already validated, so adding grasp and lift
cannot quietly change the approach.

The observation contract in this package was verified byte-exact against the
502 observations stored inside ``r6_unified_single_goal_yaw15_seed1_final.zip``
(``tools/verify_contract.py``).  These tests pin the pieces of that contract
that live in code, plus the D2 result recorded in the project findings, so a
regression shows up here rather than on the robot.
"""

import numpy as np
import pytest

from surgicai_rl_deploy.contract import GOAL_SCALE, STEP_SIZE_RAW
from surgicai_rl_deploy.controllers import D2Controller
from surgicai_rl_deploy.frames import Pose
from surgicai_rl_deploy.loop import ApproachLoop, LoopConfig, SafetyLimits
from surgicai_rl_deploy.obs import build_observation
from surgicai_rl_deploy.sequence import GraspLiftSequencer, SequenceConfig

from conftest import REAL_GOAL_POS, REAL_START_POS, REAL_START_QUAT


# --- the frozen contract --------------------------------------------------
def test_step_size_is_the_training_contract():
    """1.0 mm / 3 deg / 0.05 jaw, translation in METRES while the observation
    is in centimetres. The asymmetry is the contract, not a bug.

    This pinned 1.5 mm until 2026-09-16, taken from this repository's own
    RL/utils/utils.py. Replaying each checkpoint from its own demonstrations
    says otherwise -- R6 gives 46/50 at 1.0 mm against 36/50 at 1.5 mm, and
    upstream 19/20 against 7/20 at 0.5 mm -- and both training scripts
    hard-code 1.0e-3. See contract.LEGACY_STEP_SIZE_RAW for what was applied
    before, and tools/replay_demos.py --compare to reproduce.
    """
    np.testing.assert_allclose(
        STEP_SIZE_RAW,
        [1.0e-3, 1.0e-3, 1.0e-3, np.deg2rad(3.0), np.deg2rad(3.0), np.deg2rad(3.0), 0.05],
        rtol=1e-6, atol=0,  # STEP_SIZE_RAW is stored float32, as in training
    )


def test_the_legacy_scale_is_kept_and_is_different():
    from surgicai_rl_deploy.contract import LEGACY_STEP_SIZE_RAW

    assert LEGACY_STEP_SIZE_RAW[0] == pytest.approx(1.5e-3)
    assert not np.allclose(LEGACY_STEP_SIZE_RAW, STEP_SIZE_RAW)


def test_goal_scale_is_cm_for_position_and_raw_for_the_rest():
    np.testing.assert_allclose(GOAL_SCALE, [100, 100, 100, 1, 1, 1, 1])


def test_observation_layout():
    current = np.array([0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4])
    goal = np.array([0.04, 0.05, 0.06, 0.4, 0.5, 0.6, 0.0])
    obs = build_observation(current, goal)

    assert obs["observation"].shape == (21,)
    np.testing.assert_allclose(obs["achieved_goal"], current * GOAL_SCALE, rtol=1e-6)
    np.testing.assert_allclose(obs["desired_goal"], goal * GOAL_SCALE, rtol=1e-6)
    np.testing.assert_allclose(
        obs["observation"][:7], obs["achieved_goal"], rtol=1e-6
    )
    np.testing.assert_allclose(
        obs["observation"][7:14], obs["desired_goal"], rtol=1e-6
    )
    np.testing.assert_allclose(
        obs["observation"][14:], (goal - current) * GOAL_SCALE, rtol=1e-6
    )


def test_positions_reach_the_network_in_centimetres():
    obs = build_observation(
        [0.01, 0.0, 0.0, 0, 0, 0, 0], [0.02, 0.0, 0.0, 0, 0, 0, 0]
    )
    assert obs["achieved_goal"][0] == pytest.approx(1.0)
    assert obs["desired_goal"][0] == pytest.approx(2.0)


# --- the action application -----------------------------------------------
def test_action_is_applied_in_raw_units():
    """cmd_raw = measured_raw + action * STEP_SIZE_RAW."""

    class FixedPolicy:
        name = "fixed"

        def act(self, obs):
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        def describe(self):
            return "fixed +x"

    start = Pose.from_pos_quat(REAL_START_POS, REAL_START_QUAT, 0.0)
    loop = ApproachLoop(
        FixedPolicy(),
        LoopConfig(frame_mode="identity", goal_orientation="hold"),
        SafetyLimits(),
    )
    loop.begin(start, REAL_GOAL_POS)
    result = loop.step(start)
    moved = result.command.p - start.p
    assert moved[0] == pytest.approx(STEP_SIZE_RAW[0], rel=1e-9)


def test_actions_are_clipped_to_the_unit_box():
    class WildPolicy:
        name = "wild"

        def act(self, obs):
            return np.full(7, 50.0)

        def describe(self):
            return "wild"

    start = Pose.from_pos_quat(REAL_START_POS, REAL_START_QUAT, 0.0)
    loop = ApproachLoop(
        WildPolicy(),
        LoopConfig(frame_mode="identity", goal_orientation="hold"),
        SafetyLimits(),
    )
    loop.begin(start, REAL_GOAL_POS)
    result = loop.step(start)
    assert np.all(np.abs(result.action) <= 1.0 + 1e-9)


# --- the recorded D2 result -----------------------------------------------
def test_d2_still_reaches_the_recorded_goal():
    """The project findings record: D2 staged SE(3) servo, success in 20 steps,
    0.15 cm closest approach, on exactly these numbers."""
    from scipy.spatial.transform import Rotation

    start = Pose.from_pos_quat(REAL_START_POS, REAL_START_QUAT, 0.0)
    loop = ApproachLoop(
        D2Controller(staged=True),
        LoopConfig(frame_mode="rebase", goal_orientation="hold"),
        SafetyLimits(),
    )
    loop.begin(start, REAL_GOAL_POS)

    measured = start
    for _ in range(200):
        result = loop.step(measured)
        if result.done:
            break
        target = result.command
        rel = Rotation.from_matrix(measured.R.T @ target.R).as_rotvec()
        measured = Pose(
            target.p, measured.R @ Rotation.from_rotvec(rel).as_matrix(), target.jaw
        )

    assert result.reason == "success"
    assert result.index <= 30
    assert result.trans_err_cm < 1.0


def test_the_sequencer_reuses_the_approach_loop_unchanged(plan, baseline):
    """The approach segment must be the same ApproachLoop with the same config
    knobs, so the RL comparison stays valid."""
    sequencer = GraspLiftSequencer(
        plan,
        D2Controller(staged=True),
        SequenceConfig(frame_mode="rebase", approach_success_trans_cm=1.0,
                       approach_success_rot_deg=10.0),
        SafetyLimits(),
        baseline,
    )
    from surgicai_rl_deploy.sequence import ArmState

    sequencer.begin(ArmState(pose=plan.start, jaw_rad=plan.jaw.approach_open_rad))
    loop = sequencer._approach_loop
    assert isinstance(loop, ApproachLoop)
    assert loop.cfg.frame_mode == "rebase"
    assert loop.cfg.success_trans_cm == 1.0
    assert loop.cfg.use_policy_jaw is False
    assert loop.controller is sequencer.approach_controller


def test_grasp_and_lift_segments_run_in_the_raw_robot_frame(plan, baseline):
    """The frame bridge exists to put the *policy* in its training frame. The
    geometric servo needs no bridge, and adding one would only obscure what the
    commands mean."""
    from surgicai_rl_deploy.sequence import ArmState

    sequencer = GraspLiftSequencer(
        plan, D2Controller(staged=True), SequenceConfig(), SafetyLimits(), baseline
    )
    sequencer.begin(ArmState(pose=plan.start, jaw_rad=plan.jaw.approach_open_rad))
    sequencer._start_segment(plan.grasp, plan.lifted, 100, 0.2, 10.0)
    assert sequencer._segment_loop.cfg.frame_mode == "identity"
    np.testing.assert_allclose(sequencer._segment_loop.bridge.X.p, np.zeros(3))
    np.testing.assert_allclose(sequencer._segment_loop.bridge.X.R, np.eye(3))
