"""The simulation env shares this package's state machine, so the pieces that
bridge the two are tested here, where AMBF is not needed.

``RL.GraspLift_env`` imports the AMBF-backed ``SRC_approach``.  That base class
is stubbed out; everything the bridge itself does -- the jaw mapping, the
command-to-action conversion, the lift goal -- is real code.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

DEPLOY_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = DEPLOY_ROOT.parent / "src" / "SurgicAI"


@pytest.fixture(scope="module")
def sim_module():
    """Import RL.GraspLift_env with the AMBF-dependent base class stubbed."""
    saved = {k: v for k, v in sys.modules.items() if k == "RL" or k.startswith("RL.")}
    for key in list(saved):
        del sys.modules[key]

    rl = types.ModuleType("RL")
    rl.__path__ = [str(SIM_ROOT / "RL")]
    sys.modules["RL"] = rl

    approach = types.ModuleType("RL.Approach_env")

    class _StubApproach:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.psm_idx = 2
            self.threshold_trans = 0.5
            self.threshold_angle = float(np.deg2rad(30.0))
            self.lift_height = 0.007
            self.grasp_angle = 12.5
            self.goal_obs = None
            self.step_size = np.array(
                [1.5e-3, 1.5e-3, 1.5e-3, np.deg2rad(3.0), np.deg2rad(3.0),
                 np.deg2rad(3.0), 0.05]
            )

        def apply_goal_offset(self, goal):
            return np.asarray(goal, dtype=np.float64)

    class _StubError(RuntimeError):
        pass

    approach.SRC_approach = _StubApproach
    approach.NeedleResetValidityError = _StubError
    sys.modules["RL.Approach_env"] = approach

    utils_pkg = types.ModuleType("RL.utils")
    utils_pkg.__path__ = []
    sys.modules["RL.utils"] = utils_pkg
    utils_mod = types.ModuleType("RL.utils.utils")
    utils_mod.convert_mat_to_frame = lambda mat: mat
    utils_mod.frame_to_vector = lambda frame: np.asarray(frame, dtype=np.float64)
    sys.modules["RL.utils.utils"] = utils_mod

    sys.path.insert(0, str(SIM_ROOT))
    import RL.GraspLift_env as module  # noqa: E402

    yield module

    for key in [k for k in sys.modules if k == "RL" or k.startswith("RL.")]:
        del sys.modules[key]
    sys.modules.update(saved)
    sys.path.remove(str(SIM_ROOT))


def test_sim_jaw_calibration_is_valid(sim_module):
    """The simulator's jaw is a normalised 0..1 command, and PSM.run_grasp_logic
    actuates below 0.05, so 0.0 is the simulation's squeeze."""
    jaw = sim_module.SIM_JAW
    assert jaw.closed_rad == pytest.approx(0.05)
    assert jaw.grip_rad == pytest.approx(0.0)
    assert jaw.grip_rad < jaw.closed_rad  # it really is a squeeze
    assert jaw.normalise(0.05) == pytest.approx(0.0)
    assert jaw.normalise(1.0) == pytest.approx(1.0)


def test_env_forces_the_ghost_sensor_grasp_path(sim_module):
    env = sim_module.SRC_grasp_lift()
    assert env.kwargs["require_grasp_confirmation"] is True
    assert env.kwargs["attach_on_pose_success"] is False


def test_lift_defaults_match_the_real_deployment(sim_module):
    env = sim_module.SRC_grasp_lift()
    assert env.lift_spec.distance_m == pytest.approx(0.015)
    assert env.lift_spec.axis == "z"
    assert env.lift_spec.sign == 1  # +z is away from the pad in the PSM base frame


def test_bad_lift_source_is_rejected(sim_module):
    with pytest.raises(ValueError, match="lift_source"):
        sim_module.SRC_grasp_lift(lift_source="vibes")


def test_frame_axis_lift_goal(sim_module):
    env = sim_module.SRC_grasp_lift(lift_axis="z", lift_sign=1)
    grasp = np.array([-0.030, 0.022, -0.119, -3.1, 0.9, 2.4, 0.0])
    lifted = env.lift_goal_vec7(grasp)
    np.testing.assert_allclose(lifted[:2], grasp[:2], atol=1e-12)
    assert lifted[2] == pytest.approx(grasp[2] + 0.015)
    np.testing.assert_allclose(lifted[3:6], grasp[3:6], atol=1e-12)
    assert lifted[6] == 0.0


def test_action_from_command_inverts_the_env_step(sim_module):
    """The env applies cmd = measured + action * step_size; the bridge must
    produce exactly the action that lands on the sequencer's command."""
    from surgicai_rl_deploy.frames import Pose
    from surgicai_rl_deploy.sequence import Command

    env = sim_module.SRC_grasp_lift()
    measured = np.array([-0.030, 0.022, -0.119, 0.1, 0.2, 0.3, 0.8])
    target_pose = Pose.from_vec7(
        np.array([-0.0295, 0.022, -0.119, 0.1, 0.2, 0.3, 0.0])
    )
    action = env.action_from_command(Command(target_pose, 0.0), measured)

    applied = measured + action * env.step_size
    np.testing.assert_allclose(applied[:3], target_pose.p, atol=1e-9)
    # the jaw is 0.8 away from the target and the env moves 0.05 per step, so
    # one step closes exactly one step's worth, in the right direction
    assert applied[6] == pytest.approx(0.8 - env.step_size[6])


def test_action_from_command_is_clipped(sim_module):
    from surgicai_rl_deploy.frames import Pose
    from surgicai_rl_deploy.sequence import Command

    env = sim_module.SRC_grasp_lift()
    measured = np.zeros(7)
    far = Pose.from_vec7(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    action = env.action_from_command(Command(far, 0.0), measured)
    assert np.all(np.abs(action) <= 1.0)


def test_action_from_command_wraps_the_rpy_delta(sim_module):
    """A delta of 2*pi is no rotation at all and must not become a command."""
    from surgicai_rl_deploy.frames import Pose
    from surgicai_rl_deploy.sequence import Command

    env = sim_module.SRC_grasp_lift()
    measured = np.array([0.0, 0.0, 0.0, -np.pi + 0.01, 0.0, 0.0, 0.0])
    target = Pose.from_vec7(np.array([0.0, 0.0, 0.0, np.pi - 0.01, 0.0, 0.0, 0.0]))
    action = env.action_from_command(Command(target, 0.0), measured)
    # the short way round is -0.02 rad, not +6.26
    assert action[3] < 0.0


def test_ground_truth_confirm_waits_until_the_sensor_fires(sim_module):
    env = sim_module.SRC_grasp_lift()
    state = {"grasped": False}
    env.needle_grasped = lambda: state["grasped"]
    assert env._ground_truth_confirm() is None  # keep waiting, do not decline
    state["grasped"] = True
    assert env._ground_truth_confirm() is True


def test_succeeded_requires_a_real_grasp(sim_module):
    env = sim_module.SRC_grasp_lift()
    env.last_sequence_summary = {
        "phase": "done", "reason": "success", "needle_grasped_ground_truth": False,
    }
    assert env.succeeded() is False
    env.last_sequence_summary["needle_grasped_ground_truth"] = True
    assert env.succeeded() is True
