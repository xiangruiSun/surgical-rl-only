import time
import gymnasium as gym
import numpy as np
from ros_abstraction_layer import ral
from RL.utils.gym_manager import GymManager
from RL.needle_reset_ranges import ASSUMED_REAL_NEEDLE_RANGE
from RL.utils.scene_manager import SceneManager
from RL.utils.utils import (
    convert_mat_to_frame,
    frame_to_vector,
)


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class SRC_subtask(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human'],"reward_type":['dense']}

    # Cartesian safety envelope already used by SRC_insert for the same PSM.
    # XYZ are expressed in the PSM base frame in meters.  The jaw uses the
    # environment's historical normalized command convention.
    COMMAND_WORKSPACE_LOW_M = np.array([-0.1, -0.1, -0.25], dtype=np.float32)
    COMMAND_WORKSPACE_HIGH_M = np.array([0.1, 0.1, 0.05], dtype=np.float32)
    COMMAND_JAW_LOW = 0.0
    COMMAND_JAW_HIGH = 1.0

    def __init__(
        self,
        seed=None,
        render_mode=None,
        reward_type="sparse",
        threshold=[0.5, np.deg2rad(30)],
        max_episode_step=200,
        step_size=None,
        stepDR=False,
        checkpoint_compat=False,
        command_integrated=False,
        command_state_clamp=False,
        synchronous_physics=False,
        physics_steps_per_action=10,
        physics_barrier_timeout_s=2.0,
        randomize_psm_reset=True,
    ):

        super(SRC_subtask, self).__init__()
        # R2 default: assumed real-placement support (not a real measurement).
        self.random_range = ASSUMED_REAL_NEEDLE_RANGE.copy()
        self.max_timestep = max_episode_step
        print(f"max episode length is {self.max_timestep}")
        self.base_step_size = step_size
        self.step_size = step_size
        print(f"step size is {self.step_size}")
        self.stepDR = stepDR
        self.checkpoint_compat = bool(checkpoint_compat)
        self.command_integrated = self.checkpoint_compat or bool(command_integrated)
        self.command_state_clamp = bool(command_state_clamp)
        self.synchronous_physics = bool(synchronous_physics)
        self.physics_steps_per_action = int(physics_steps_per_action)
        self.physics_barrier_timeout_s = float(physics_barrier_timeout_s)
        self.last_command_clamp_events = []
        self.command_clamp_events = []
        self.randomize_psm_reset = False if self.checkpoint_compat else bool(randomize_psm_reset)
        if self.stepDR:
            print("State-Space Domain Randomization is enabled")

        if seed is not None:
            self.seed = seed
            print("Set random seed")
        else:
            self.seed = None

        print(f"reward type is {reward_type}")
        self.reward_type = reward_type
        self.threshold_trans = threshold[0]
        self.threshold_angle = threshold[1]
        print(f"Translation threshold: {self.threshold_trans}, angle threshold: {self.threshold_angle}")

        self.ral_instance = ral("src_subtask_env")
        time.sleep(0.5)
        self.ral_instance.spin()

        self.gym_manager = GymManager(self, reward_type=reward_type, threshold=[threshold])
        self.scene_manager = SceneManager(self, self.ral_instance)

        self.goal_obs = None
        self.obs = None
        self.info = None
        self.timestep = 0
        self.psm_idx = None
        self.last_goal_rotation = [None, None]

        print("Initialized!!!")

    def set_seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)

    def measured_achieved_goal(self):
        """Return measured [cm, rad, measured-jaw] state, or None if unavailable."""
        if self.psm_idx is None:
            return None
        psm = self.scene_manager.psm_list[self.psm_idx - 1]
        measured_mat = psm.measured_cp()
        measured_jaw = (
            psm.measured_jaw_angle()
            if hasattr(psm, "measured_jaw_angle")
            else None
        )
        if measured_mat is None or measured_jaw is None:
            return None
        measured = np.append(
            frame_to_vector(convert_mat_to_frame(measured_mat)),
            float(measured_jaw),
        ).astype(np.float32)
        measured[:3] *= 100.0
        return measured

    def reward_achieved_goal(self, observed_achieved_goal):
        if not bool(getattr(self, "measured_success_reward", False)):
            return observed_achieved_goal
        measured = self.measured_achieved_goal()
        return observed_achieved_goal if measured is None else measured

    def apply_command_safety_bounds(self, proposed_vec7_b, *, wrap_rpy):
        """Apply the physical command envelope and record every correction."""
        applied_vec7_b = np.asarray(proposed_vec7_b, dtype=np.float32).copy()
        applied_vec7_b[:3] = np.clip(
            applied_vec7_b[:3],
            self.COMMAND_WORKSPACE_LOW_M,
            self.COMMAND_WORKSPACE_HIGH_M,
        )
        if wrap_rpy:
            # Only the legacy integrated state needs periodic canonicalization.
            # A measured controller starts from fresh feedback every step.
            applied_vec7_b[3:6] = (
                (applied_vec7_b[3:6] + np.pi) % (2.0 * np.pi)
            ) - np.pi
        applied_vec7_b[6] = np.clip(
            applied_vec7_b[6],
            self.COMMAND_JAW_LOW,
            self.COMMAND_JAW_HIGH,
        )

        clipped_off = np.asarray(proposed_vec7_b, dtype=np.float32) - applied_vec7_b
        axis_names = ("x", "y", "z", "roll", "pitch", "yaw", "jaw")
        for axis_idx in range(7):
            if abs(float(clipped_off[axis_idx])) <= 1.0e-12:
                continue
            unit = (
                "m"
                if axis_idx < 3
                else "rad"
                if axis_idx < 6
                else "normalized_jaw"
            )
            event = {
                "step": int(self.timestep),
                "axis": axis_names[axis_idx],
                "unit": unit,
                "proposed": float(proposed_vec7_b[axis_idx]),
                "applied": float(applied_vec7_b[axis_idx]),
                "clipped_amount": float(clipped_off[axis_idx]),
                "operation": "wrap" if 3 <= axis_idx < 6 else "clip",
                "observation_contract": (
                    "command_integrated" if self.command_integrated else "measured"
                ),
            }
            if axis_idx < 3:
                event["clipped_amount_cm"] = float(clipped_off[axis_idx] * 100.0)
            self.last_command_clamp_events.append(event)
            self.command_clamp_events.append(event)
            print("COMMAND_STATE_CLAMP:", event)
        return applied_vec7_b

    def step(self, action):
        """
        Step function, defines the system dynamic and updates the observation.
        Training/eval semantic:
        - command in PSM base frame
        - achieved_goal is commanded psm_goal_list
        - desired_goal stays in PSM base frame
        """
        self.timestep += 1
        if self.timestep == 1:
            self.command_clamp_events = []
        self.last_command_clamp_events = []

        if self.command_integrated:
            achieved_vec7_b = np.array(
                self.scene_manager.psm_goal_list[self.psm_idx - 1],
                dtype=np.float32,
            )
            if np.any(action != 0):
                proposed_vec7_b = (
                    achieved_vec7_b
                    + np.asarray(action, dtype=np.float32) * self.step_size
                )
                achieved_vec7_b = proposed_vec7_b.copy()
                if self.command_state_clamp:
                    achieved_vec7_b = self.apply_command_safety_bounds(
                        proposed_vec7_b,
                        wrap_rpy=True,
                    )
                self.scene_manager.psm_goal_list[self.psm_idx - 1] = achieved_vec7_b
            self.scene_manager.step()
            return self.gym_manager.update_observation(achieved_vec7_b)

        psm = self.scene_manager.psm_list[self.psm_idx - 1]
        measured_mat = psm.measured_cp()
        pre_jp = psm.measured_jp()  # [3PHASE-DIAG] raw JP before command

        if measured_mat is None:
            achieved_vec7_b = np.array(
                self.scene_manager.psm_goal_list[self.psm_idx - 1],
                dtype=np.float32,
            )

            if self.timestep % 20 == 1:
                print(
                    "MEASURED_CP_NONE_FREEZE:",
                    "measured_cp is None; skip action update and keep commanded observation",
                    "achieved_cmd_b=", achieved_vec7_b,
                )

            self.scene_manager.step()
            return self.gym_manager.update_observation(achieved_vec7_b)

        current_obs_b = frame_to_vector(convert_mat_to_frame(measured_mat))
        current_jaw = float(psm.get_jaw_angle())

        if self.timestep <= 5 or self.timestep % 20 == 0:
            desired_b = np.array(self.goal_obs, dtype=np.float32)
            current_vec7_b = np.append(current_obs_b, current_jaw).astype(np.float32)
            scaled_action = action * self.step_size

            print(
                "ACTION_STEP_DIAG:",
                "t=", self.timestep,
                "action=", action,
                "scaled_action=", scaled_action,
                "measured_current_b=", current_vec7_b,
                "desired_b=", desired_b,
                "delta_trans_cm=", (desired_b[:3] - current_vec7_b[:3]) * 100.0,
                "delta_rpy_deg=", np.degrees(wrap_to_pi(desired_b[3:6] - current_vec7_b[3:6])),
            )

        if np.any(action != 0):
            goal_vector_b = np.append(current_obs_b, current_jaw).astype(np.float32)
            proposed_vec7_b = goal_vector_b + action * self.step_size
            goal_vector_b = (
                self.apply_command_safety_bounds(proposed_vec7_b, wrap_rpy=False)
                if self.command_state_clamp
                else proposed_vec7_b
            )

            self.scene_manager.psm_goal_list[self.psm_idx - 1] = goal_vector_b

        cmd_7 = np.array(self.scene_manager.psm_goal_list[self.psm_idx - 1], dtype=np.float32)
        self.scene_manager.step()

        # The command publisher and AMBF state publisher run asynchronously.
        # Give the simulator at least one publish interval, then report the
        # measured Cartesian state rather than echoing the command as success.
        measured_after = None
        for _ in range(10):
            time.sleep(0.02)
            measured_after = psm.measured_cp()
            if measured_after is not None:
                break

        # [3PHASE-DIAG] Phase 3: measured after scene_manager.step()
        if self.timestep <= 5:
            post_jp = psm.measured_jp()
            post_mat = psm.measured_cp()
            post_vec6 = frame_to_vector(convert_mat_to_frame(post_mat)) if post_mat is not None else None
            jp_delta = (np.array(post_jp) - np.array(pre_jp)).tolist() if (pre_jp is not None and post_jp is not None) else None
            print(
                "CMD_MEASURED_3PHASE_DIAG:",
                "t=", self.timestep,
                "pre_jp=", pre_jp,
                "cmd_7=", cmd_7.tolist(),
                "post_jp=", post_jp,
                "jp_delta=", jp_delta,
                "post_cp_b=", np.round(post_vec6, 4).tolist() if post_vec6 is not None else None,
            )

        if measured_after is not None:
            measured_after_vec6 = frame_to_vector(convert_mat_to_frame(measured_after))
            achieved_vec7_b = np.append(
                measured_after_vec6,
                float(psm.get_jaw_angle()),
            ).astype(np.float32)
        else:
            achieved_vec7_b = cmd_7.copy()

        if self.timestep % 20 == 0:
            print(
                "BASE_COMMAND_STEP_DEBUG:",
                "achieved_measured_b=", achieved_vec7_b,
                "commanded_b=", cmd_7,
                "desired_b=", np.array(self.goal_obs, dtype=np.float32),
                "delta=", np.array(self.goal_obs, dtype=np.float32) - achieved_vec7_b,
            )

        return self.gym_manager.update_observation(achieved_vec7_b)

    def render(self, mode='human', close=False):
        pass

    def update_difficulty(self, difficulty_settings):
        self.threshold_trans = difficulty_settings['trans_tolerance']
        self.threshold_angle = difficulty_settings['angle_tolerance']
        self.random_range = difficulty_settings['random_range']

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        goal_len = 7
        achieved_goal = np.array(achieved_goal).reshape(-1, goal_len)
        desired_goal = np.array(desired_goal).reshape(-1, goal_len)

        distances_trans = np.linalg.norm(achieved_goal[:, 0:3] - desired_goal[:, 0:3], axis=1)
        distances_angle = np.linalg.norm(achieved_goal[:, 3:6] - desired_goal[:, 3:6], axis=1)

        if self.reward_type == "dense":
            rewards = -(distances_trans / 100 + distances_angle / 10)
        else:
            rewards = np.where(
                (distances_trans <= self.threshold_trans) & (distances_angle <= self.threshold_angle),
                0, -1
            )
        return np.asarray(rewards, dtype=np.float32)

    def criteria(self):
        achieved_goal = self.obs["achieved_goal"]
        desired_goal = self.obs["desired_goal"]
        distances_trans = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
        distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
        print(
            f"Distance to goal - Translation: {distances_trans:.4f} cm, "
            f"Rotation: {np.rad2deg(distances_angle):.2f} degrees"
        )
        return (distances_trans <= self.threshold_trans) and (distances_angle <= self.threshold_angle)
