import time

import numpy as np
from RL.subtask_env import SRC_subtask
from RL.utils.utils import frame_to_vector, rotation_geodesic_rad


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class NeedleResetValidityError(RuntimeError):
    """Raised when the needle-reset validity gate exhausts all attempts.

    A dedicated type so evaluation/collection loops can catch *this* specific
    failure (skip the episode as reset_invalid) without swallowing unrelated
    RuntimeErrors (e.g. a genuinely dead simulator)."""


class SRC_approach(SRC_subtask):
    # Mean reset goal embedded inside the packaged TD3_HER_BC checkpoint.
    CHECKPOINT_GOAL_MEAN_RAW = np.array(
        [-0.032555993, 0.011748599, -0.11873012, -3.775371, 0.493273, 1.3117456, 0.0],
        dtype=np.float32,
    )

    def __init__(
        self,
        seed=None,
        render_mode=None,
        reward_type="sparse",
        threshold=[0.5, np.deg2rad(30)],
        max_episode_step=200,
        step_size=None,
        stepDR=None,
        checkpoint_compat=False,
        command_integrated=False,
        command_state_clamp=False,
        measured_success_reward=False,
        needle_settle_steps=60,
        needle_settle_interval_s=0.1,
        needle_settle_stable_steps=5,
        needle_settle_translation_tol_m=0.0002,
        needle_settle_rotation_tol_deg=0.3,
        synchronous_physics=False,
        physics_steps_per_action=10,
        physics_barrier_timeout_s=2.0,
        randomize_psm_reset=True,
        randomize_needle_reset=True,
        fixed_historical_goal=False,
        raw_rpy_contract=False,
        require_closed_jaw=False,
        jaw_success_source="measured",
        attach_on_pose_success=True,
        require_grasp_confirmation=False,
        grasp_confirmation_steps=3,
        checkpoint_goal_bank=None,
        checkpoint_goal_random=False,
        checkpoint_goal_random_goals=None,
        checkpoint_goal_jitter=False,
        checkpoint_goal_jitter_goals=None,
        checkpoint_goal_offset_sweep=False,
        checkpoint_goal_offset_goals=None,
        freeze_live_goal=True,
        needle_random_range=None,
        needle_reset_offset=None,
        psm_reset_random_range=None,
        psm2_reset_offset=None,
        goal_offset=None,
        grasp_start_deg=5.0,
        grasp_end_deg=20.0,
        grasp_angle_deg=None,
        lift_height=0.007,
        goal_source_audit=False,
        needle_reset_validity_gate=True,
        needle_valid_xy_max_cm=5.0,
        needle_valid_z_pad_m=None,
        needle_valid_z_tol_cm=1.0,
        needle_valid_so3_max_deg=15.0,
        needle_reset_validity_max_attempts=8,
    ):

        super(SRC_approach, self).__init__(
            seed,
            render_mode,
            reward_type,
            threshold,
            max_episode_step,
            step_size,
            stepDR,
            checkpoint_compat,
            command_integrated,
            command_state_clamp,
            synchronous_physics,
            physics_steps_per_action,
            physics_barrier_timeout_s,
            randomize_psm_reset,
        )
        self.randomize_needle_reset = False if self.checkpoint_compat else bool(randomize_needle_reset)
        self.measured_success_reward = bool(measured_success_reward)
        self.needle_settle_steps = int(needle_settle_steps)
        self.needle_settle_interval_s = float(needle_settle_interval_s)
        self.needle_settle_stable_steps = int(needle_settle_stable_steps)
        self.needle_settle_translation_tol_m = float(
            needle_settle_translation_tol_m
        )
        self.needle_settle_rotation_tol_rad = float(
            np.deg2rad(needle_settle_rotation_tol_deg)
        )
        if self.needle_settle_steps < 2 * self.needle_settle_stable_steps:
            raise ValueError(
                "needle_settle_steps must allow one held and one released "
                "stability window"
            )
        if (
            self.needle_settle_interval_s <= 0.0
            or self.needle_settle_stable_steps < 1
            or self.needle_settle_translation_tol_m <= 0.0
            or self.needle_settle_rotation_tol_rad <= 0.0
        ):
            raise ValueError("needle settle interval and stability thresholds must be positive")
        self.reset_settle_audit = None
        self.fixed_historical_goal = self.checkpoint_compat or bool(fixed_historical_goal)
        self.raw_rpy_contract = self.checkpoint_compat or bool(raw_rpy_contract)
        self.require_closed_jaw = self.checkpoint_compat or bool(require_closed_jaw)
        if jaw_success_source not in {"measured", "command"}:
            raise ValueError("jaw_success_source must be 'measured' or 'command'")
        self.jaw_success_source = str(jaw_success_source)
        self.attach_on_pose_success = bool(attach_on_pose_success)
        self.require_grasp_confirmation = bool(require_grasp_confirmation)
        self.grasp_confirmation_steps = int(grasp_confirmation_steps)
        if self.grasp_confirmation_steps <= 0:
            raise ValueError("grasp_confirmation_steps must be positive")
        self.grasp_confirmation_count = 0
        self.last_grasp_status = None
        goal_bank_values = [] if checkpoint_goal_bank is None else checkpoint_goal_bank
        self.checkpoint_goal_bank = [
            np.asarray(goal, dtype=np.float32) for goal in goal_bank_values
        ]
        random_goal_values = [] if checkpoint_goal_random_goals is None else checkpoint_goal_random_goals
        self.checkpoint_goal_random = bool(checkpoint_goal_random)
        self.checkpoint_goal_random_goals = [
            np.asarray(goal, dtype=np.float32) for goal in random_goal_values
        ]
        if self.checkpoint_goal_random and not self.checkpoint_goal_random_goals:
            raise ValueError("checkpoint_goal_random requires non-empty checkpoint_goal_random_goals")
        jitter_goal_values = [] if checkpoint_goal_jitter_goals is None else checkpoint_goal_jitter_goals
        self.checkpoint_goal_jitter = bool(checkpoint_goal_jitter)
        self.checkpoint_goal_jitter_goals = [
            np.asarray(goal, dtype=np.float32) for goal in jitter_goal_values
        ]
        if self.checkpoint_goal_jitter and not self.checkpoint_goal_jitter_goals:
            raise ValueError("checkpoint_goal_jitter requires non-empty checkpoint_goal_jitter_goals")
        offset_goal_values = [] if checkpoint_goal_offset_goals is None else checkpoint_goal_offset_goals
        self.checkpoint_goal_offset_sweep = bool(checkpoint_goal_offset_sweep)
        self.checkpoint_goal_offset_goals = [
            np.asarray(goal, dtype=np.float32) for goal in offset_goal_values
        ]
        if self.checkpoint_goal_offset_sweep and not self.checkpoint_goal_offset_goals:
            raise ValueError("checkpoint_goal_offset_sweep requires non-empty checkpoint_goal_offset_goals")
        self.freeze_live_goal = bool(freeze_live_goal)
        if needle_random_range is not None:
            self.random_range = np.asarray(needle_random_range, dtype=np.float32)
        self.needle_reset_offset = (
            None
            if needle_reset_offset is None
            else np.asarray(needle_reset_offset, dtype=np.float32)
        )
        if self.needle_reset_offset is not None:
            if self.needle_reset_offset.shape != (3,):
                raise ValueError("needle_reset_offset must be [x_m, y_m, rz_rad]")
            if np.any(np.abs(self.needle_reset_offset) > self.random_range + 1.0e-9):
                raise ValueError("needle_reset_offset exceeds needle_random_range")
        self.psm_reset_random_range = (
            np.asarray(psm_reset_random_range, dtype=np.float32)
            if psm_reset_random_range is not None else
            np.array([0.005, 0.005, 0.005, 0.5, 0.5, 0.5, 0.3], dtype=np.float32)
        )
        self.psm2_reset_offset = (
            np.asarray(psm2_reset_offset, dtype=np.float32)
            if psm2_reset_offset is not None else None
        )
        self.goal_offset = (
            np.asarray(goal_offset, dtype=np.float32)
            if goal_offset is not None else np.zeros(7, dtype=np.float32)
        )
        self.config_grasp_start_deg = float(grasp_start_deg)
        self.config_grasp_end_deg = float(grasp_end_deg)
        self.config_grasp_angle_deg = None if grasp_angle_deg is None else float(grasp_angle_deg)
        self.lift_height = float(lift_height)
        self.goal_source_audit = bool(goal_source_audit)
        self.reset_goal_audit = None
        # --- needle reset validity gate (rejects off-pad / flipped placements) ---
        self.needle_reset_validity_gate = bool(needle_reset_validity_gate)
        self.needle_valid_xy_max_m = float(needle_valid_xy_max_cm) / 100.0
        self.needle_valid_z_pad_m = (
            None if needle_valid_z_pad_m is None else float(needle_valid_z_pad_m)
        )
        self.needle_valid_z_tol_m = float(needle_valid_z_tol_cm) / 100.0
        self.needle_valid_so3_max_rad = float(np.deg2rad(float(needle_valid_so3_max_deg)))
        self.needle_reset_validity_max_attempts = int(needle_reset_validity_max_attempts)
        if self.needle_valid_xy_max_m <= 0 or self.needle_valid_z_tol_m <= 0 or self.needle_valid_so3_max_rad <= 0:
            raise ValueError("needle reset validity thresholds must be positive")
        if self.needle_reset_validity_max_attempts < 1:
            raise ValueError("needle_reset_validity_max_attempts must be >= 1")
        self.reset_validity_audit = None
        self.reset_validity_rejections = []
        self.reset_attempts_used = 0
        self.checkpoint_goal_index = 0
        self.checkpoint_goal_random_index = 0
        self.checkpoint_goal_jitter_index = 0
        self.checkpoint_goal_offset_index = 0
        self.fixed_episode_goal = None
        self.psm_idx = 2

    def apply_goal_offset(self, goal):
        goal = np.asarray(goal, dtype=np.float32).copy()
        goal += self.goal_offset
        return goal

    def _needle_reset_valid(self):
        """Validity gate for the settled needle pose.

        Rejects placements where the free needle bounced off the pad or landed
        flipped. Checks, against configurable conservative thresholds:
          - xy deviation from the commanded target (default <= 5 cm),
          - z deviation from the pad plane (default within +/-1 cm; pad plane
            defaults to needle_init_pos z),
          - SO(3) geodesic to the intended natural-rest orientation
            R_rest * Rz(target_rz) (default <= 15 deg).

        Returns (is_valid: bool, reason: str, diag: dict). It reads state only;
        it does not modify the scene or relax any success criterion.
        """
        from PyKDL import Rotation

        sm = self.scene_manager
        pose = sm.needle.get_pose()  # world frame
        px, py, pz = float(pose.p.x()), float(pose.p.y()), float(pose.p.z())
        init = sm.needle_init_pos
        ix, iy, iz = float(init.x()), float(init.y()), float(init.z())
        pad_z = self.needle_valid_z_pad_m if self.needle_valid_z_pad_m is not None else iz

        target_xy = getattr(sm, "last_needle_target_xy", (ix, iy))
        tx, ty = float(target_xy[0]), float(target_xy[1])
        xy_dev = float(np.hypot(px - tx, py - ty))
        z_dev = float(abs(pz - pad_z))

        target_rz = float(getattr(sm, "last_needle_target_rz", 0.0))
        r_expected = sm.needle_rest_rotation * Rotation.RotZ(target_rz)
        qa = np.asarray(pose.M.GetQuaternion(), dtype=np.float64)
        qb = np.asarray(r_expected.GetQuaternion(), dtype=np.float64)
        qa /= (np.linalg.norm(qa) + 1e-12)
        qb /= (np.linalg.norm(qb) + 1e-12)
        dot = abs(float(np.dot(qa, qb)))
        so3_rad = float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))

        reasons = []
        if xy_dev > self.needle_valid_xy_max_m:
            reasons.append("xy_off_pad")
        if z_dev > self.needle_valid_z_tol_m:
            reasons.append("z_off_pad")
        if so3_rad > self.needle_valid_so3_max_rad:
            reasons.append("orientation_flip")

        diag = {
            "measured_world_xyz_cm": [px * 100.0, py * 100.0, pz * 100.0],
            "needle_init_xyz_cm": [ix * 100.0, iy * 100.0, iz * 100.0],
            "target_xy_cm": [tx * 100.0, ty * 100.0],
            "pad_z_cm": pad_z * 100.0,
            "xy_dev_cm": xy_dev * 100.0,
            "z_dev_cm": z_dev * 100.0,
            "so3_to_target_deg": float(np.degrees(so3_rad)),
            "target_rz_deg": float(np.degrees(target_rz)),
            "rest_quaternion_xyzw": list(sm.needle_rest_quaternion),
            "thresholds": {
                "xy_max_cm": self.needle_valid_xy_max_m * 100.0,
                "z_tol_cm": self.needle_valid_z_tol_m * 100.0,
                "so3_max_deg": float(np.degrees(self.needle_valid_so3_max_rad)),
            },
        }
        return (len(reasons) == 0), (",".join(reasons) if reasons else "ok"), diag

    @staticmethod
    def _pose_step_motion(previous_pose, current_pose):
        previous_xyz = np.array(
            [previous_pose.p.x(), previous_pose.p.y(), previous_pose.p.z()],
            dtype=np.float64,
        )
        current_xyz = np.array(
            [current_pose.p.x(), current_pose.p.y(), current_pose.p.z()],
            dtype=np.float64,
        )
        qa = np.asarray(previous_pose.M.GetQuaternion(), dtype=np.float64)
        qb = np.asarray(current_pose.M.GetQuaternion(), dtype=np.float64)
        qa /= np.linalg.norm(qa) + 1.0e-12
        qb /= np.linalg.norm(qb) + 1.0e-12
        quaternion_dot = abs(float(np.dot(qa, qb)))
        rotation_rad = float(
            2.0 * np.arccos(np.clip(quaternion_dot, -1.0, 1.0))
        )
        return float(np.linalg.norm(current_xyz - previous_xyz)), rotation_rad

    def _settle_randomized_needle(self):
        """Quasi-statically place, release, and verify a free-body steady state."""
        needle = self.scene_manager.needle
        target = self.scene_manager.last_needle_target_frame
        start_pose = needle.get_pose()
        previous_pose = start_pose
        samples = []
        stable_count = 0
        released = False
        release_step = None
        settled = False

        for settle_step in range(self.needle_settle_steps):
            phase = "released" if released else "held"
            if not released:
                # Match the verified Stage-C placement method: refresh the same
                # position command throughout the quasi-static hold.
                needle.hold_pose(target)
            self.scene_manager.step()
            time.sleep(self.needle_settle_interval_s)
            current_pose = needle.get_pose()
            translation_step_m, rotation_step_rad = self._pose_step_motion(
                previous_pose, current_pose
            )
            stable = (
                translation_step_m < self.needle_settle_translation_tol_m
                and rotation_step_rad < self.needle_settle_rotation_tol_rad
            )
            stable_count = stable_count + 1 if stable else 0
            samples.append(
                {
                    "step": settle_step + 1,
                    "phase": phase,
                    "pose_world_m_rad": frame_to_vector(current_pose).astype(float).tolist(),
                    "translation_step_mm": translation_step_m * 1000.0,
                    "rotation_step_deg": float(np.degrees(rotation_step_rad)),
                    "stable": bool(stable),
                    "stable_count": stable_count,
                }
            )

            if not released and stable_count >= self.needle_settle_stable_steps:
                needle.release_pose_control(
                    step_callback=(
                        (lambda: self.scene_manager.world_manager.update(requested_steps=1))
                        if self.scene_manager.world_manager.synchronous
                        else None
                    )
                )
                released = True
                release_step = settle_step + 1
                stable_count = 0
            elif released and stable_count >= self.needle_settle_stable_steps:
                settled = True
                previous_pose = current_pose
                break

            previous_pose = current_pose

        if not released:
            # Never leave a timed-out placement pinned.
            needle.release_pose_control()

        end_pose = needle.get_pose()
        start_vector = frame_to_vector(start_pose).astype(float)
        end_vector = frame_to_vector(end_pose).astype(float)
        target_vector = frame_to_vector(target).astype(float)
        return {
            "settled": bool(settled),
            "timed_out": not settled,
            "steps": len(samples),
            "max_steps": self.needle_settle_steps,
            "interval_s": self.needle_settle_interval_s,
            "stable_steps_required": self.needle_settle_stable_steps,
            "translation_step_threshold_mm": (
                self.needle_settle_translation_tol_m * 1000.0
            ),
            "rotation_step_threshold_deg": float(
                np.degrees(self.needle_settle_rotation_tol_rad)
            ),
            "release_step": release_step,
            "released_stable_count": stable_count if released else 0,
            "start_world_m_rad": start_vector.tolist(),
            "end_world_m_rad": end_vector.tolist(),
            "target_world_m_rad": target_vector.tolist(),
            "translation_drift_cm": float(
                np.linalg.norm(end_vector[:3] - start_vector[:3]) * 100.0
            ),
            "target_translation_error_cm": float(
                np.linalg.norm(end_vector[:3] - target_vector[:3]) * 100.0
            ),
            "target_rotation_error_deg": float(
                np.degrees(rotation_geodesic_rad(end_vector, target_vector))
            ),
            "samples": samples,
            # Compatibility with old audit readers.
            "samples_world_m_rad": [
                sample["pose_world_m_rad"] for sample in samples
            ],
        }

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.set_seed(seed)

        # Needle-reset validity gate: after the free needle settles, reject
        # placements that bounced off the pad or landed flipped, and redo the
        # whole reset (up to N attempts). Only active when the needle is actually
        # randomized; deterministic/checkpoint resets keep the needle at init.
        gate_on = self.needle_reset_validity_gate and self.randomize_needle_reset
        settle_on = self.randomize_needle_reset
        retry_on = gate_on or settle_on
        max_attempts = self.needle_reset_validity_max_attempts if retry_on else 1
        self.reset_validity_audit = None
        self.reset_validity_rejections = []
        reset_attempt = 0
        while True:
            reset_attempt += 1
            self.scene_manager.env_reset()
            if self.randomize_needle_reset:
                self.scene_manager.needle_randomization()

            if settle_on:
                self.reset_settle_audit = self._settle_randomized_needle()
                print(
                    "NEEDLE_RESET_SETTLE:",
                    {
                        key: self.reset_settle_audit[key]
                        for key in (
                            "settled",
                            "timed_out",
                            "steps",
                            "max_steps",
                            "release_step",
                            "translation_drift_cm",
                            "target_translation_error_cm",
                            "target_rotation_error_deg",
                        )
                    },
                )
            else:
                pose = frame_to_vector(
                    self.scene_manager.needle.get_pose()
                ).astype(float)
                self.reset_settle_audit = {
                    "settled": True,
                    "timed_out": False,
                    "steps": 0,
                    "max_steps": self.needle_settle_steps,
                    "interval_s": self.needle_settle_interval_s,
                    "start_world_m_rad": pose.tolist(),
                    "end_world_m_rad": pose.tolist(),
                    "translation_drift_cm": 0.0,
                    "samples_world_m_rad": [],
                }

            settle_valid = bool(self.reset_settle_audit["settled"])
            if not settle_valid:
                _, gate_reason, gate_diag = self._needle_reset_valid()
                reason = "settle_timeout"
                if gate_reason != "ok":
                    reason = f"{reason},{gate_reason}"
                valid = False
            elif gate_on:
                valid, reason, gate_diag = self._needle_reset_valid()
            else:
                self.reset_attempts_used = reset_attempt
                break
            self.reset_validity_audit = {
                "attempt": reset_attempt,
                "max_attempts": max_attempts,
                "valid": bool(valid),
                "reason": reason,
                "settle": self.reset_settle_audit,
                **gate_diag,
            }
            if valid:
                self.reset_attempts_used = reset_attempt
                break
            self.reset_validity_rejections.append(self.reset_validity_audit)
            print("NEEDLE_RESET_REJECTED:", {
                "reason": reason,
                "attempt": reset_attempt,
                "max_attempts": max_attempts,
                "measured_world_xyz_cm": gate_diag["measured_world_xyz_cm"],
                "xy_dev_cm": round(gate_diag["xy_dev_cm"], 4),
                "z_dev_cm": round(gate_diag["z_dev_cm"], 4),
                "so3_to_target_deg": round(gate_diag["so3_to_target_deg"], 3),
            })
            if reset_attempt >= max_attempts:
                self.reset_attempts_used = reset_attempt
                raise NeedleResetValidityError(
                    "Needle reset validity gate failed after "
                    f"{max_attempts} attempts (last reason: {reason}; "
                    f"xy_dev_cm={gate_diag['xy_dev_cm']:.3f}, "
                    f"z_dev_cm={gate_diag['z_dev_cm']:.3f}, "
                    f"so3_deg={gate_diag['so3_to_target_deg']:.2f})"
                )

        psm = self.scene_manager.psm_list[self.psm_idx - 1]
        measured_ready = False
        for warmup_i in range(50):
            self.scene_manager.step()
            if psm.measured_cp() is not None:
                measured_ready = True
                print(
                    "MEASURED_CP_RESET_WARMUP: ready",
                    "warmup_i=", warmup_i,
                )
                break
            time.sleep(0.02)

        if not measured_ready:
            print("MEASURED_CP_RESET_WARMUP: still None after reset warmup")

        self.grasp_start_deg = self.config_grasp_start_deg
        self.grasp_end_deg = self.config_grasp_end_deg
        if self.fixed_historical_goal:
            self.grasp_angle = 12.5
        elif self.config_grasp_angle_deg is not None:
            self.grasp_angle = self.config_grasp_angle_deg
        else:
            self.grasp_angle = np.random.uniform(self.grasp_start_deg, self.grasp_end_deg)
        self.last_goal_rotation = [None, None]
        live_goal_raw_uncalibrated = self.scene_manager.needle_goal_evaluator(
            lift_height=self.lift_height,
            deg_angle=self.grasp_angle,
        )
        live_goal_raw = self.apply_goal_offset(live_goal_raw_uncalibrated)

        if self.checkpoint_goal_bank:
            self.fixed_episode_goal = self.checkpoint_goal_bank[
                self.checkpoint_goal_index % len(self.checkpoint_goal_bank)
            ].copy()
            self.checkpoint_goal_index += 1
            self.needle_obs = self.fixed_episode_goal.copy()
        elif self.checkpoint_goal_random:
            self.fixed_episode_goal = self.checkpoint_goal_random_goals[
                self.checkpoint_goal_random_index % len(self.checkpoint_goal_random_goals)
            ].copy()
            self.checkpoint_goal_random_index += 1
            self.needle_obs = self.fixed_episode_goal.copy()
        elif self.checkpoint_goal_jitter:
            self.fixed_episode_goal = self.checkpoint_goal_jitter_goals[
                self.checkpoint_goal_jitter_index % len(self.checkpoint_goal_jitter_goals)
            ].copy()
            self.checkpoint_goal_jitter_index += 1
            self.needle_obs = self.fixed_episode_goal.copy()
        elif self.checkpoint_goal_offset_sweep:
            self.fixed_episode_goal = self.checkpoint_goal_offset_goals[
                self.checkpoint_goal_offset_index % len(self.checkpoint_goal_offset_goals)
            ].copy()
            self.checkpoint_goal_offset_index += 1
            self.needle_obs = self.fixed_episode_goal.copy()
        elif self.freeze_live_goal:
            self.fixed_episode_goal = live_goal_raw.copy()
            self.needle_obs = self.fixed_episode_goal.copy()
        elif self.fixed_historical_goal:
            self.fixed_episode_goal = self.CHECKPOINT_GOAL_MEAN_RAW.copy()
            self.needle_obs = self.fixed_episode_goal.copy()
        else:
            self.fixed_episode_goal = None
            self.needle_obs = live_goal_raw.copy()
        self.goal_obs = self.needle_obs
        if self.fixed_episode_goal is not None:
            self.multigoal_obs = [self.fixed_episode_goal.copy()]
        else:
            self.multigoal_obs = [
                self.apply_goal_offset(goal) for goal in self.scene_manager.needle_multigoal_evaluator(
                lift_height=self.lift_height, start_degree=self.grasp_start_deg, end_degree=self.grasp_end_deg
                )
            ]
        self.reset_goal_audit = {
            "grasp_angle_deg": float(self.grasp_angle),
            "grasp_start_deg": float(self.grasp_start_deg),
            "grasp_end_deg": float(self.grasp_end_deg),
            "lift_height_m": float(self.lift_height),
            "needle_random_range": np.asarray(self.random_range, dtype=float).tolist(),
            "needle_reset_offset": (
                None
                if self.needle_reset_offset is None
                else np.asarray(self.needle_reset_offset, dtype=float).tolist()
            ),
            "psm_reset_random_range": np.asarray(self.psm_reset_random_range, dtype=float).tolist(),
            "psm2_reset_offset": (
                None if self.psm2_reset_offset is None
                else np.asarray(self.psm2_reset_offset, dtype=float).tolist()
            ),
            "goal_offset": np.asarray(self.goal_offset, dtype=float).tolist(),
            "live_goal_raw_uncalibrated": np.asarray(live_goal_raw_uncalibrated, dtype=float).tolist(),
            "live_goal_raw": np.asarray(live_goal_raw, dtype=float).tolist(),
            "selected_goal_raw": np.asarray(self.goal_obs, dtype=float).tolist(),
            "using_fixed_episode_goal": bool(self.fixed_episode_goal is not None),
            "needle_settle": self.reset_settle_audit,
            "needle_reset_validity": self.reset_validity_audit,
            "reset_attempts_used": int(self.reset_attempts_used),
            "reset_rejections": len(self.reset_validity_rejections),
        }
        if self.goal_source_audit:
            print("LIVE_GOAL_SOURCE_AUDIT:", self.reset_goal_audit)

        measured_mat = psm.measured_cp()
        if measured_mat is not None:
            from RL.utils.utils import convert_mat_to_frame
            measured_vec6_b = frame_to_vector(convert_mat_to_frame(measured_mat))
            measured_vec7_b = np.append(measured_vec6_b, float(psm.get_jaw_angle())).astype(np.float32)
            commanded_vec7_b = np.array(self.scene_manager.psm_goal_list[self.psm_idx - 1], dtype=np.float32)

            print(
                "RESET_COMMAND_MEASURED_DIAG:",
                "commanded_b=", commanded_vec7_b,
                "measured_b=", measured_vec7_b,
                "trans_delta_cm=", np.linalg.norm((commanded_vec7_b[:3] - measured_vec7_b[:3]) * 100.0),
                "rpy_delta_deg=", np.degrees(wrap_to_pi(commanded_vec7_b[3:6] - measured_vec7_b[3:6])),
                "jaw_delta=", commanded_vec7_b[6] - measured_vec7_b[6],
            )
        else:
            print("RESET_COMMAND_MEASURED_DIAG: measured_cp still None")

        current_pos = self.scene_manager.psm_goal_list[self.psm_idx - 1]
        if not self.command_integrated and measured_mat is not None:
            current_pos = measured_vec7_b
            self.scene_manager.psm_goal_list[self.psm_idx - 1] = current_pos.copy()
        obs_current = np.asarray(current_pos, dtype=np.float32)
        goal_vec = np.asarray(self.goal_obs, dtype=np.float32)
        if getattr(self, "align_obs_rpy_branch", False):
            # Match gym_manager.update_observation: snap achieved RPY onto the
            # desired goal's 2*pi branch so the t=0 observation (current block AND
            # delta) is consistent with t>=1 and with the training distribution.
            obs_current = obs_current.copy()
            obs_current[3:6] = obs_current[3:6] + 2 * np.pi * np.round(
                (goal_vec[3:6] - obs_current[3:6]) / (2 * np.pi)
            )
            reset_delta = goal_vec - obs_current
        elif getattr(self, "wrap_obs_rpy_delta", False):
            reset_delta = goal_vec - obs_current
            reset_delta = reset_delta.copy()
            reset_delta[3:6] = (reset_delta[3:6] + np.pi) % (2 * np.pi) - np.pi
        else:
            reset_delta = goal_vec - obs_current
        self.init_obs_array = np.concatenate(
            (obs_current, self.goal_obs, reset_delta), dtype=np.float32
        )
        self.init_obs_dict = {
            "observation": self.init_obs_array,
            "achieved_goal": obs_current,
            "desired_goal": self.goal_obs,
        }

        self.obs = self.gym_manager.normalize_observation(self.init_obs_dict)

        self.info = {"is_success": False}
        print("reset!!!")
        self.timestep = 0
        self.grasp_confirmation_count = 0
        self.last_grasp_status = None

        return self.obs, self.info

    def step(self, action):
        if self.fixed_episode_goal is not None:
            self.goal_obs = self.fixed_episode_goal.copy()
            self.multigoal_obs = [self.fixed_episode_goal.copy()]
            return super(SRC_approach, self).step(action)
        self.needle_obs = self.apply_goal_offset(self.scene_manager.needle_goal_evaluator(
            lift_height=self.lift_height, deg_angle=self.grasp_angle
        ))
        self.multigoal_obs = [
            self.apply_goal_offset(goal) for goal in self.scene_manager.needle_multigoal_evaluator(
            lift_height=self.lift_height, start_degree=self.grasp_start_deg, end_degree=self.grasp_end_deg
            )
        ]
        self.goal_obs = self.needle_obs
        return super(SRC_approach, self).step(action)

    def criteria(self):
        achieved_goal = (
            self.measured_achieved_goal()
            if self.measured_success_reward
            else self.obs["achieved_goal"]
        )
        if achieved_goal is None:
            print("MEASURED_SUCCESS_UNAVAILABLE: no measured pose/jaw feedback")
            return False

        min_trans = np.inf
        min_angle = np.inf
        min_angle_raw = np.inf
        min_angle_wrapped = np.inf
        best_idx = None
        best_pair = None
        best_raw_delta = None
        best_wrapped_delta = None
        best_raw_l2 = None
        best_wrapped_l2 = None

        completion_candidate = None
        grasp_status = None
        if self.require_grasp_confirmation:
            psm = self.scene_manager.psm_list[self.psm_idx - 1]
            grasp_status = psm.grasp_status()
            self.last_grasp_status = grasp_status

        command_jaw = float(
            self.scene_manager.psm_goal_list[self.psm_idx - 1][6]
        )
        measured_jaw = float(achieved_goal[6])
        success_jaw = command_jaw if self.jaw_success_source == "command" else measured_jaw

        # Reward, HER desired_goal, and termination must describe the same task.
        # ``multigoal_obs`` is retained only as a diagnostic view of alternative
        # grasp angles; accepting any of its 25 entries made historical training
        # termination strictly looser than the single desired_goal reward used.
        success_goals = [self.goal_obs]
        for idx, desired_goal in enumerate(success_goals):
            desired_goal = desired_goal * np.array([100, 100, 100, 1, 1, 1, 1])

            distances_trans = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])

            raw_rpy_delta = achieved_goal[3:6] - desired_goal[3:6]
            wrapped_rpy_delta = raw_rpy_delta if self.raw_rpy_contract else wrap_to_pi(raw_rpy_delta)

            distances_angle_raw = np.linalg.norm(raw_rpy_delta)
            distances_angle = (
                rotation_geodesic_rad(achieved_goal, desired_goal)
                if self.measured_success_reward
                else np.linalg.norm(wrapped_rpy_delta)
            )

            if min_trans > distances_trans:
                min_trans = distances_trans
            if min_angle_raw > distances_angle_raw:
                min_angle_raw = distances_angle_raw
            if min_angle > distances_angle:
                min_angle = distances_angle
                min_angle_wrapped = distances_angle
                best_idx = idx
                best_pair = (achieved_goal.copy(), desired_goal.copy())
                best_raw_delta = raw_rpy_delta.copy()
                best_wrapped_delta = wrapped_rpy_delta.copy()
                best_raw_l2 = distances_angle_raw
                best_wrapped_l2 = distances_angle

            grasp_ready = (
                not self.require_grasp_confirmation
                or bool(grasp_status and grasp_status["needle_grasped"])
            )
            jaw_ready = (
                grasp_ready
                if self.require_grasp_confirmation
                else (not self.require_closed_jaw) or success_jaw <= 0.1
            )
            if (
                distances_trans <= self.threshold_trans
                and distances_angle <= self.threshold_angle
                and jaw_ready
                and grasp_ready
            ):
                completion_candidate = {
                    "idx": idx,
                    "distance_trans": float(distances_trans),
                    "distance_angle": float(distances_angle),
                }
                break

        if completion_candidate is not None:
            self.grasp_confirmation_count += 1
            if self.grasp_confirmation_count >= (
                self.grasp_confirmation_steps
                if self.require_grasp_confirmation
                else 1
            ):
                idx = completion_candidate["idx"]
                matched_degree = float(self.grasp_angle)
                print(
                    f"SUCCESS_POSE_SOURCE={'measured' if self.measured_success_reward else 'command'}: "
                    f"matched_degree={matched_degree:.2f}, "
                    f"distance_trans={completion_candidate['distance_trans']:.4f} cm, "
                    f"distance_angle={np.degrees(completion_candidate['distance_angle']):.2f} deg, "
                    f"grasp_confirmation_count={self.grasp_confirmation_count}, "
                    f"jaw_success_source={self.jaw_success_source}, "
                    f"jaw_success_value={success_jaw:.4f}, "
                    f"measured_jaw={measured_jaw:.4f}, command_jaw={command_jaw:.4f}"
                )
                if not self.require_grasp_confirmation and self.attach_on_pose_success:
                    # Historical checkpoint behavior.  The explicit P2 grasp
                    # servo path instead relies on PSM ghost-sensor grasp logic.
                    print("Attach the needle to the gripper")
                    self.scene_manager.psm2.actuate("Needle")
                    self.scene_manager.needle.release()
                elif not self.require_grasp_confirmation:
                    print("Pose-close success recorded without synthetic needle attachment")
                else:
                    print("Needle grasp confirmed by finger-sensor actuator path")
                return True
        else:
            self.grasp_confirmation_count = 0

        self.min_trans = min_trans
        self.min_angle_distance = min_angle

        if self.timestep % 20 == 0:
            print(
                f"SUCCESS_POSE_SOURCE={'measured' if self.measured_success_reward else 'command'}: "
                f"min_trans={min_trans:.4f} cm, "
                f"min_angle={np.degrees(min_angle):.2f} deg"
            )
            print(
                "RPY_BEST_DEBUG:",
                "best_idx=", best_idx,
                "achieved_rpy=", best_pair[0][3:6] if best_pair else None,
                "desired_rpy=", best_pair[1][3:6] if best_pair else None,
                "rpy_delta_deg=", np.degrees(best_pair[1][3:6] - best_pair[0][3:6]) if best_pair else None,
            )
            print(
                "RPY_WRAP_DIAG:",
                "best_idx=", best_idx,
                "raw_delta_deg=", np.degrees(best_raw_delta) if best_raw_delta is not None else None,
                "wrapped_delta_deg=", np.degrees(best_wrapped_delta) if best_wrapped_delta is not None else None,
                "raw_l2_deg=", np.degrees(best_raw_l2) if best_raw_l2 is not None else None,
                "wrapped_l2_deg=", np.degrees(best_wrapped_l2) if best_wrapped_l2 is not None else None,
            )

        return False

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        goal_len = 7
        scalar_input = np.asarray(achieved_goal).ndim == 1
        achieved_goal = np.array(achieved_goal).reshape(-1, goal_len)
        desired_goal = np.array(desired_goal).reshape(-1, goal_len)

        distances_trans = np.linalg.norm(achieved_goal[:, 0:3] - desired_goal[:, 0:3], axis=1)
        raw_rpy_delta = achieved_goal[:, 3:6] - desired_goal[:, 3:6]
        rpy_delta = raw_rpy_delta if self.raw_rpy_contract else wrap_to_pi(raw_rpy_delta)
        if self.measured_success_reward:
            distances_angle = np.array([
                rotation_geodesic_rad(actual, goal)
                for actual, goal in zip(achieved_goal, desired_goal)
            ], dtype=np.float32)
        else:
            distances_angle = np.linalg.norm(rpy_delta, axis=1)

        in_region = (distances_trans <= self.threshold_trans) & (distances_angle <= self.threshold_angle)
        grasp_zone_reward = np.zeros_like(distances_trans) if self.raw_rpy_contract else np.where(in_region, 0.5, 0.0)

        if self.reward_type == "dense":
            rewards = -(distances_trans / 100 + distances_angle / 10) + grasp_zone_reward
        else:
            rewards = np.where(in_region, 0, -1)
        rewards = np.asarray(rewards, dtype=np.float32)
        return float(rewards[0]) if scalar_input else rewards
