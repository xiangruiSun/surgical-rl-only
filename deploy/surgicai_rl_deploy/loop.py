"""The closed-loop approach controller, with no ROS dependency.

The same object drives the ROS node and the offline replay tool, so what you
validate offline is literally the code that runs on the robot.

One iteration:

    measured pose (robot frame)
        -> bridge into the policy frame
        -> build the 21-dim observation against the frozen goal
        -> controller returns action in [-1, 1]^7
        -> cmd_policy = measured_policy + action * STEP_SIZE_RAW   (raw units)
        -> bridge back to the robot frame
        -> safety clamps
        -> commanded pose
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .contract import (
    MAX_EPISODE_STEPS,
    R6_START_OFFSET_TOOL_MAX,
    R6_START_OFFSET_TOOL_MIN,
    R6_START_TO_GOAL_ROTVEC_MEAN,
    R6_START_ROT_DEG_MAX,
    R6_START_ROT_DEG_MIN,
    R6_TRAINED_GOAL_VEC7,
    SUPPORT_EPS_CM,
    STEP_SIZE_RAW,
    SUCCESS_ROT_RAD,
    SUCCESS_TRANS_CM,
)
from .frames import (
    FrameBridge,
    Pose,
    rotation_error_rad,
    translation_error_cm,
)
from .obs import observation_from_poses


@dataclass
class SafetyLimits:
    """Hard bounds applied to every commanded pose, in the robot frame."""

    #: box padding, in cm, around the axis-aligned box spanned by start+goal
    workspace_pad_cm: float = 2.0
    #: largest translation between the measured pose and the new command
    max_step_translation_mm: float = 2.5
    #: largest rotation between the measured pose and the new command
    max_step_rotation_deg: float = 5.0
    #: abort if the arm lags this far behind the previous command
    max_tracking_error_cm: float = 1.5
    #: abort if measured_cp goes stale (ROS node only)
    max_pose_age_s: float = 0.25


@dataclass
class LoopConfig:
    frame_mode: str = "rebase"  # rebase | translate | identity
    goal_orientation: str = "hold"  # hold | trained_relative | explicit
    goal_quat_xyzw: Optional[tuple] = None
    goal_jaw: str = "hold"  # hold | closed | open | <float>
    use_policy_jaw: bool = False
    max_steps: int = MAX_EPISODE_STEPS
    success_trans_cm: float = SUCCESS_TRANS_CM
    success_rot_rad: float = SUCCESS_ROT_RAD
    align_rpy_branch: bool = False
    wrap_rpy_delta: bool = False
    step_size: np.ndarray = field(
        default_factory=lambda: np.asarray(STEP_SIZE_RAW, dtype=np.float64)
    )


@dataclass
class StepResult:
    index: int
    action: np.ndarray
    command: Pose  # robot frame
    measured: Pose  # robot frame
    trans_err_cm: float
    rot_err_deg: float
    done: bool
    reason: str
    clamps: list = field(default_factory=list)


class ApproachLoop:
    def __init__(self, controller, config: Optional[LoopConfig] = None,
                 limits: Optional[SafetyLimits] = None):
        self.controller = controller
        self.cfg = config or LoopConfig()
        self.limits = limits or SafetyLimits()
        self.trained_goal = Pose.from_vec7(R6_TRAINED_GOAL_VEC7)

        self.start: Optional[Pose] = None
        self.goal_robot: Optional[Pose] = None
        self.goal_policy: Optional[Pose] = None
        self.bridge: Optional[FrameBridge] = None
        self.last_command: Optional[Pose] = None
        self.index = 0
        self._box_low = None
        self._box_high = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _resolve_goal_jaw(self, start_jaw: float) -> float:
        spec = self.cfg.goal_jaw
        if spec == "hold":
            return float(start_jaw)
        if spec == "closed":
            return 0.0
        if spec == "open":
            return 1.0
        return float(spec)

    def _resolve_goal_rotation(self, start: Pose) -> np.ndarray:
        mode = self.cfg.goal_orientation
        if mode == "hold":
            return start.R.copy()
        if mode == "explicit":
            if self.cfg.goal_quat_xyzw is None:
                raise ValueError("goal_orientation='explicit' needs goal_quat_xyzw")
            return Pose.from_pos_quat([0, 0, 0], self.cfg.goal_quat_xyzw).R
        if mode == "trained_relative":
            # Rotate the wrist by the same amount the policy rotated it during
            # training (mean start->goal rotation, ~65 deg).  This is the only
            # goal orientation that leaves the policy's rotation channels in
            # distribution; with "hold" they are zero, which never happened in
            # any demonstration episode.
            from scipy.spatial.transform import Rotation

            rel = Rotation.from_rotvec(R6_START_TO_GOAL_ROTVEC_MEAN).as_matrix()
            return start.R @ rel
        raise ValueError(f"unknown goal_orientation {mode!r}")

    def begin(self, start: Pose, goal_position_m) -> dict:
        """Freeze the goal for the episode and build the frame bridge."""
        self.start = start
        goal_p = np.asarray(goal_position_m, dtype=np.float64).reshape(3)
        goal_R = self._resolve_goal_rotation(start)
        goal_jaw = self._resolve_goal_jaw(start.jaw)
        self.goal_robot = Pose(goal_p, goal_R, goal_jaw)

        self.bridge = FrameBridge.build(
            self.cfg.frame_mode, self.goal_robot, self.trained_goal
        )
        self.goal_policy = self.bridge.to_policy(self.goal_robot)
        if self.cfg.frame_mode == "rebase":
            # Numerically identical to the trained goal; keep the jaw we chose.
            self.goal_policy = self.trained_goal.with_jaw(goal_jaw)

        lo = np.minimum(start.p, goal_p) - self.limits.workspace_pad_cm / 100.0
        hi = np.maximum(start.p, goal_p) + self.limits.workspace_pad_cm / 100.0
        self._box_low, self._box_high = lo, hi
        self.last_command = None
        self.index = 0
        return self.distribution_report()

    # ------------------------------------------------------------------
    # in-distribution report
    # ------------------------------------------------------------------
    def distribution_report(self) -> dict:
        """How far the episode sits outside the R6 training support."""
        start_policy = self.bridge.to_policy(self.start)
        goal_policy = self.goal_policy

        dp_world_cm = (goal_policy.p - start_policy.p) * 100.0
        dp_tool_cm = start_policy.R.T @ dp_world_cm
        rot_deg = np.degrees(rotation_error_rad(start_policy, goal_policy))

        below = dp_tool_cm < R6_START_OFFSET_TOOL_MIN - SUPPORT_EPS_CM
        above = dp_tool_cm > R6_START_OFFSET_TOOL_MAX + SUPPORT_EPS_CM
        axes = "xyz"
        offenders = [
            f"tool-{axes[i]} {dp_tool_cm[i]:+.2f} cm outside "
            f"[{R6_START_OFFSET_TOOL_MIN[i]:+.2f}, {R6_START_OFFSET_TOOL_MAX[i]:+.2f}]"
            for i in range(3)
            if below[i] or above[i]
        ]
        if not (R6_START_ROT_DEG_MIN <= rot_deg <= R6_START_ROT_DEG_MAX):
            offenders.append(
                f"start->goal rotation {rot_deg:.1f} deg outside "
                f"[{R6_START_ROT_DEG_MIN:.1f}, {R6_START_ROT_DEG_MAX:.1f}]"
            )
        return {
            "frame_mode": self.cfg.frame_mode,
            "start_offset_tool_cm": dp_tool_cm,
            "start_offset_policy_frame_cm": dp_world_cm,
            "start_to_goal_rotation_deg": float(rot_deg),
            "translation_cm": float(np.linalg.norm(dp_world_cm)),
            "out_of_distribution": offenders,
            "in_distribution": not offenders,
        }

    # ------------------------------------------------------------------
    # one control step
    # ------------------------------------------------------------------
    def _clamp(self, measured: Pose, command: Pose):
        clamps = []
        p = command.p.copy()

        boxed = np.clip(p, self._box_low, self._box_high)
        if not np.allclose(boxed, p):
            clamps.append(
                {"kind": "workspace_box", "proposed_cm": p * 100.0, "applied_cm": boxed * 100.0}
            )
            p = boxed

        delta = p - measured.p
        norm = float(np.linalg.norm(delta))
        max_step = self.limits.max_step_translation_mm / 1000.0
        if norm > max_step:
            p = measured.p + delta * (max_step / norm)
            clamps.append(
                {"kind": "step_translation", "proposed_mm": norm * 1000.0,
                 "applied_mm": max_step * 1000.0}
            )

        from scipy.spatial.transform import Rotation

        rel = Rotation.from_matrix(measured.R.T @ command.R)
        rotvec = rel.as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        max_rot = np.deg2rad(self.limits.max_step_rotation_deg)
        R_cmd = command.R
        if angle > max_rot:
            R_cmd = measured.R @ Rotation.from_rotvec(rotvec * (max_rot / angle)).as_matrix()
            clamps.append(
                {"kind": "step_rotation", "proposed_deg": np.degrees(angle),
                 "applied_deg": self.limits.max_step_rotation_deg}
            )
        return Pose(p, R_cmd, command.jaw), clamps

    def step(self, measured: Pose) -> StepResult:
        if self.bridge is None:
            raise RuntimeError("call begin() before step()")
        self.index += 1

        # tracking guard: did the arm actually follow the previous command?
        if self.last_command is not None:
            lag = translation_error_cm(measured, self.last_command)
            if lag > self.limits.max_tracking_error_cm:
                return StepResult(
                    self.index, np.zeros(7), self.last_command, measured,
                    translation_error_cm(measured, self.goal_robot),
                    float(np.degrees(rotation_error_rad(measured, self.goal_robot))),
                    True, f"abort: tracking error {lag:.2f} cm exceeds limit",
                )

        measured_policy = self.bridge.to_policy(measured)
        obs = observation_from_poses(
            measured_policy,
            self.goal_policy,
            align_rpy_branch=self.cfg.align_rpy_branch,
            wrap_rpy_delta=self.cfg.wrap_rpy_delta,
        )
        action = np.asarray(self.controller.act(obs), dtype=np.float64).reshape(7)
        action = np.clip(action, -1.0, 1.0)

        cur_vec7 = measured_policy.to_vec7()
        cmd_vec7 = cur_vec7 + action * self.cfg.step_size
        if not self.cfg.use_policy_jaw:
            cmd_vec7[6] = measured.jaw

        command_policy = Pose.from_vec7(cmd_vec7)
        command_robot = self.bridge.to_robot(command_policy)
        command_robot, clamps = self._clamp(measured, command_robot)

        trans_err = translation_error_cm(measured, self.goal_robot)
        rot_err = rotation_error_rad(measured, self.goal_robot)

        done, reason = False, "running"
        if trans_err <= self.cfg.success_trans_cm and rot_err <= self.cfg.success_rot_rad:
            done, reason = True, "success"
        elif self.index >= self.cfg.max_steps:
            done, reason = True, "max_steps"

        self.last_command = command_robot
        return StepResult(
            self.index, action.astype(np.float32), command_robot, measured,
            trans_err, float(np.degrees(rot_err)), done, reason, clamps,
        )
