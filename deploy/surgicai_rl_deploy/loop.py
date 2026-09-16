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
    bound_roll,
    rotation_error_rad,
    translation_error_cm,
    unwrap_rpy_to,
    vec7_bound,
)
from .obs import build_observation


@dataclass
class SafetyLimits:
    """Hard bounds applied to every commanded pose, in the robot frame."""

    #: box padding, in cm, around the axis-aligned box spanned by start+goal
    workspace_pad_cm: float = 2.0
    #: largest translation between consecutive commands
    max_step_translation_mm: float = 2.5
    #: largest rotation between consecutive commands
    max_step_rotation_deg: float = 5.0
    #: what the per-step caps are measured against.
    #:
    #: ``"command"`` -- the previous command. This bounds how fast the
    #: *commanded trajectory* moves, which is the thing a per-step cap is for.
    #: ``"measured"`` -- the arm's current pose, which was the original
    #: behaviour and is wrong under the open-loop observation contract: a
    #: lagging arm makes the command legitimately run ahead of it, the cap
    #: fires every cycle, and the clamp quietly drags the command back toward
    #: the arm -- which is the closed loop we removed, reintroduced through the
    #: safety layer. Measured on a mock arm closing half the gap per cycle,
    #: this alone aborted a policy run that succeeds with "command".
    #:
    #: How far the arm may fall *behind* is a separate question, and
    #: ``max_tracking_error_cm`` below is what answers it.
    step_reference: str = "command"
    #: abort if the arm lags this far behind the previous command
    max_tracking_error_cm: float = 1.5
    #: abort if measured_cp goes stale (ROS node only)
    max_pose_age_s: float = 0.25
    #: abort after this many consecutive cycles in which a safety clamp had to
    #: modify the command.  With an open-loop observation (the training
    #: contract) the policy's internal state keeps integrating while the clamp
    #: holds the robot back, so sustained clamping means the policy and the arm
    #: have quietly stopped agreeing about where the tool is.  That is the state
    #: in which a "small" next command can be a large real motion.  0 disables.
    max_consecutive_clamps: int = 12


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
    #: RPY of the goal on the *training* branch, in the policy frame.  The
    #: SurgicAI environments integrate RPY as free state, so training data sits
    #: on 2*pi branches that a matrix round-trip destroys; observations are
    #: unwrapped onto this reference before they reach the network.  None in
    #: 'rebase' mode means "use the trained goal's own RPY", which is right by
    #: construction.  Set unwrap_rpy=False to restore the old behaviour.
    goal_rpy_train: Optional[tuple] = None
    unwrap_rpy: bool = True
    step_size: np.ndarray = field(
        default_factory=lambda: np.asarray(STEP_SIZE_RAW, dtype=np.float64)
    )

    # -- the three fidelity switches ---------------------------------------
    #: ``"command"`` (the training contract) or ``"measured"``.
    #:
    #: ``RL/subtask_env.py :: step`` reads
    #:
    #:     current = self.psm_goal_list[self.psm_idx-1]
    #:     self.psm_goal_list[self.psm_idx-1] = current + action*step_size
    #:     self._update_observation(self.psm_goal_list[self.psm_idx-1])
    #:
    #: ``measured_cp`` is never read back inside a subtask.  The observation is
    #: therefore a pure integrator over the commands, and the policy is
    #: open-loop within an episode.  Feeding it the measured pose instead --
    #: which this package did until now -- looks harmless on a perfect
    #: kinematic arm and is not: on real hardware the arm lags the command, so
    #: the achieved block, the desired-minus-achieved block and the next
    #: integration step all drift away from anything the policy saw in
    #: training, a little more on every cycle.
    #:
    #: Geometric servo segments want the opposite -- they should react to where
    #: the arm actually is -- so they pass ``"measured"`` explicitly.
    observation_source: str = "command"
    #: ``"surgicai_bound"`` reproduces ``Frame2Vec(bound=True)`` exactly: roll in
    #: ``(-2*pi, 0]``, pitch and yaw untouched.  ``"unwrap"`` is the older
    #: reference-based heuristic, kept so the two can be compared.
    #: ``"canonical"`` is scipy's ``[-pi, pi]``, i.e. the defect, kept only so a
    #: test can pin what it did.
    rpy_convention: str = "surgicai_bound"
    #: ``"rpy_norm"`` is SurgicAI's own success test -- the Euclidean norm of the
    #: RPY difference vector, which is what the reported 96%/97% were measured
    #: with.  ``"geodesic"`` is the true angle between the two orientations.
    #: They are different numbers; both are always reported.
    rot_metric: str = "geodesic"
    #: Seed the policy's internal jaw channel with this normalised value rather
    #: than the measured jaw.  The Approach demonstrations all begin at 0.80 and
    #: end at 0.00, and the jaw is 3 of the 21 observation dimensions, so a jaw
    #: channel pinned at whatever the gripper happens to be doing is off
    #: distribution from the first cycle.  None = use the measured jaw.
    policy_jaw_start: Optional[float] = None
    #: Let the action integrate the internal jaw channel even when the physical
    #: jaw is held open by the sequencer.  This is what training did; the
    #: deliberate deviation is that we do not *publish* that jaw command during
    #: the approach.
    integrate_policy_jaw: bool = True
    #: Re-seed the internal state from the clamped command whenever a safety
    #: clamp fires.  Off by default: it hides divergence that
    #: ``max_consecutive_clamps`` is there to catch.
    resync_state_on_clamp: bool = False

    def __post_init__(self):
        if self.observation_source not in ("command", "measured"):
            raise ValueError(
                f"observation_source must be 'command' or 'measured'; "
                f"got {self.observation_source!r}"
            )
        if self.rpy_convention not in ("surgicai_bound", "unwrap", "canonical"):
            raise ValueError(f"unknown rpy_convention {self.rpy_convention!r}")
        if self.rot_metric not in ("geodesic", "rpy_norm"):
            raise ValueError(f"unknown rot_metric {self.rot_metric!r}")

    @classmethod
    def from_contract(cls, contract, **overrides) -> "LoopConfig":
        """Build a config whose action scale and tolerances match a checkpoint."""
        base = dict(
            step_size=np.asarray(contract.step_size, dtype=np.float64),
            max_steps=int(contract.max_steps),
            success_trans_cm=float(contract.success_trans_cm),
            success_rot_rad=float(contract.success_rot_rad),
            policy_jaw_start=float(contract.demo_start_jaw),
            goal_jaw=str(float(contract.demo_goal_jaw)),
        )
        base.update(overrides)
        return cls(**base)


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
    #: error between the policy's *internal* state and the goal, in the policy
    #: frame.  With an open-loop observation this is what the policy believes;
    #: ``trans_err_cm`` above is what the arm actually did.  A growing gap
    #: between the two is the signature of an arm that is not tracking.
    state_trans_err_cm: float = float("nan")
    state_rot_err_deg: float = float("nan")
    #: SurgicAI's own rotation metric, ||rpy_achieved - rpy_desired||, radians
    rpy_norm_err_rad: float = float("nan")
    #: the policy's internal 7-vector this cycle, policy frame, raw units
    state_vec7: Optional[np.ndarray] = None


class ApproachLoop:
    def __init__(self, controller, config: Optional[LoopConfig] = None,
                 limits: Optional[SafetyLimits] = None, contract=None):
        self.controller = controller
        self.cfg = config or LoopConfig()
        self.limits = limits or SafetyLimits()
        self.contract = contract
        trained_goal_vec7 = (
            R6_TRAINED_GOAL_VEC7 if contract is None else contract.trained_goal_vec7
        )
        self.trained_goal = Pose.from_vec7(trained_goal_vec7)
        self._trained_goal_vec7 = np.asarray(trained_goal_vec7, dtype=np.float64)

        self.start: Optional[Pose] = None
        self.goal_robot: Optional[Pose] = None
        self.goal_policy: Optional[Pose] = None
        self.bridge: Optional[FrameBridge] = None
        self.last_command: Optional[Pose] = None
        self.index = 0
        self._box_low = None
        self._box_high = None
        self._goal_rpy = None
        #: the policy's internal state, policy frame, raw units.  This is the
        #: `psm_goal_list` of the training environment.
        self._state_vec7: Optional[np.ndarray] = None
        self._goal_vec7: Optional[np.ndarray] = None
        self._clamp_streak = 0

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

        goal_vec7 = self.goal_policy.to_vec7()
        if self._convention() == "canonical":
            # "canonical" means *reproduce the defect exactly*, which includes
            # ignoring any stated training branch.  It exists so a test can pin
            # what the broken path did; nothing else should select it.
            self._goal_rpy = goal_vec7[3:6].copy()
        elif self.cfg.goal_rpy_train is not None:
            self._goal_rpy = np.asarray(self.cfg.goal_rpy_train, dtype=np.float64)
        elif self.cfg.frame_mode == "rebase":
            # goal_policy IS the trained goal, so its contract RPY is the branch.
            self._goal_rpy = self._trained_goal_vec7[3:6].copy()
        else:
            self._goal_rpy = self._on_branch(goal_vec7[3:6], reference=goal_vec7[3:6])
        goal_vec7[3:6] = self._goal_rpy
        self._goal_vec7 = goal_vec7

        # Seed the policy's internal state.  From here on the episode is an
        # integrator: no pose is ever re-derived from a rotation matrix again,
        # which is precisely why the branch cannot drift.
        start_policy = self.bridge.to_policy(start)
        state = start_policy.to_vec7()
        state[3:6] = self._on_branch(state[3:6], reference=self._goal_rpy)
        if self.cfg.policy_jaw_start is not None:
            state[6] = float(self.cfg.policy_jaw_start)
        self._state_vec7 = state

        lo = np.minimum(start.p, goal_p) - self.limits.workspace_pad_cm / 100.0
        hi = np.maximum(start.p, goal_p) + self.limits.workspace_pad_cm / 100.0
        self._box_low, self._box_high = lo, hi
        self.last_command = None
        self.index = 0
        self._clamp_streak = 0
        return self.distribution_report()

    # ------------------------------------------------------------------
    def _convention(self) -> str:
        """``unwrap_rpy=False`` is the legacy switch for "leave it canonical"."""
        return self.cfg.rpy_convention if self.cfg.unwrap_rpy else "canonical"

    def _on_branch(self, rpy, reference=None) -> np.ndarray:
        convention = self._convention()
        rpy = np.asarray(rpy, dtype=np.float64)
        if convention == "surgicai_bound":
            return bound_roll(rpy)
        if convention == "unwrap":
            if reference is None:
                return rpy.copy()
            return unwrap_rpy_to(rpy, reference)
        return rpy.copy()

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

        if self.contract is not None:
            offenders = self.contract.in_support(dp_tool_cm, rot_deg)
        else:
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
            "contract": None if self.contract is None else self.contract.name,
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

        # Per-step caps bound the commanded trajectory, not the gap to the arm;
        # see SafetyLimits.step_reference.
        reference = (
            self.last_command
            if self.limits.step_reference == "command" and self.last_command is not None
            else measured
        )

        delta = p - reference.p
        norm = float(np.linalg.norm(delta))
        max_step = self.limits.max_step_translation_mm / 1000.0
        if norm > max_step:
            p = reference.p + delta * (max_step / norm)
            clamps.append(
                {"kind": "step_translation", "proposed_mm": norm * 1000.0,
                 "applied_mm": max_step * 1000.0}
            )

        from scipy.spatial.transform import Rotation

        rel = Rotation.from_matrix(reference.R.T @ command.R)
        rotvec = rel.as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        max_rot = np.deg2rad(self.limits.max_step_rotation_deg)
        R_cmd = command.R
        if angle > max_rot:
            R_cmd = reference.R @ Rotation.from_rotvec(
                rotvec * (max_rot / angle)
            ).as_matrix()
            clamps.append(
                {"kind": "step_rotation", "proposed_deg": np.degrees(angle),
                 "applied_deg": self.limits.max_step_rotation_deg}
            )
        return Pose(p, R_cmd, command.jaw), clamps

    def _terminal(self, measured: Pose, reason: str) -> StepResult:
        return StepResult(
            self.index,
            np.zeros(7),
            self.last_command or measured,
            measured,
            translation_error_cm(measured, self.goal_robot),
            float(np.degrees(rotation_error_rad(measured, self.goal_robot))),
            True,
            reason,
            state_vec7=None if self._state_vec7 is None else self._state_vec7.copy(),
        )

    def step(self, measured: Pose) -> StepResult:
        if self.bridge is None:
            raise RuntimeError("call begin() before step()")
        self.index += 1

        # tracking guard: did the arm actually follow the previous command?
        if self.last_command is not None:
            lag = translation_error_cm(measured, self.last_command)
            if lag > self.limits.max_tracking_error_cm:
                return self._terminal(
                    measured, f"abort: tracking error {lag:.2f} cm exceeds limit"
                )

        measured_policy = self.bridge.to_policy(measured)
        goal_vec7 = self._goal_vec7

        if self.cfg.observation_source == "measured":
            state_vec7 = measured_policy.to_vec7()
            state_vec7[3:6] = self._on_branch(state_vec7[3:6], reference=self._goal_rpy)
            if self.cfg.policy_jaw_start is not None:
                # the jaw channel stays on the integrator even when the pose
                # channels are closed-loop: the physical gripper is deliberately
                # not doing what the policy asked during the approach
                state_vec7[6] = self._state_vec7[6]
        else:
            state_vec7 = self._state_vec7.copy()

        obs = build_observation(
            state_vec7,
            goal_vec7,
            align_rpy_branch=self.cfg.align_rpy_branch,
            wrap_rpy_delta=self.cfg.wrap_rpy_delta,
        )
        action = np.asarray(self.controller.act(obs), dtype=np.float64).reshape(7)
        action = np.clip(action, -1.0, 1.0)

        next_state = state_vec7 + action * self.cfg.step_size
        if not self.cfg.integrate_policy_jaw:
            next_state[6] = state_vec7[6]
        self._state_vec7 = next_state

        cmd_vec7 = next_state.copy()
        if not self.cfg.use_policy_jaw:
            cmd_vec7[6] = measured.jaw

        command_policy = Pose.from_vec7(cmd_vec7)
        command_robot = self.bridge.to_robot(command_policy)
        command_robot, clamps = self._clamp(measured, command_robot)

        if clamps:
            self._clamp_streak += 1
            if self.cfg.resync_state_on_clamp:
                resynced = self.bridge.to_policy(command_robot).to_vec7()
                resynced[3:6] = self._on_branch(
                    resynced[3:6], reference=self._goal_rpy
                )
                resynced[6] = next_state[6]
                self._state_vec7 = resynced
        else:
            self._clamp_streak = 0

        # -- what the arm did, and what the policy believes ------------------
        trans_err = translation_error_cm(measured, self.goal_robot)
        rot_err = rotation_error_rad(measured, self.goal_robot)

        # SurgicAI's criteria() compares the *integrator* to the goal, and the
        # integrator is what makes the RPY metric meaningful: it runs free, so
        # a wrist that rotates past +pi in yaw simply keeps counting.  Deriving
        # the same number from a pose -- i.e. through a rotation matrix -- wraps
        # that yaw back and reports a ~2*pi error for a millimetre of motion.
        # The Place demonstrations sit right on that boundary (goal yaw is
        # +-3.13 rad), so measuring them the wrong way turns a 97% policy into a
        # 48% one.  This is the branch defect again, one channel over.
        state_pose = Pose.from_vec7(next_state)
        rpy_norm_err = float(np.linalg.norm(next_state[3:6] - goal_vec7[3:6]))
        state_trans_err = float(
            np.linalg.norm(next_state[:3] - goal_vec7[:3]) * 100.0
        )
        state_rot_err = float(
            np.degrees(rotation_error_rad(state_pose, self.goal_policy))
        )

        # Two internally consistent ways to call it done, never mixed:
        #   rpy_norm -- reproduce criteria() exactly, on the integrator
        #   geodesic -- physical truth, on the measured pose
        if self.cfg.rot_metric == "rpy_norm":
            success = (
                state_trans_err <= self.cfg.success_trans_cm
                and rpy_norm_err <= self.cfg.success_rot_rad
            )
        else:
            success = (
                trans_err <= self.cfg.success_trans_cm
                and rot_err <= self.cfg.success_rot_rad
            )

        done, reason = False, "running"
        if success:
            done, reason = True, "success"
        elif (
            self.limits.max_consecutive_clamps
            # Only meaningful under the open-loop observation. There, the
            # policy's state free-runs while the clamp holds the arm back, and
            # sustained clamping means the two have silently stopped agreeing.
            # A closed-loop servo sees the clamped pose every cycle, so it has
            # not diverged from anything -- and a deliberately rate-limited
            # segment, like the slow descent onto the needle, is clamped on
            # purpose every single cycle.
            and self.cfg.observation_source == "command"
            and self._clamp_streak >= self.limits.max_consecutive_clamps
        ):
            kinds = sorted({c["kind"] for c in clamps})
            done, reason = True, (
                f"abort: safety clamp active for {self._clamp_streak} consecutive "
                f"cycles ({', '.join(kinds)}). The controller and the arm have "
                "stopped agreeing about where the tool is."
            )
        elif self.index >= self.cfg.max_steps:
            done, reason = True, "max_steps"

        self.last_command = command_robot
        return StepResult(
            self.index, action.astype(np.float32), command_robot, measured,
            trans_err, float(np.degrees(rot_err)), done, reason, clamps,
            state_trans_err_cm=state_trans_err,
            state_rot_err_deg=state_rot_err,
            rpy_norm_err_rad=rpy_norm_err,
            state_vec7=next_state.copy(),
        )
