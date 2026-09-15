"""The approach -> close -> observe -> lift state machine.

No ROS, no AMBF, no torch.  The ROS node, the offline replay tool and the
simulation env all drive this same object, so what is validated offline is
literally the code that runs on the robot.

Phases
------
``approach``
    Drive to the grasp pose with the jaw held open.  This is the existing,
    contract-verified :class:`~.loop.ApproachLoop`, unchanged, with whichever
    controller the operator chose (``rl`` / ``d2`` / ``residual``).
``settle``
    Stop.  Hold station at the grasp pose and require the *measured* pose to
    stop moving for several cycles.  Closing a gripper while the wrist is still
    drifting is how a needle gets flicked off the pad.
``close``
    Ramp the jaw command from open to the squeeze angle, a few degrees per
    cycle, while station-keeping on the grasp pose.
``observe``
    Hold everything still and watch the jaw.  See :mod:`.jaw` -- what comes out
    is an observation, never a confirmed grasp.  At the end of this phase the
    *gate* decides whether the lift is allowed to start.
``lift``
    Translate to the lift pose (1.5 cm by default), orientation frozen at the
    grasp orientation, jaw held at the squeeze angle.
``hold``
    Keep station at the lift pose for a few cycles and take a final reading.
``done`` / ``aborted``
    Terminal.  On abort the last command is held and **the jaw is not
    opened** -- opening a gripper that may be holding a needle several
    centimetres above the tissue is not a safe default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .controllers import D2Controller
from .frames import Pose, rotation_error_rad, translation_error_cm
from .jaw import (
    JawBaseline,
    JawCalibration,
    JawEvidence,
    JawEvidenceWindow,
    evaluate_jaw_evidence,
)
from .loop import ApproachLoop, LoopConfig, SafetyLimits
from .plan import GraspLiftPlan

PHASE_APPROACH = "approach"
PHASE_SETTLE = "settle"
PHASE_CLOSE = "close"
PHASE_OBSERVE = "observe"
PHASE_WAIT_OPERATOR = "wait_operator"
PHASE_LIFT = "lift"
PHASE_HOLD = "hold"
PHASE_DONE = "done"
PHASE_ABORTED = "aborted"

TERMINAL_PHASES = (PHASE_DONE, PHASE_ABORTED)

#: the jaw command moves this far per control cycle, mirroring the training
#: contract's 0.05 normalised jaw step (0.05 * 60 deg = 3 deg)
DEFAULT_JAW_RAMP_RAD = float(np.deg2rad(3.0))


@dataclass
class ArmState:
    """Everything one control cycle knows about the arm."""

    #: measured Cartesian pose in the robot frame; ``pose.jaw`` is normalised
    pose: Pose
    #: measured jaw angle in radians, or None if jaw/measured_js is silent
    jaw_rad: Optional[float] = None
    #: |effort| from jaw/measured_js, or None if the arm publishes none
    jaw_effort: Optional[float] = None


@dataclass
class Command:
    """What the node should publish this cycle."""

    pose: Pose
    jaw_rad: float
    publish_pose: bool = True
    publish_jaw: bool = True


@dataclass
class SequenceStep:
    index: int
    phase: str
    phase_step: int
    command: Command
    measured: ArmState
    trans_err_cm: float
    rot_err_deg: float
    action: np.ndarray
    jaw_evidence: Optional[JawEvidence]
    clamps: list = field(default_factory=list)
    events: list = field(default_factory=list)
    done: bool = False
    reason: str = "running"

    def as_dict(self) -> dict:
        return {
            "i": self.index,
            "phase": self.phase,
            "phase_step": self.phase_step,
            "measured_cm": (self.measured.pose.p * 100.0).tolist(),
            "command_cm": (self.command.pose.p * 100.0).tolist(),
            "command_jaw_deg": float(np.degrees(self.command.jaw_rad)),
            "measured_jaw_deg": (
                None if self.measured.jaw_rad is None
                else float(np.degrees(self.measured.jaw_rad))
            ),
            "trans_err_cm": self.trans_err_cm,
            "rot_err_deg": self.rot_err_deg,
            "action": np.asarray(self.action, dtype=float).round(4).tolist(),
            "jaw": None if self.jaw_evidence is None else self.jaw_evidence.as_dict(),
            "clamps": self.clamps,
            "events": self.events,
            "reason": self.reason,
        }


@dataclass
class SequenceConfig:
    # -- approach ----------------------------------------------------------
    frame_mode: str = "rebase"
    approach_max_steps: int = 200
    approach_success_trans_cm: float = 1.0
    approach_success_rot_deg: float = 10.0

    # -- settle ------------------------------------------------------------
    #: half-window length: the mean pose over the last ``settle_steps`` cycles
    #: is compared against the mean over the ``settle_steps`` before that
    settle_steps: int = 5
    #: how far the arm may have *drifted* between those two windows.  This is a
    #: drift test, not a per-cycle motion test: comparing consecutive samples
    #: makes measurement noise look like motion, and an arm whose measured_cp
    #: carries a few tenths of a millimetre of noise then never settles at all.
    settle_translation_tol_mm: float = 0.5
    settle_rotation_tol_deg: float = 0.5
    settle_timeout_steps: int = 60

    # -- close -------------------------------------------------------------
    jaw_ramp_rad: float = DEFAULT_JAW_RAMP_RAD
    close_dwell_steps: int = 5
    close_timeout_steps: int = 80

    # -- observe -----------------------------------------------------------
    observe_steps: int = 10
    #: how many consecutive blocked readings the evidence gate needs
    evidence_streak: int = 3
    #: 'manual' | 'evidence' | 'always' | 'never'
    grasp_gate: str = "manual"
    operator_timeout_steps: int = 0  # 0 = wait indefinitely

    # -- lift --------------------------------------------------------------
    lift_max_steps: int = 120
    lift_success_trans_cm: float = 0.2
    lift_success_rot_deg: float = 10.0
    hold_steps: int = 10
    #: consecutive non-blocked readings, after evidence was established, that
    #: count as the needle having been lost
    slip_streak: int = 4
    #: 'abort' | 'continue' | 'lower'
    on_slip: str = "abort"

    # -- the policy's action contract --------------------------------------
    #: applied as cmd = measured + action * step_size, raw units.  Recover it
    #: from a checkpoint with tools/recover_step_size.py rather than trusting
    #: the default: the upstream Approach checkpoint uses 0.5 mm / 2 deg where
    #: R6 uses 1.5 mm / 3 deg, and the wrong scale makes a policy diverge.
    step_size: Optional[np.ndarray] = None
    #: goal RPY on the training 2*pi branch; None means the trained goal's own
    goal_rpy_train: Optional[tuple] = None
    unwrap_rpy: bool = True

    # -- evidence thresholds -----------------------------------------------
    residual_margin_deg: float = 1.0
    effort_margin: Optional[float] = None

    def __post_init__(self):
        if self.grasp_gate not in ("manual", "evidence", "always", "never"):
            raise ValueError(f"unknown grasp gate {self.grasp_gate!r}")
        if self.on_slip not in ("abort", "continue", "lower"):
            raise ValueError(f"unknown on_slip policy {self.on_slip!r}")
        for name in ("settle_steps", "close_dwell_steps", "observe_steps",
                     "evidence_streak", "slip_streak"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.jaw_ramp_rad <= 0.0:
            raise ValueError("jaw_ramp_rad must be positive")


class GraspLiftSequencer:
    """Runs one grasp-and-lift episode, one :meth:`step` per control cycle."""

    def __init__(
        self,
        plan: GraspLiftPlan,
        approach_controller,
        config: Optional[SequenceConfig] = None,
        limits: Optional[SafetyLimits] = None,
        jaw_baseline: Optional[JawBaseline] = None,
        *,
        confirm_callback: Optional[Callable[[], Optional[bool]]] = None,
    ):
        self.plan = plan
        self.cfg = config or SequenceConfig()
        self.limits = limits or SafetyLimits()
        self.baseline = jaw_baseline
        self.confirm_callback = confirm_callback
        self.jaw: JawCalibration = plan.jaw

        self.approach_controller = approach_controller
        #: close, observe and lift are driven by the geometric servo in the raw
        #: robot frame.  They are short, near-goal, orientation-frozen motions
        #: and safety-critical; a learned policy buys nothing here and the
        #: released one was never trained on them.
        self.hold_controller = D2Controller(staged=True)

        self.phase = PHASE_APPROACH
        self.index = 0
        self.phase_step = 0
        self.reason = "running"

        self._approach_loop: Optional[ApproachLoop] = None
        self._segment_loop: Optional[ApproachLoop] = None
        self._segment_goal: Optional[Pose] = None

        self.jaw_command_rad = float(self.jaw.approach_open_rad)
        self.window = JawEvidenceWindow(required_streak=self.cfg.evidence_streak)
        self.evidence_established = False
        self.slip_count = 0
        self._settle_streak = 0
        self._lowering = False
        self._pose_window: list = []
        self.last_settle_drift_mm: Optional[float] = None
        self._prev_measured: Optional[Pose] = None
        self._last_command: Optional[Command] = None
        self.grasp_pose_measured: Optional[Pose] = None
        self.events: list = []
        self.gate_decision: Optional[dict] = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def begin(self, start: ArmState) -> dict:
        cfg = LoopConfig(
            frame_mode=self.cfg.frame_mode,
            goal_orientation="explicit",
            goal_quat_xyzw=tuple(self.plan.grasp.quat_xyzw()),
            goal_jaw=str(self.plan.grasp.jaw),
            use_policy_jaw=False,
            max_steps=self.cfg.approach_max_steps,
            success_trans_cm=self.cfg.approach_success_trans_cm,
            success_rot_rad=float(np.deg2rad(self.cfg.approach_success_rot_deg)),
            goal_rpy_train=self.cfg.goal_rpy_train,
            unwrap_rpy=self.cfg.unwrap_rpy,
            **({} if self.cfg.step_size is None
               else {"step_size": np.asarray(self.cfg.step_size, dtype=np.float64)}),
        )
        self._approach_loop = ApproachLoop(self.approach_controller, cfg, self.limits)
        report = self._approach_loop.begin(start.pose, self.plan.grasp.p)
        self._segment_goal = self.plan.grasp
        self.jaw_command_rad = float(self.jaw.approach_open_rad)
        self._prev_measured = start.pose
        return report

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _enter(self, phase: str, note: str = ""):
        self.events.append(
            {"i": self.index, "from": self.phase, "to": phase, "note": note}
        )
        self.phase = phase
        self.phase_step = 0
        self._settle_streak = 0
        self._pose_window = []

    def _start_segment(self, measured: Pose, goal: Pose, max_steps: int,
                       success_trans_cm: float, success_rot_deg: float):
        """A geometric servo segment in the raw robot frame."""
        cfg = LoopConfig(
            frame_mode="identity",
            goal_orientation="explicit",
            goal_quat_xyzw=tuple(goal.quat_xyzw()),
            use_policy_jaw=False,
            max_steps=max_steps,
            success_trans_cm=success_trans_cm,
            success_rot_rad=float(np.deg2rad(success_rot_deg)),
            unwrap_rpy=self.cfg.unwrap_rpy,
        )
        loop = ApproachLoop(self.hold_controller, cfg, self.limits)
        loop.begin(measured, goal.p)
        self._segment_loop = loop
        self._segment_goal = goal
        return loop

    def _jaw_evidence(self, measured: ArmState) -> JawEvidence:
        return evaluate_jaw_evidence(
            self.jaw_command_rad,
            measured.jaw_rad,
            measured.jaw_effort,
            self.baseline,
            residual_margin_rad=float(np.deg2rad(self.cfg.residual_margin_deg)),
            effort_margin=self.cfg.effort_margin,
        )

    def _stationary(self, measured: Pose) -> bool:
        """Has the arm stopped *drifting*?

        Two consecutive samples cannot answer that: on an arm whose
        ``measured_cp`` carries a few tenths of a millimetre of noise, the
        per-cycle difference never falls below any useful threshold and the
        settle phase times out on a perfectly stationary arm.  So the mean of
        the last ``settle_steps`` samples is compared against the mean of the
        ``settle_steps`` before it, which averages the noise down and leaves
        only real motion.
        """
        window = self.cfg.settle_steps
        self._pose_window.append(measured)
        if len(self._pose_window) > 2 * window:
            self._pose_window.pop(0)
        if len(self._pose_window) < 2 * window:
            return False

        older = self._pose_window[:window]
        newer = self._pose_window[window:]
        drift_mm = float(
            np.linalg.norm(
                np.mean([p.p for p in newer], axis=0)
                - np.mean([p.p for p in older], axis=0)
            )
            * 1000.0
        )
        turned_deg = float(
            np.degrees(rotation_error_rad(older[-1], newer[-1]))
        )
        self.last_settle_drift_mm = drift_mm
        return (
            drift_mm <= self.cfg.settle_translation_tol_mm
            and turned_deg <= self.cfg.settle_rotation_tol_deg
        )

    def _hold_command(self, measured: ArmState):
        """Station-keep on the current segment goal; returns (command, action, clamps)."""
        loop = self._segment_loop
        result = loop.step(measured.pose)
        return result

    def _finish(self, phase: str, reason: str):
        self._enter(phase, reason)
        self.reason = reason

    # ------------------------------------------------------------------
    # the cycle
    # ------------------------------------------------------------------
    def step(self, measured: ArmState) -> SequenceStep:
        if self._approach_loop is None:
            raise RuntimeError("call begin() before step()")

        self.index += 1
        self.phase_step += 1
        events: list = []
        clamps: list = []
        action = np.zeros(7)
        evidence: Optional[JawEvidence] = None
        entry_phase = self.phase

        # The jaw is watched in every phase from the close onwards, so a needle
        # lost during the lift shows up in the same record as the grasp.
        if self.phase in (PHASE_CLOSE, PHASE_OBSERVE, PHASE_WAIT_OPERATOR,
                          PHASE_LIFT, PHASE_HOLD):
            evidence = self._jaw_evidence(measured)
            self.window.update(evidence)

        # ------------------------------------------------------------------
        if self.phase == PHASE_APPROACH:
            result = self._approach_loop.step(measured.pose)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw.approach_open_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            if result.reason == "success":
                self.grasp_pose_measured = measured.pose
                self._start_segment(
                    measured.pose,
                    Pose(self.plan.grasp.p, self.plan.grasp.R, self.plan.grasp.jaw),
                    max_steps=10_000,
                    success_trans_cm=self.cfg.approach_success_trans_cm,
                    success_rot_deg=self.cfg.approach_success_rot_deg,
                )
                self._enter(PHASE_SETTLE, "approach reached the grasp pose")
            elif result.done:
                self._finish(PHASE_ABORTED, f"approach {result.reason}")

        # ------------------------------------------------------------------
        elif self.phase == PHASE_SETTLE:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw.approach_open_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            in_tolerance = (
                trans_err <= self.cfg.approach_success_trans_cm
                and rot_err <= self.cfg.approach_success_rot_deg
            )
            if self._stationary(measured.pose) and in_tolerance:
                self._settle_streak += 1
            else:
                self._settle_streak = 0

            if self._settle_streak >= self.cfg.settle_steps:
                self.grasp_pose_measured = measured.pose
                events.append({"i": self.index, "event": "settled",
                               "trans_err_cm": trans_err, "rot_err_deg": rot_err,
                               "drift_mm": self.last_settle_drift_mm})
                self._enter(PHASE_CLOSE, "arm stationary at the grasp pose")
            elif self.phase_step >= self.cfg.settle_timeout_steps:
                drift = (
                    "unknown" if self.last_settle_drift_mm is None
                    else f"{self.last_settle_drift_mm:.2f} mm"
                )
                self._finish(
                    PHASE_ABORTED,
                    f"settle timeout: the arm never held still within tolerance "
                    f"(goal error {trans_err:.2f} cm / {rot_err:.1f} deg, drift "
                    f"{drift} against a {self.cfg.settle_translation_tol_mm:.2f} mm "
                    "limit)",
                )

        # ------------------------------------------------------------------
        elif self.phase == PHASE_CLOSE:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            target = float(self.jaw.grip_rad)
            delta = target - self.jaw_command_rad
            stepped = float(np.clip(delta, -self.cfg.jaw_ramp_rad, self.cfg.jaw_ramp_rad))
            self.jaw_command_rad += stepped
            at_target = abs(target - self.jaw_command_rad) <= 1e-9
            command = Command(result.command, self.jaw_command_rad)

            if at_target:
                self._settle_streak += 1
                if self._settle_streak >= self.cfg.close_dwell_steps:
                    events.append(
                        {
                            "i": self.index,
                            "event": "jaw_at_grip_command",
                            "grip_deg": float(np.degrees(self.jaw.grip_rad)),
                        }
                    )
                    self.window.reset()
                    self._enter(PHASE_OBSERVE, "jaw held at the squeeze angle")
            elif self.phase_step >= self.cfg.close_timeout_steps:
                self._finish(PHASE_ABORTED, "close timeout: the jaw ramp never finished")

        # ------------------------------------------------------------------
        elif self.phase == PHASE_OBSERVE:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw_command_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            if self.phase_step >= self.cfg.observe_steps:
                self.evidence_established = bool(self.window.blocked_streak_met)
                decision = self._gate_decision()
                self.gate_decision = decision
                events.append({"i": self.index, "event": "grasp_gate", **decision})
                if decision["action"] == "lift":
                    self._start_segment(
                        measured.pose,
                        self.plan.lifted,
                        max_steps=self.cfg.lift_max_steps,
                        success_trans_cm=self.cfg.lift_success_trans_cm,
                        success_rot_deg=self.cfg.lift_success_rot_deg,
                    )
                    self._enter(PHASE_LIFT, decision["reason"])
                elif decision["action"] == "wait":
                    self._enter(PHASE_WAIT_OPERATOR, decision["reason"])
                elif decision["action"] == "stop":
                    self._finish(PHASE_DONE, decision["reason"])
                else:
                    self._finish(PHASE_ABORTED, decision["reason"])

        # ------------------------------------------------------------------
        elif self.phase == PHASE_WAIT_OPERATOR:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw_command_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            answer = None if self.confirm_callback is None else self.confirm_callback()
            if answer is True:
                events.append({"i": self.index, "event": "operator_confirmed"})
                self._start_segment(
                    measured.pose,
                    self.plan.lifted,
                    max_steps=self.cfg.lift_max_steps,
                    success_trans_cm=self.cfg.lift_success_trans_cm,
                    success_rot_deg=self.cfg.lift_success_rot_deg,
                )
                self._enter(PHASE_LIFT, "operator released the lift")
            elif answer is False:
                self._finish(PHASE_DONE, "operator declined the lift")
            elif (
                self.cfg.operator_timeout_steps
                and self.phase_step >= self.cfg.operator_timeout_steps
            ):
                self._finish(PHASE_ABORTED, "operator confirmation timed out")

        # ------------------------------------------------------------------
        elif self.phase == PHASE_LIFT:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw_command_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            slip = self._check_slip(evidence, events)
            if slip == "abort":
                self._finish(PHASE_ABORTED,
                             "jaw evidence disappeared during the lift")
            elif slip == "lower":
                # Put the gripper back where it picked up, then stop.  Slip
                # monitoring is switched off for the descent so a second
                # trigger cannot bounce the arm up and down.
                self._lowering = True
                self.evidence_established = False
                self._start_segment(
                    measured.pose,
                    Pose(self.plan.grasp.p, self.plan.grasp.R, self.plan.grasp.jaw),
                    max_steps=self.cfg.lift_max_steps,
                    success_trans_cm=self.cfg.lift_success_trans_cm,
                    success_rot_deg=self.cfg.lift_success_rot_deg,
                )
            elif self._lowering and result.reason == "success":
                self._finish(
                    PHASE_DONE, "lowered back to the grasp pose after losing jaw evidence"
                )
            elif result.reason == "success":
                events.append(
                    {"i": self.index, "event": "lift_reached",
                     "height_cm": float(
                         np.linalg.norm(measured.pose.p - self.plan.grasp.p) * 100.0
                     )}
                )
                self._enter(PHASE_HOLD, "lift reached")
            elif result.done:
                self._finish(PHASE_ABORTED, f"lift {result.reason}")

        # ------------------------------------------------------------------
        elif self.phase == PHASE_HOLD:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw_command_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg
            self._check_slip(evidence, events)
            if self.phase_step >= self.cfg.hold_steps:
                self._finish(PHASE_DONE, "success")

        # ------------------------------------------------------------------
        else:  # terminal: hold the last command, never open the jaw
            command = self._last_command or Command(measured.pose, self.jaw_command_rad)
            command = Command(command.pose, command.jaw_rad, False, False)
            trans_err = translation_error_cm(measured.pose, self._segment_goal or self.plan.grasp)
            rot_err = float(
                np.degrees(rotation_error_rad(measured.pose, self._segment_goal or self.plan.grasp))
            )

        self._last_command = command
        self._prev_measured = measured.pose
        done = self.phase in TERMINAL_PHASES

        return SequenceStep(
            index=self.index,
            phase=entry_phase,
            phase_step=self.phase_step,
            command=command,
            measured=measured,
            trans_err_cm=float(trans_err),
            rot_err_deg=float(rot_err),
            action=np.asarray(action, dtype=np.float64),
            jaw_evidence=evidence,
            clamps=clamps,
            events=events + [e for e in self.events if e.get("i") == self.index
                             and e not in events],
            done=done,
            reason=self.reason if done else "running",
        )

    # ------------------------------------------------------------------
    def _gate_decision(self) -> dict:
        gate = self.cfg.grasp_gate
        summary = self.window.summary()
        if gate == "never":
            return {
                "gate": gate,
                "action": "stop",
                "reason": "stopped after the jaw closed, as requested",
                "evidence": summary,
            }
        if gate == "always":
            return {
                "gate": gate,
                "action": "lift",
                "reason": "gate 'always': lifting without requiring any evidence",
                "evidence": summary,
            }
        if gate == "manual":
            return {
                "gate": gate,
                "action": "wait",
                "reason": "waiting for a human to confirm the needle is held",
                "evidence": summary,
            }
        # evidence
        if self.evidence_established:
            return {
                "gate": gate,
                "action": "lift",
                "reason": (
                    f"the jaw stopped early on {summary['current_streak']} "
                    "consecutive cycles. This is consistent with something "
                    "between the fingers; it is NOT a confirmed grasp."
                ),
                "evidence": summary,
            }
        return {
            "gate": gate,
            "action": "abort",
            "reason": (
                "the jaw closed like an empty gripper, so there is no evidence "
                "anything was picked up"
            ),
            "evidence": summary,
        }

    def _check_slip(self, evidence: Optional[JawEvidence], events: list) -> Optional[str]:
        """Watch for the evidence going away after it had been established."""
        if not self.evidence_established or evidence is None:
            return None
        if evidence.jaw_blocked:
            self.slip_count = 0
            return None
        self.slip_count += 1
        if self.slip_count < self.cfg.slip_streak:
            return None
        events.append(
            {
                "i": self.index,
                "event": "jaw_evidence_lost",
                "consecutive_cycles": self.slip_count,
                "policy": self.cfg.on_slip,
                "note": (
                    "the jaw has closed to where an empty jaw closes. The needle "
                    "was probably dropped -- but this channel could never prove "
                    "it was held in the first place."
                ),
            }
        )
        self.slip_count = 0
        return None if self.cfg.on_slip == "continue" else self.cfg.on_slip

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "phase": self.phase,
            "reason": self.reason,
            "steps": self.index,
            "grasp_gate": self.cfg.grasp_gate,
            "gate_decision": self.gate_decision,
            "jaw_evidence": self.window.summary(),
            "grasp_verified": False,
            "grasp_verification_note": (
                "This deployment has no grasp sensor. Success here means the "
                "commanded sequence completed, not that a needle was picked up."
            ),
            "events": self.events,
        }
