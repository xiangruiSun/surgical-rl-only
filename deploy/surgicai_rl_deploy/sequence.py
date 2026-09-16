"""The stage -> approach -> grasp -> lift -> transport -> place state machine.

No ROS, no AMBF, no torch.  The ROS node, the offline replay tool and the
simulation env all drive this same object, so what is validated offline is
literally the code that runs on the robot.

The phases are data-driven off the plan: a plan with no ``staged`` pose starts
at ``approach``, and one with no ``suture`` pose ends after the lift.  The same
class therefore runs a bare grasp-and-lift and the full suturing pipeline, and
Insert or Pullout drop in as two more segments when someone decides that is a
safe thing to do.

Phases
------
``stage``
    Servo from wherever the arm is to the pose that puts the approach policy
    inside its own training support (see :mod:`.staging`).  Deterministic, no
    policy involved -- the policy cannot be trusted to get itself in
    distribution, since being out of it is the problem.
``approach``
    Drive to the grasp pose with the jaw held open.  This is the existing,
    contract-verified :class:`~.loop.ApproachLoop`, unchanged, with whichever
    controller the operator chose (``rl`` / ``d2`` / ``residual``).
``descend``
    Close the standoff.  The approach policy aims 7 mm short of the needle,
    along the tool's own axis, because that is where its training goal is
    (``scene_manager.needle_goal_evaluator``'s ``lift_height``); in simulation
    the grasp is then faked, so nothing there ever had to travel the last
    7 mm.  On hardware something must, slowly and straight down the jaw axis.
    Skipped when the plan has no standoff.
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
``transport``
    Carry the needle from the lift pose to a point directly above the suturing
    pose, turning the wrist on the way.  The jaw is watched throughout and the
    arm never descends here.  If a second controller was supplied it runs
    alongside on the same geometry, logged and never published.
``place``
    Descend onto the suturing pose with the orientation already correct.  The
    tightest tolerance in the run, and the only motion that approaches tissue
    with a needle in the jaws.
``hold``
    Keep station wherever the sequence ended for a few cycles and take a final
    reading.
``done`` / ``aborted``
    Terminal.  On abort the last command is held and **the jaw is not
    opened** -- opening a gripper that may be holding a needle several
    centimetres above the tissue is not a safe default.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

PHASE_STAGE = "stage"
PHASE_APPROACH = "approach"
PHASE_DESCEND = "descend"
PHASE_SETTLE = "settle"
PHASE_CLOSE = "close"
PHASE_OBSERVE = "observe"
PHASE_WAIT_OPERATOR = "wait_operator"
PHASE_LIFT = "lift"
PHASE_TRANSPORT = "transport"
PHASE_PLACE = "place"
PHASE_HOLD = "hold"
PHASE_DONE = "done"
PHASE_ABORTED = "aborted"

TERMINAL_PHASES = (PHASE_DONE, PHASE_ABORTED)

#: every phase in which the needle is (believed to be) in the jaws, so the jaw
#: is watched and a slip matters
LOADED_PHASES = (PHASE_LIFT, PHASE_TRANSPORT, PHASE_PLACE, PHASE_HOLD)

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
    # -- stage -------------------------------------------------------------
    #: servo the arm to plan.staged before the policy takes over, so the
    #: approach leg begins inside its own training support
    stage_max_steps: int = 400
    stage_success_trans_cm: float = 0.2
    stage_success_rot_deg: float = 2.0

    # -- approach ----------------------------------------------------------
    frame_mode: str = "rebase"
    approach_max_steps: int = 200
    approach_success_trans_cm: float = 1.0
    approach_success_rot_deg: float = 10.0
    #: what to do when the approach policy does not converge.
    #:   'hold'  -- stop at the last command, jaw untouched, hand back to a human
    #:   'servo' -- finish the approach geometrically and carry on
    #: 'hold' is the default because an approach that failed is, by definition,
    #: an arm that did something nobody predicted.
    on_approach_failure: str = "hold"
    #: the checkpoint contract driving the approach leg, if any.  Supplies the
    #: action scale, the episode budget and the jaw envelope the policy was
    #: trained with, all of which differ per checkpoint.
    approach_contract: object = None

    # -- suturing-pose compensation ----------------------------------------
    #: 'apply' | 'report' | 'off'.  A hand-taught suturing pose encodes how the
    #: needle sat in the jaws when it was recorded; if the jaws close somewhere
    #: else, the placement is wrong by that difference, and the correction is
    #: exact (see GraspLiftPlan.compensated_suture).
    compensate_suture: str = "apply"
    #: a correction larger than this means something happened that the
    #: compensation cannot model -- most likely the needle moved -- so the run
    #: stops rather than confidently placing it somewhere new
    max_suture_compensation_mm: float = 10.0

    # -- transport and place ------------------------------------------------
    transport_max_steps: int = 600
    transport_success_trans_cm: float = 0.3
    transport_success_rot_deg: float = 5.0
    place_max_steps: int = 400
    #: the descent onto the entry point is the tightest motion in the run
    place_success_trans_cm: float = 0.2
    place_success_rot_deg: float = 3.0

    # -- descend -----------------------------------------------------------
    descend_max_steps: int = 200
    descend_success_trans_cm: float = 0.05
    descend_success_rot_deg: float = 2.0
    #: the descent onto the needle is the one motion that can move the needle
    #: before it is held, so it is deliberately slower than everything else
    descend_step_mm: float = 0.5

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
        if self.compensate_suture not in ("apply", "report", "off"):
            raise ValueError(
                f"unknown compensate_suture {self.compensate_suture!r}"
            )
        if self.on_approach_failure not in ("hold", "servo"):
            raise ValueError(
                f"unknown on_approach_failure {self.on_approach_failure!r}"
            )
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
        shadow_controller=None,
        shadow_contract=None,
    ):
        self.plan = plan
        self.cfg = config or SequenceConfig()
        #: an optional second controller run alongside the transport and place
        #: legs on identical geometry.  Its actions are recorded and never
        #: published, so "would the Place policy have worked here" can be
        #: answered from real runs without ever letting it hold the needle.
        self.shadow_controller = shadow_controller
        self.shadow_contract = shadow_contract
        self._shadow_loop: Optional[ApproachLoop] = None
        self.shadow_log: list = []
        self.shadow_support: Optional[dict] = None
        #: the suturing pose actually aimed at, after compensation
        self.suture_target: Optional[Pose] = plan.suture
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
        self._begun = False
        self.approach_report: Optional[dict] = None

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
    def _approach_config(self) -> LoopConfig:
        contract = self.cfg.approach_contract
        cfg = LoopConfig(
            frame_mode=self.cfg.frame_mode,
            goal_orientation="explicit",
            goal_quat_xyzw=tuple(self.plan.approach_target.quat_xyzw()),
            goal_jaw=str(self.plan.approach_target.jaw),
            use_policy_jaw=False,
            max_steps=self.cfg.approach_max_steps,
            success_trans_cm=self.cfg.approach_success_trans_cm,
            success_rot_rad=float(np.deg2rad(self.cfg.approach_success_rot_deg)),
            goal_rpy_train=self.cfg.goal_rpy_train,
            unwrap_rpy=self.cfg.unwrap_rpy,
            **({} if self.cfg.step_size is None
               else {"step_size": np.asarray(self.cfg.step_size, dtype=np.float64)}),
        )
        if contract is not None:
            # The policy's own action scale and jaw envelope, unless the
            # operator overrode the scale on the command line.
            if self.cfg.step_size is None:
                cfg.step_size = np.asarray(contract.step_size, dtype=np.float64)
            # Training closed the jaw during the approach and the jaw is three
            # of the twenty-one observation dimensions; the gripper stays open
            # regardless, because use_policy_jaw is False.
            cfg.policy_jaw_start = float(contract.demo_start_jaw)
            cfg.goal_jaw = str(float(contract.demo_goal_jaw))
        return cfg

    def _begin_approach(self, measured: Pose) -> dict:
        self._approach_loop = ApproachLoop(
            self.approach_controller, self._approach_config(), self.limits,
            contract=self.cfg.approach_contract,
        )
        report = self._approach_loop.begin(measured, self.plan.approach_target.p)
        self.approach_report = report
        self._segment_goal = self.plan.approach_target
        return report

    def begin(self, start: ArmState) -> dict:
        self.jaw_command_rad = float(self.jaw.approach_open_rad)
        self._prev_measured = start.pose

        self._begun = True
        if self.plan.staged is not None:
            # Servo to the staging pose first; the policy is handed an arm that
            # is already inside its training support.  The approach loop is not
            # created until that motion has finished, because it must be seeded
            # from where the arm actually ends up, not from where it was asked
            # to go.
            self.phase = PHASE_STAGE
            self._start_segment(
                start.pose, self.plan.staged,
                max_steps=self.cfg.stage_max_steps,
                success_trans_cm=self.cfg.stage_success_trans_cm,
                success_rot_deg=self.cfg.stage_success_rot_deg,
            )
            from .staging import support_report

            report = {
                "phase": PHASE_STAGE,
                "staging": (
                    None if self.cfg.approach_contract is None
                    else support_report(
                        self.plan.staged, self.plan.approach_target,
                        self.cfg.approach_contract,
                    )
                ),
                "stage_travel_cm": float(
                    np.linalg.norm(self.plan.staged.p - start.pose.p) * 100.0
                ),
                "note": "approach distribution is reported once the arm is staged",
            }
            self.approach_report = None
            return report

        self.phase = PHASE_APPROACH
        return self._begin_approach(start.pose)

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
                       success_trans_cm: float, success_rot_deg: float,
                       limits: Optional[SafetyLimits] = None):
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
            # A geometric servo is supposed to react to where the arm actually
            # is.  The open-loop observation belongs to the policy legs, where
            # it reproduces the training contract; here it would make the servo
            # keep pushing at a target the arm never reached.
            observation_source="measured",
            integrate_policy_jaw=False,
            rot_metric="geodesic",
        )
        loop = ApproachLoop(self.hold_controller, cfg, limits or self.limits)
        loop.begin(measured, goal.p)
        self._segment_loop = loop
        self._segment_goal = goal
        return loop

    def _begin_descent(self, measured: Pose, events: list):
        """Travel the last few millimetres onto the needle, gently."""
        # The descent cannot be gentler than the arm's deadband: a per-step cap
        # below the floor leaves every command stretched straight back up to
        # the floor, and the two fight until the segment times out.
        slow = replace(
            self.limits,
            max_step_translation_mm=max(
                self.cfg.descend_step_mm, self.limits.min_command_mm
            ),
        )
        self._start_segment(
            measured, Pose(self.plan.grasp.p, self.plan.grasp.R, self.plan.grasp.jaw),
            max_steps=self.cfg.descend_max_steps,
            success_trans_cm=self.cfg.descend_success_trans_cm,
            success_rot_deg=self.cfg.descend_success_rot_deg,
            limits=slow,
        )
        events.append({
            "i": self.index, "event": "descend_begin",
            "standoff_mm": self.plan.grasp_standoff_m * 1000.0,
            "step_mm": self.cfg.descend_step_mm,
        })
        self._enter(PHASE_DESCEND, "closing the standoff onto the needle")

    def _resolve_suture(self, events: list):
        """Correct the taught suturing pose for where the jaws actually closed."""
        self.suture_target = self.plan.suture
        if self.grasp_pose_measured is None or self.cfg.compensate_suture == "off":
            return True

        corrected = self.plan.compensated_suture(self.grasp_pose_measured)
        shift_mm = float(np.linalg.norm(corrected.p - self.plan.suture.p) * 1000.0)
        turn_deg = float(np.degrees(rotation_error_rad(corrected, self.plan.suture)))
        record = {
            "i": self.index,
            "event": "suture_compensation",
            "mode": self.cfg.compensate_suture,
            "shift_mm": shift_mm,
            "turn_deg": turn_deg,
            "taught_grasp_cm": (
                (self.plan.taught_grasp or self.plan.grasp).p * 100.0
            ).tolist(),
            "measured_grasp_cm": (self.grasp_pose_measured.p * 100.0).tolist(),
            "note": (
                "the suturing pose was taught with one grasp; the jaws closed at "
                "another, so the needle sits differently in them. This corrects "
                "for that exactly, ASSUMING the needle itself did not move."
            ),
        }
        events.append(record)

        if shift_mm > self.cfg.max_suture_compensation_mm:
            self._finish(
                PHASE_ABORTED,
                f"the grasp landed {shift_mm:.1f} mm from where the suturing "
                f"pose was taught, past the {self.cfg.max_suture_compensation_mm:.1f} "
                "mm limit. Either the arm missed badly or the needle moved; "
                "either way the taught suturing pose no longer describes where "
                "this needle has to go.",
            )
            return False

        if self.cfg.compensate_suture == "apply":
            self.suture_target = corrected
        return True

    def _begin_transport(self, measured: Pose, events: list):
        """Leave the lift pose for the suturing point, with the needle held."""
        if not self._resolve_suture(events):
            return
        target = self.plan.via_for(self.suture_target) or self.suture_target
        self._start_segment(
            measured, target,
            max_steps=self.cfg.transport_max_steps,
            success_trans_cm=self.cfg.transport_success_trans_cm,
            success_rot_deg=self.cfg.transport_success_rot_deg,
        )
        self._start_shadow(measured, events)
        events.append({
            "i": self.index, "event": "transport_begin",
            "target": ("via" if self.plan.via_for(self.suture_target) is not None
                       else "suture"),
            "travel_cm": float(np.linalg.norm(target.p - measured.p) * 100.0),
            "rotation_deg": float(
                np.degrees(rotation_error_rad(Pose(measured.p, measured.R, 0.0), target))
            ),
        })
        self._enter(PHASE_TRANSPORT, "carrying the needle to the suturing point")

    # ------------------------------------------------------------------
    # the shadow policy: measured, never in command
    # ------------------------------------------------------------------
    def _start_shadow(self, measured: Pose, events: list):
        if self.shadow_controller is None or self.plan.suture is None:
            return
        contract = self.shadow_contract
        cfg = LoopConfig(
            frame_mode=self.cfg.frame_mode,
            goal_orientation="explicit",
            goal_quat_xyzw=tuple(self.suture_target.quat_xyzw()),
            goal_jaw=str(self.suture_target.jaw),
            use_policy_jaw=False,
            max_steps=self.cfg.transport_max_steps + self.cfg.place_max_steps,
            success_trans_cm=(
                0.5 if contract is None else contract.success_trans_cm
            ),
            success_rot_rad=(
                float(np.deg2rad(30.0)) if contract is None
                else contract.success_rot_rad
            ),
            # A shadow is a one-step advisor, not a rollout: at each pose the
            # arm actually reached, what would this policy have commanded next?
            # So its observation comes from the measured pose. A free-running
            # integrator would be answering a question about a trajectory that
            # never happened -- and that counterfactual is better run offline
            # from the logged start with tools/replay_demos.py, where it can be
            # repeated and varied.
            observation_source="measured",
            integrate_policy_jaw=False,
            **({} if contract is None
               else {"step_size": np.asarray(contract.step_size, dtype=np.float64),
                     "policy_jaw_start": float(contract.demo_start_jaw)}),
        )
        try:
            loop = ApproachLoop(self.shadow_controller, cfg, self.limits,
                                contract=contract)
            report = loop.begin(measured, self.suture_target.p)
        except Exception as exc:  # a shadow must never take down the real run
            events.append({"i": self.index, "event": "shadow_unavailable",
                           "error": f"{type(exc).__name__}: {exc}"})
            self._shadow_loop = None
            return
        self._shadow_loop = loop
        self.shadow_support = {
            "in_distribution": report.get("in_distribution"),
            "out_of_distribution": report.get("out_of_distribution") or [],
        }
        events.append({"i": self.index, "event": "shadow_begin",
                       "controller": self.shadow_controller.describe(),
                       **self.shadow_support})

    def _shadow_step(self, measured: ArmState, events: list):
        """Ask the shadow what it would command from here, and write it down."""
        if self._shadow_loop is None:
            return
        try:
            result = self._shadow_loop.step(measured.pose)
        except Exception as exc:
            events.append({"i": self.index, "event": "shadow_failed",
                           "error": f"{type(exc).__name__}: {exc}"})
            self._shadow_loop = None
            return

        real = self._last_command
        divergence_mm = divergence_deg = None
        if real is not None:
            divergence_mm = float(
                np.linalg.norm(result.command.p - real.pose.p) * 1000.0
            )
            divergence_deg = float(
                np.degrees(rotation_error_rad(result.command, real.pose))
            )
        self.shadow_log.append({
            "i": self.index,
            "phase": self.phase,
            "action": np.asarray(result.action, dtype=float).round(4).tolist(),
            "proposed_cm": (result.command.p * 100.0).tolist(),
            "divergence_from_commanded_mm": divergence_mm,
            "divergence_from_commanded_deg": divergence_deg,
            "clamped": [c["kind"] for c in result.clamps],
        })
        # A one-step advisor cannot finish anything, so the loop's own
        # termination is ignored -- except for a clamp abort, which means its
        # advice had stopped being usable and is worth recording once.
        if result.done and "clamp" in result.reason:
            events.append({"i": self.index, "event": "shadow_unusable",
                           "reason": result.reason})
            self._shadow_loop = None

    def shadow_summary(self) -> Optional[dict]:
        if not self.shadow_log:
            return None
        mm = [r["divergence_from_commanded_mm"] for r in self.shadow_log
              if r["divergence_from_commanded_mm"] is not None]
        deg = [r["divergence_from_commanded_deg"] for r in self.shadow_log
               if r["divergence_from_commanded_deg"] is not None]
        return {
            "cycles": len(self.shadow_log),
            "median_divergence_mm": float(np.median(mm)) if mm else None,
            "max_divergence_mm": float(np.max(mm)) if mm else None,
            "median_divergence_deg": float(np.median(deg)) if deg else None,
            "max_divergence_deg": float(np.max(deg)) if deg else None,
            "cycles_clamped": sum(1 for r in self.shadow_log if r["clamped"]),
            "support": getattr(self, "shadow_support", None),
            "note": (
                "One-step advice at the poses the arm actually visited: at each "
                "cycle, what this controller would have commanded next. It is "
                "NOT a rollout -- had it been driving, the arm would have been "
                "somewhere else. For the counterfactual, replay it offline from "
                "the logged transport start with tools/replay_demos.py."
            ),
        }

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
        if not self._begun:
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
                          *LOADED_PHASES):
            evidence = self._jaw_evidence(measured)
            self.window.update(evidence)

        # ------------------------------------------------------------------
        if self.phase == PHASE_STAGE:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw.approach_open_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            if result.reason == "success":
                report = self._begin_approach(measured.pose)
                events.append({
                    "i": self.index, "event": "staged",
                    "trans_err_cm": trans_err, "rot_err_deg": rot_err,
                    "in_distribution": report.get("in_distribution"),
                    "out_of_distribution": report.get("out_of_distribution"),
                })
                self._enter(PHASE_APPROACH, "arm staged inside the training support")
            elif result.done:
                self._finish(
                    PHASE_ABORTED,
                    f"stage {result.reason}: the arm never reached the staging "
                    "pose, so the approach policy was never started",
                )

        # ------------------------------------------------------------------
        elif self.phase == PHASE_APPROACH:
            result = self._approach_loop.step(measured.pose)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw.approach_open_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            if result.reason == "success":
                self.grasp_pose_measured = measured.pose
                if self.plan.hover is not None:
                    self._begin_descent(measured.pose, events)
                else:
                    self._start_segment(
                        measured.pose,
                        Pose(self.plan.grasp.p, self.plan.grasp.R, self.plan.grasp.jaw),
                        max_steps=10_000,
                        success_trans_cm=self.cfg.approach_success_trans_cm,
                        success_rot_deg=self.cfg.approach_success_rot_deg,
                    )
                    self._enter(PHASE_SETTLE, "approach reached the grasp pose")
            elif result.done:
                if self.cfg.on_approach_failure == "servo":
                    events.append({
                        "i": self.index, "event": "approach_fallback",
                        "approach_reason": result.reason,
                        "note": "the policy did not converge; the geometric "
                                "servo is finishing the approach",
                    })
                    self._start_segment(
                        measured.pose,
                        Pose(self.plan.grasp.p, self.plan.grasp.R, self.plan.grasp.jaw),
                        max_steps=self.cfg.approach_max_steps,
                        success_trans_cm=self.cfg.approach_success_trans_cm,
                        success_rot_deg=self.cfg.approach_success_rot_deg,
                    )
                    self._approach_loop = self._segment_loop
                    self.approach_controller = self.hold_controller
                else:
                    self._finish(
                        PHASE_ABORTED,
                        f"approach {result.reason}. Holding the last command with "
                        "the jaw untouched; nothing else will move until a human "
                        "decides what to do.",
                    )

        # ------------------------------------------------------------------
        elif self.phase == PHASE_DESCEND:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw.approach_open_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg

            if result.reason == "success":
                self.grasp_pose_measured = measured.pose
                events.append({
                    "i": self.index, "event": "standoff_closed",
                    "trans_err_cm": trans_err, "rot_err_deg": rot_err,
                })
                self._start_segment(
                    measured.pose,
                    Pose(self.plan.grasp.p, self.plan.grasp.R, self.plan.grasp.jaw),
                    max_steps=10_000,
                    success_trans_cm=self.cfg.approach_success_trans_cm,
                    success_rot_deg=self.cfg.approach_success_rot_deg,
                )
                self._enter(PHASE_SETTLE, "jaws are at the needle")
            elif result.done:
                self._finish(
                    PHASE_ABORTED,
                    f"descend {result.reason}: the last "
                    f"{self.plan.grasp_standoff_m * 1000:.1f} mm onto the needle "
                    "did not finish, so the jaw was never closed",
                )

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
                if self.plan.suture is None:
                    self._enter(PHASE_HOLD, "lift reached")
                else:
                    self._begin_transport(measured.pose, events)
            elif result.done:
                self._finish(PHASE_ABORTED, f"lift {result.reason}")

        # ------------------------------------------------------------------
        elif self.phase == PHASE_TRANSPORT:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw_command_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg
            self._shadow_step(measured, events)

            slip = self._check_slip(evidence, events)
            if slip in ("abort", "lower"):
                # Nothing is lowered while loaded and in transit: the needle is
                # somewhere between the pad and the entry point, and driving
                # down through an unknown gap is worse than stopping.
                self._finish(
                    PHASE_ABORTED,
                    "jaw evidence disappeared during the transport; holding "
                    "position rather than descending",
                )
            elif result.reason == "success":
                events.append({"i": self.index, "event": "transport_reached",
                               "trans_err_cm": trans_err, "rot_err_deg": rot_err})
                self._start_segment(
                    measured.pose, self.suture_target,
                    max_steps=self.cfg.place_max_steps,
                    success_trans_cm=self.cfg.place_success_trans_cm,
                    success_rot_deg=self.cfg.place_success_rot_deg,
                )
                self._enter(PHASE_PLACE, "above the suturing point, descending")
            elif result.done:
                self._finish(PHASE_ABORTED, f"transport {result.reason}")

        # ------------------------------------------------------------------
        elif self.phase == PHASE_PLACE:
            result = self._hold_command(measured)
            action, clamps = result.action, result.clamps
            command = Command(result.command, self.jaw_command_rad)
            trans_err, rot_err = result.trans_err_cm, result.rot_err_deg
            self._shadow_step(measured, events)

            slip = self._check_slip(evidence, events)
            if slip in ("abort", "lower"):
                self._finish(
                    PHASE_ABORTED,
                    "jaw evidence disappeared during the descent onto the "
                    "suturing point",
                )
            elif result.reason == "success":
                events.append({
                    "i": self.index, "event": "suture_pose_reached",
                    "trans_err_cm": trans_err, "rot_err_deg": rot_err,
                    "note": "the tool is at the commanded pose. Where the NEEDLE "
                            "is depends on how it sits in the jaws, which this "
                            "deployment never measured.",
                })
                self._enter(PHASE_HOLD, "suturing pose reached")
            elif result.done:
                self._finish(PHASE_ABORTED, f"place {result.reason}")

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

        done = self.phase in TERMINAL_PHASES
        if done and entry_phase not in TERMINAL_PHASES:
            # The cycle that ends the run must not issue a fresh command. On an
            # abort the arm holds whatever it was last told, which is the pose
            # it is already tracking; on success there is nothing left to say.
            held = self._last_command or Command(measured.pose, self.jaw_command_rad)
            command = Command(held.pose, held.jaw_rad, False, False)
        else:
            self._last_command = command

        self._prev_measured = measured.pose

        merged = events + [
            e for e in self.events if e.get("i") == self.index and e not in events
        ]
        # Phase-local events are worth keeping on the sequencer too: summary()
        # is what an operator reads after the fact, and "the policy failed and
        # the servo finished the approach" is not a detail.
        for entry in events:
            if entry not in self.events:
                self.events.append(entry)

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
            events=merged,
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
            "suture_compensation": next(
                (e for e in self.events
                 if e.get("event") == "suture_compensation"), None
            ),
            "reached_suture_pose": bool(
                self.plan.suture is not None
                and any(e.get("event") == "suture_pose_reached" for e in self.events)
            ),
            "shadow": self.shadow_summary(),
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
