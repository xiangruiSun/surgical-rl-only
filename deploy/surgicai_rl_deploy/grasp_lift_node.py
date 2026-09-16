"""ROS 2 node: approach, close the jaw, observe, lift -- on a real dVRK PSM.

Defaults are deliberately timid:

* **dry run** -- nothing is published until ``--execute``;
* the precheck must pass before the first cycle (``feasibility.precheck``);
* the grasp gate defaults to ``manual``: the arm closes the jaw, stops, prints
  what it saw and waits for a human;
* the lift sign must be stated on the command line for a live run;
* on abort the last command is held and **the jaw is never auto-opened**.

Topics (all under ``--arm``, default ``/PSM1``)::

    subscribe  <arm>/measured_cp          geometry_msgs/PoseStamped
               <arm>/jaw/measured_js      sensor_msgs/JointState   (position, effort)
               <arm>/goal_reached         std_msgs/Bool            (move_cp only)
               <confirm topic>            std_msgs/Bool            (manual gate)
    publish    <arm>/servo_cp | move_cp   geometry_msgs/PoseStamped
               <arm>/jaw/servo_jp | move_jp
                                          sensor_msgs/JointState

Jaw units are radians, and **negative means squeeze** -- ``dvrk.psm``'s own
``jaw.close()`` commands -20 deg.  See :mod:`.jaw`.
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
except ImportError as exc:  # pragma: no cover - only on a host without ROS
    raise SystemExit(
        "ROS 2 is not on the path. Run: source /opt/ros/humble/setup.bash"
    ) from exc

from .controllers import D2Controller, RLController, ResidualController
from .feasibility import precheck
from .frames import Pose
from .jaw import JawBaseline, JawCalibration, JawCalibrationError
from .loop import SafetyLimits
from .contract import CONTRACTS
from .plan import LiftSpec, TransportSpec, build_plan

_CONTRACT_NAMES = tuple(CONTRACTS)
from .sequence import (
    ArmState,
    GraspLiftSequencer,
    SequenceConfig,
    PHASE_DONE,
    PHASE_HOLD,
    PHASE_OBSERVE,
    PHASE_WAIT_OPERATOR,
)


class GraspLiftNode(Node):
    def __init__(self, args, plan, sequencer, jaw_cal: JawCalibration):
        super().__init__("surgicai_grasp_lift")
        self.args = args
        self.arm = args.arm.rstrip("/")
        self.plan = plan
        self.sequencer = sequencer
        self.jaw_cal = jaw_cal

        # Subscriptions default to BEST_EFFORT because a RELIABLE subscriber
        # will not match a BEST_EFFORT publisher, while a BEST_EFFORT
        # subscriber matches either.  dVRK state topics are not uniform across
        # arms and versions -- on lcsr-dvrk-15 measured_cp matched a RELIABLE
        # subscriber and jaw/measured_js did not, which silently left the jaw
        # unmonitored.  Commands stay RELIABLE: a dropped setpoint matters.
        sub_qos = QoSProfile(depth=10)
        sub_qos.reliability = (
            ReliabilityPolicy.RELIABLE
            if args.sub_reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        pub_qos = QoSProfile(depth=10)
        pub_qos.reliability = ReliabilityPolicy.RELIABLE
        qos = sub_qos

        self._measured = None
        self._measured_stamp = 0.0
        self._measured_frame = None
        self._jaw_rad: Optional[float] = None
        self._jaw_effort: Optional[float] = None
        self._jaw_stamp = 0.0
        self._confirm: Optional[bool] = None
        self._confirm_prompted = False
        self._confirm_prompted_at = 0.0
        self._confirm_first_at = time.monotonic()
        self._last_logged_phase = None
        self._last_logged_at = 0.0

        self.create_subscription(
            PoseStamped, f"{self.arm}/measured_cp", self._on_measured_cp, qos
        )
        self.create_subscription(
            JointState, f"{self.arm}/jaw/measured_js", self._on_jaw, qos
        )
        if args.confirm_topic:
            self.create_subscription(Bool, args.confirm_topic, self._on_confirm, qos)

        self.cmd_pub = self.create_publisher(
            PoseStamped, f"{self.arm}/{args.interface}", pub_qos
        )
        self.jaw_pub = self.create_publisher(
            JointState, f"{self.arm}/jaw/{args.jaw_interface}", pub_qos
        )

        # Dry-run walkthrough: with nothing published the arm cannot move, so
        # the loop can never converge and every dry run ends in
        # "approach max_steps", which tells the operator nothing.  Feeding the
        # command back as the next measured pose walks the whole sequence
        # instead.  Clearly labelled: these poses are simulated, not measured.
        self.simulate = bool(getattr(args, "dry_run_simulate", False)) and not args.execute
        self._sim_pose = None
        self._sim_jaw_rad = None
        self._warned_stale_cp = False
        self._warned_stale_jaw = False

        self.trace_file = open(args.trace, "w") if args.trace else None
        self.finished = False
        self._started = False

    # -- callbacks ---------------------------------------------------------
    def _on_measured_cp(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        self._measured = (
            np.array([p.x, p.y, p.z]),
            np.array([q.x, q.y, q.z, q.w]),
        )
        self._measured_stamp = time.monotonic()
        self._measured_frame = msg.header.frame_id

    def _on_jaw(self, msg: JointState):
        if msg.position:
            self._jaw_rad = float(msg.position[0])
        self._jaw_effort = float(msg.effort[0]) if msg.effort else None
        self._jaw_stamp = time.monotonic()

    def _on_confirm(self, msg: Bool):
        self._confirm = bool(msg.data)

    # -- helpers -----------------------------------------------------------
    def _state(self) -> Optional[ArmState]:
        if self._measured is None:
            return None
        if self.simulate and self._sim_pose is not None:
            # A perfect arm that lands exactly on the last command.  Optimistic
            # by construction, which is the point: it shows the plan, not the
            # tracking.
            jaw_rad = self._sim_jaw_rad
            return ArmState(
                pose=Pose(
                    self._sim_pose.p,
                    self._sim_pose.R,
                    0.0 if jaw_rad is None else self.jaw_cal.normalise(jaw_rad),
                ),
                jaw_rad=jaw_rad,
                jaw_effort=None,
            )
        pos, quat = self._measured
        jaw_norm = (
            0.0 if self._jaw_rad is None else self.jaw_cal.normalise(self._jaw_rad)
        )
        return ArmState(
            pose=Pose.from_pos_quat(pos, quat, jaw_norm),
            jaw_rad=self._jaw_rad,
            jaw_effort=self._jaw_effort,
        )

    def _confirm_callback(self) -> Optional[bool]:
        """Operator says go / no.  Topic first, then the keyboard.

        The prompt is re-printed periodically.  It used to be printed once and
        then buried under a 10 Hz status line, so within seconds the only thing
        on screen was a stationary arm repeating itself and no hint that it was
        waiting for a person.
        """
        now = time.monotonic()
        due = (
            not self._confirm_prompted
            or now - self._confirm_prompted_at >= self.args.confirm_reprompt_s
        )
        if not self._confirm_prompted:
            self._confirm_first_at = now
        if due:
            self._confirm_prompted = True
            self._confirm_prompted_at = now
            summary = self.sequencer.window.summary()
            self.get_logger().warn("=" * 68)
            self.get_logger().warn("JAW IS CLOSED. The lift is waiting for you.")
            self.get_logger().warn(
                f"  jaw evidence: {summary['blocked_samples']}/{summary['samples']} "
                f"samples show the jaw stopping early"
            )
            if summary["last"]:
                self.get_logger().warn(f"  last reading: {summary['last']['note']}")
            self.get_logger().warn(
                "  THIS IS NOT A GRASP CONFIRMATION. Look at the scene."
            )
            self.get_logger().warn(
                f"  lift: {self.plan.lift_spec.describe()}"
            )
            if self.args.confirm_topic:
                self.get_logger().warn(
                    f"  publish true on {self.args.confirm_topic} to lift, "
                    "false to stop"
                )
            self.get_logger().warn("  or type 'y' + Enter to lift, 'n' + Enter to stop")
            self.get_logger().warn("=" * 68)

        if self._confirm is not None:
            answer, self._confirm = self._confirm, None
            return answer

        # A dry run publishes nothing, so blocking forever on a human defeats
        # the point of walking the sequence.  Answer it, loudly, and say that a
        # live run would not.
        if not self.args.execute and self.args.dry_run_confirm_after_s > 0.0:
            waited = now - self._confirm_first_at
            if waited >= self.args.dry_run_confirm_after_s:
                self.get_logger().warn(
                    f"DRY RUN: nobody answered in {waited:.0f} s, so the gate is "
                    "being released automatically to show the rest of the "
                    "sequence. A live run waits indefinitely."
                )
                return True

        if sys.stdin is not None and sys.stdin.isatty():
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                line = sys.stdin.readline().strip().lower()
                if line in ("y", "yes"):
                    return True
                if line in ("n", "no"):
                    return False
        return None

    def _publish(self, command):
        stamp = self.get_clock().now().to_msg()
        if command.publish_pose:
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self._measured_frame or self.args.expect_frame
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
                command.pose.p
            )
            q = command.pose.quat_xyzw()
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ) = q
            self.cmd_pub.publish(msg)
        if command.publish_jaw and not self.args.freeze_jaw:
            js = JointState()
            js.header.stamp = stamp
            js.position = [float(command.jaw_rad)]
            self.jaw_pub.publish(js)

    def command_subscribers(self) -> dict:
        """Who, if anyone, is listening to the topics we command on."""
        return {
            f"{self.arm}/{self.args.interface}": self.cmd_pub.get_subscription_count(),
            f"{self.arm}/jaw/{self.args.jaw_interface}":
                self.jaw_pub.get_subscription_count(),
        }

    def _log(self, record: dict):
        if self.trace_file:
            self.trace_file.write(json.dumps(record, default=str) + "\n")
            self.trace_file.flush()

    # -- episode -----------------------------------------------------------
    def start_episode(self, precheck_report):
        state = self._state()
        if state is None:
            raise RuntimeError(f"no message on {self.arm}/measured_cp")
        if self.args.expect_frame and self._measured_frame != self.args.expect_frame:
            self.get_logger().warn(
                f"measured_cp frame_id is {self._measured_frame!r}, expected "
                f"{self.args.expect_frame!r}. The grasp position you passed must "
                "be in the same frame as measured_cp."
            )
        if self._jaw_rad is None:
            self.get_logger().warn(
                f"nothing on {self.arm}/jaw/measured_js: the jaw will be commanded "
                "blind and no evidence can be collected"
            )
        elif self._jaw_effort is None:
            self.get_logger().warn(
                "jaw/measured_js carries no effort field on this arm; evidence "
                "falls back to the jaw angle alone"
            )

        report = self.sequencer.begin(state)
        self.get_logger().info(f"frame           : {self._measured_frame}")
        self.get_logger().info(f"start        cm : {np.round(state.pose.p * 100, 3)}")
        if (hover := self.plan.hover) is not None:
            self.get_logger().info(
                f"hover        cm : {np.round(hover.p * 100, 3)}  "
                f"({self.plan.grasp_standoff_m * 1000:.1f} mm short of the grasp "
                "pose, along the tool axis -- this is where the POLICY aims)"
            )
        self.get_logger().info(f"grasp        cm : {np.round(self.plan.grasp.p * 100, 3)}")
        self.get_logger().info(f"lift target  cm : {np.round(self.plan.lifted.p * 100, 3)}")
        self.get_logger().info(f"lift            : {self.plan.lift_spec.describe()}")
        if self.plan.staged is not None:
            self.get_logger().info(
                f"staged       cm : {np.round(self.plan.staged.p * 100, 3)}  "
                "(servo target before the policy takes over)"
            )
        if self.plan.suture is not None:
            if (via := self.plan.via) is not None:
                self.get_logger().info(
                    f"via          cm : {np.round(via.p * 100, 3)}"
                )
            self.get_logger().info(
                f"suture       cm : {np.round(self.plan.suture.p * 100, 3)}  "
                "(TOOL pose; where the needle ends up also depends on how it "
                "sits in the jaws, which nothing here measures)"
            )
        self.get_logger().info(f"{self.jaw_cal.describe()}")
        self.get_logger().info(f"grasp gate      : {self.sequencer.cfg.grasp_gate}")
        # The R6 support bounds the learned policy and nothing else.  Logging
        # it as a warning under the geometric servo contradicts the precheck,
        # which has already said the trained region does not apply -- and a
        # warning mid-run on a live arm should mean "consider stopping", not
        # "here is a number that does not bind you".
        controller_name = getattr(self.sequencer.approach_controller, "name", "")
        policy_driven = controller_name in ("rl", "residual")
        if report.get("staging") is not None:
            staging = report["staging"]
            self.get_logger().info(
                f"staging      : {staging['contract']}, offset "
                f"{np.round(staging['offset_tool_cm'], 2)} cm, rotation "
                f"{staging['rotation_deg']:.1f} deg, in support="
                f"{staging['in_support']}"
            )
        if report.get("out_of_distribution"):
            if policy_driven:
                self.get_logger().warn(
                    "approach is OUTSIDE the checkpoint's demonstration "
                    "support, where "
                    "the policy has been measured to orbit the goal rather "
                    "than reach it:"
                )
                for line in report["out_of_distribution"]:
                    self.get_logger().warn(f"  - {line}")
            else:
                self.get_logger().info(
                    f"the checkpoint's demonstration support does not bind "
                    f"controller '{controller_name}' (geometric). For reference "
                    "only, the RL path would flag:"
                )
                for line in report["out_of_distribution"]:
                    self.get_logger().info(f"  - {line}")
        if not self.args.execute:
            if self.simulate:
                self.get_logger().info(
                    "DRY RUN: nothing is published. Poses after the first cycle "
                    "are SIMULATED (a perfect arm landing on each command), so "
                    "you see the whole sequence. They are not measurements."
                )
            else:
                self.get_logger().info(
                    "DRY RUN (static): nothing is published, so the arm will not "
                    "move and the approach cannot converge. Expect the episode "
                    "to end at max_steps; that is not a controller failure."
                )

        self._log(
            {
                "event": "begin",
                "execute": bool(self.args.execute),
                "frame": self._measured_frame,
                "plan": self.plan.as_dict(),
                "precheck": precheck_report.as_dict(),
                "approach_report": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in report.items()
                },
            }
        )
        self._started = True

    def tick(self):
        if self.finished or not self._started:
            return

        # The freshness guards exist to stop motion against a dead feed.  In a
        # dry run nothing moves, so a silent topic is worth saying once, not
        # worth killing the walkthrough over.
        age = time.monotonic() - self._measured_stamp
        if age > self.sequencer.limits.max_pose_age_s:
            if self.args.execute:
                self.get_logger().error(f"measured_cp stale ({age:.2f} s) - stopping")
                self.finish("abort: stale measured_cp")
                return
            if not self._warned_stale_cp:
                self._warned_stale_cp = True
                self.get_logger().warn(
                    f"measured_cp stale ({age:.2f} s). In a live run this aborts."
                )
        jaw_age = time.monotonic() - self._jaw_stamp
        if self._jaw_rad is not None and jaw_age > self.args.max_jaw_age_s:
            if self.args.execute:
                self.get_logger().error("jaw/measured_js stale - stopping")
                self.finish("abort: stale jaw/measured_js")
                return
            if not self._warned_stale_jaw:
                self._warned_stale_jaw = True
                self.get_logger().warn(
                    f"jaw/measured_js stale ({jaw_age:.2f} s). In a live run "
                    "this aborts."
                )

        state = self._state()
        step = self.sequencer.step(state)

        jaw = step.jaw_evidence
        jaw_txt = ""
        if jaw is not None and jaw.measured_rad is not None:
            jaw_txt = (
                f"  jaw cmd {np.degrees(jaw.commanded_rad):+6.1f} "
                f"meas {np.degrees(jaw.measured_rad):+6.1f} "
                f"blocked={jaw.jaw_blocked}"
            )
        # Waiting for a human is not news every 100 ms.  Anything that moves,
        # changes phase, clamps or raises an event still prints immediately.
        line = (
            f"{step.index:4d} {step.phase:<14s} err {step.trans_err_cm:6.2f} cm / "
            f"{step.rot_err_deg:6.2f} deg{jaw_txt}"
            + (
                f"  clamped[{','.join(c['kind'] for c in step.clamps)}]"
                if step.clamps
                else ""
            )
        )
        quiet = (
            step.phase == self._last_logged_phase
            and not step.events
            and not step.clamps
            and step.phase in (PHASE_WAIT_OPERATOR, PHASE_HOLD, PHASE_OBSERVE)
        )
        if not quiet or (time.monotonic() - self._last_logged_at) >= 2.0:
            self.get_logger().info(line)
            self._last_logged_at = time.monotonic()
        self._last_logged_phase = step.phase
        for event in step.events:
            self.get_logger().warn(f"  * {json.dumps(event, default=str)}")
        self._log({"event": "step", **step.as_dict()})

        if self.args.execute and not step.reason.startswith("abort"):
            self._publish(step.command)

        if self.simulate:
            self._sim_pose = step.command.pose
            self._sim_jaw_rad = step.command.jaw_rad

        if step.done:
            self.finish(step.reason)

    def finish(self, reason: str):
        self.finished = True
        summary = self.sequencer.summary()
        log = self.get_logger().info if reason == "success" else self.get_logger().warn
        log(f"episode finished: {reason}")
        if not self.args.execute:
            log(
                "this was a DRY RUN: no command reached the arm"
                + (
                    ". The poses above after the first cycle were simulated."
                    if self.simulate
                    else ", so the arm could not move and this outcome says "
                    "nothing about the controller."
                )
            )
        log(
            "grasp verified: NO. This arm has no grasp sensor; the jaw readings "
            "in the trace are observations, not proof the needle was picked up."
        )
        if self.sequencer.phase not in (PHASE_DONE,):
            log(
                "the last command is held and the jaw was NOT opened: if the "
                "needle is held above the tissue, opening it would drop it. "
                "Take manual control."
            )
        self._log({"event": "finish", "reason": reason, "summary": summary})
        if self.trace_file:
            self.trace_file.close()
            self.trace_file = None


# ----------------------------------------------------------------------------
def resolve_jaw_baseline(spec, *, grip_deg: float, arm: str = "/PSM1"):
    """Load and sanity-check an empty-jaw baseline.

    Returns ``(baseline, problem)``.  ``problem`` is a ready-to-print message
    when the file cannot be trusted, in which case the run must not start: a
    baseline recorded against a different grip angle, or in a dry run where the
    jaw never actually moved, silently compares every later reading against the
    wrong reference, which is worse than having no baseline at all.
    """
    if not spec:
        return None, None

    path = Path(spec).expanduser()
    if not path.is_file():
        return None, (
            f"jaw baseline not found: {path}\n"
            "Create it with an EMPTY gripper (the arm does not move):\n"
            f"    python3 tools/calibrate_jaw.py --arm {arm} "
            f"--jaw-grip-deg {grip_deg} --out {path} --execute\n"
            "Or drop --jaw-baseline to run without one: jaw readings are then "
            "logged raw, with no reference for what an empty close looks like."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"jaw baseline {path} is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"jaw baseline {path} should contain a JSON object"

    try:
        baseline = JawBaseline.from_dict(payload)
    except JawCalibrationError as exc:
        return None, f"jaw baseline {path} is unusable: {exc}"

    if not payload.get("executed", True):
        return None, (
            f"jaw baseline {path} was recorded in a DRY RUN, so the jaw never "
            "moved and the numbers describe wherever it already was. Re-run "
            "tools/calibrate_jaw.py with --execute."
        )

    recorded = payload.get("grip_command_deg")
    if recorded is not None and abs(float(recorded) - float(grip_deg)) > 0.5:
        return None, (
            f"jaw baseline {path} was recorded closing to {float(recorded):.1f} "
            f"deg but this run commands {float(grip_deg):.1f} deg. Every "
            "residual would be measured against the wrong reference. Re-record "
            "it, or match --jaw-grip-deg to the baseline."
        )
    return baseline, None



def _step_size_from(args):
    import numpy as _np
    from surgicai_rl_deploy.contract import STEP_SIZE_RAW as _S
    step = _np.asarray(_S, dtype=_np.float64).copy()
    if getattr(args, "trans_step_mm", None) is not None:
        step[:3] = args.trans_step_mm / 1000.0
    if getattr(args, "angle_step_deg", None) is not None:
        step[3:6] = _np.deg2rad(args.angle_step_deg)
    if getattr(args, "jaw_step", None) is not None:
        step[6] = args.jaw_step
    return step

def _load_policy(path, device, verify, named_contract=None):
    """Return ``(policy, contract)``, resolving the contract from the digest."""
    from .contract import contract_for_digest
    from .policy import ApproachPolicy

    policy = ApproachPolicy.load(path, device=device, verify=verify)
    if named_contract:
        return policy, CONTRACTS[named_contract]
    contract = contract_for_digest(policy.sha256)
    if contract is None:
        print(
            f"WARNING: {Path(path).name} has no registered contract "
            f"(sha256 {policy.sha256[:12]}). Its action scale, episode budget "
            "and tolerances are therefore guesses, and a policy driven at the "
            "wrong action scale tracks briefly and then diverges. Settle it "
            "with tools/replay_demos.py --compare before --execute.",
            file=sys.stderr,
        )
    return policy, contract


def _shutdown():
    """_shutdown() twice raises and buries the real error."""
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def build_limits(args) -> SafetyLimits:
    """The safety envelope for a run, including the dry-run exception.

    In a *static* dry run nothing is published, so the arm cannot follow and
    the tracking guard would be reporting the dry run itself as a fault. It is
    the one case where the guard is meaningless rather than merely quiet.
    """
    tracking = args.max_tracking_error_cm
    if not args.execute and not getattr(args, "dry_run_simulate", True):
        tracking = float("inf")
    return SafetyLimits(
        workspace_pad_cm=args.workspace_pad_cm,
        max_step_translation_mm=args.max_step_translation_mm,
        max_step_rotation_deg=args.max_step_rotation_deg,
        max_tracking_error_cm=tracking,
    )


def build_controller(args):
    """Return ``(controller, contract)`` for the approach leg."""
    if args.controller == "d2":
        # A named contract still applies: staging into a demonstrated support
        # is geometry, and comparing the servo against the same support is the
        # whole point of having both controllers.
        named = getattr(args, "contract", None)
        return D2Controller(staged=True), (CONTRACTS[named] if named else None)
    if not args.model:
        raise SystemExit("--model is required for the rl/residual controllers")
    policy, contract = _load_policy(
        args.model, args.device, not args.allow_unknown_model,
        getattr(args, "contract", None),
    )
    if args.controller == "rl":
        return RLController(policy), contract
    return ResidualController(
        policy, policy_weight=args.policy_weight, servo_weight=args.servo_weight
    ), contract


def build_shadow(args):
    """Return ``(controller, contract)`` for the logged-only transport policy."""
    if not getattr(args, "shadow_model", None):
        return None, None
    policy, contract = _load_policy(
        args.shadow_model, args.device, not args.allow_unknown_model,
        getattr(args, "shadow_contract", None),
    )
    return RLController(policy), contract


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="/PSM1")
    ap.add_argument("--contract", choices=sorted(_CONTRACT_NAMES),
                    help="force a checkpoint contract instead of resolving it "
                         "from the model's SHA256")
    ap.add_argument("--grasp-standoff-mm", type=float, default=None,
                    help="how far short of the grasp pose the approach policy "
                         "aims, along the tool axis. Default: the checkpoint "
                         "contract's value (7 mm for the Approach policies, "
                         "matching needle_goal_evaluator's lift_height). The "
                         "remaining distance is a separate slow descent. 0 "
                         "disables it and aims straight at the needle.")
    ap.add_argument("--descend-step-mm", type=float, default=0.5)
    ap.add_argument("--descend-max-steps", type=int, default=200)
    ap.add_argument("--suture-pos", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="TOOL position at the suturing point, metres, in the "
                         "frame measured_cp reports. This is where the gripper "
                         "goes, not where the needle goes.")
    ap.add_argument("--suture-quat", nargs=4, type=float, default=None,
                    metavar=("QX", "QY", "QZ", "QW"),
                    help="tool orientation at the suturing point, i.e. the "
                         "needle angle expressed as a gripper pose")
    ap.add_argument("--suture-confirmed", action="store_true",
                    help="required for --execute: a human has checked this pose "
                         "against the scene")
    ap.add_argument("--confirm-reprompt-s", type=float, default=10.0,
                    help="re-print the operator prompt this often while "
                         "waiting, so it cannot scroll away")
    ap.add_argument("--dry-run-confirm-after-s", type=float, default=15.0,
                    help="in a DRY RUN only, release the operator gate after "
                         "this long so the rest of the sequence can be walked. "
                         "0 waits forever. A live run always waits.")
    ap.add_argument("--taught-grasp-pos", nargs=3, type=float, default=None,
                    help="the grasp pose the SUTURING pose was taught with, if "
                         "different from --grasp-pos. The suturing pose encodes "
                         "how the needle sat in the jaws at that moment.")
    ap.add_argument("--taught-grasp-quat", nargs=4, type=float, default=None)
    ap.add_argument("--compensate-suture", choices=["apply", "report", "off"],
                    default="apply",
                    help="correct the suturing pose for where the jaws actually "
                         "closed. Exact, and needs neither the needle pose nor "
                         "the entry pose; assumes the needle did not move.")
    ap.add_argument("--max-suture-compensation-mm", type=float, default=10.0)
    ap.add_argument("--transport-via", choices=["lift_height", "direct"],
                    default="lift_height")
    ap.add_argument("--transport-clearance-cm", type=float, default=None)
    ap.add_argument("--transport-max-steps", type=int, default=600)
    ap.add_argument("--transport-success-trans-cm", type=float, default=0.3)
    ap.add_argument("--transport-success-rot-deg", type=float, default=5.0)
    ap.add_argument("--place-max-steps", type=int, default=400)
    ap.add_argument("--place-success-trans-cm", type=float, default=0.2)
    ap.add_argument("--place-success-rot-deg", type=float, default=3.0)
    ap.add_argument("--on-approach-failure", choices=["hold", "servo"],
                    default="hold",
                    help="hold stops at the last command with the jaw untouched "
                         "and waits for a human; servo finishes the approach "
                         "geometrically and carries on")
    ap.add_argument("--stage", action="store_true",
                    help="servo to a pose inside the checkpoint's demonstrated "
                         "support before handing over to the policy")
    ap.add_argument("--stage-offset-tool-cm", nargs=3, type=float, default=None)
    ap.add_argument("--stage-rotation-deg", type=float, default=None)
    ap.add_argument("--stage-max-steps", type=int, default=400)
    ap.add_argument("--shadow-model",
                    help="a second checkpoint run alongside the transport and "
                         "place legs, logged and never published")
    ap.add_argument("--shadow-contract", choices=sorted(_CONTRACT_NAMES))
    ap.add_argument(
        "--grasp-pos", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
        help="where to close the jaw, in metres, in the SAME frame as measured_cp",
    )
    ap.add_argument("--grasp-quat", nargs=4, type=float, default=None,
                    metavar=("QX", "QY", "QZ", "QW"))
    ap.add_argument("--goal-orientation",
                    choices=["hold", "trained_relative", "explicit"], default=None)

    # lift
    ap.add_argument("--lift-axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--lift-sign", type=int, choices=[-1, 1], default=None,
                    help="REQUIRED for --execute. Which way along the axis moves "
                         "the gripper AWAY from the tissue. Check it in the scene.")
    ap.add_argument("--lift-distance-cm", type=float, default=1.5)
    ap.add_argument("--lift-frame", choices=["robot", "tool"], default="robot")

    # jaw
    ap.add_argument("--jaw-open-deg", type=float, default=60.0)
    ap.add_argument("--jaw-closed-deg", type=float, default=0.0)
    ap.add_argument("--jaw-grip-deg", type=float, default=-15.0,
                    help="negative squeezes; dvrk's own close() uses -20")
    ap.add_argument("--jaw-approach-open-deg", type=float, default=40.0)
    ap.add_argument("--jaw-baseline", help="JSON from tools/calibrate_jaw.py")
    ap.add_argument("--jaw-interface", choices=["servo_jp", "move_jp"],
                    default="servo_jp")
    ap.add_argument("--freeze-jaw", action="store_true",
                    help="run the whole sequence but never publish a jaw command")
    ap.add_argument("--max-jaw-age-s", type=float, default=0.5)

    # gate
    # settle
    ap.add_argument("--settle-steps", type=int, default=5,
                    help="half-window, in cycles, of the drift test run before "
                         "the jaw is allowed to close")
    ap.add_argument("--settle-translation-tol-mm", type=float, default=0.5,
                    help="permitted drift between the two halves of that window; "
                         "raise it if a noisy measured_cp causes settle timeouts")
    ap.add_argument("--settle-rotation-tol-deg", type=float, default=0.5)
    ap.add_argument("--settle-timeout-steps", type=int, default=60)
    ap.add_argument("--observe-steps", type=int, default=10)
    ap.add_argument("--hold-steps", type=int, default=10)
    ap.add_argument("--residual-margin-deg", type=float, default=1.0,
                    help="how far past the empty-jaw stop the jaw must remain "
                         "before a reading counts as blocked")

    ap.add_argument("--grasp-gate",
                    choices=["manual", "evidence", "always", "never"], default="manual")
    ap.add_argument("--confirm-topic", default=None,
                    help="std_msgs/Bool topic that releases the manual gate")
    ap.add_argument("--operator-timeout-steps", type=int, default=0)
    ap.add_argument("--on-slip", choices=["abort", "continue", "lower"], default="abort")

    # controller
    ap.add_argument("--controller", choices=["rl", "d2", "residual"], default="d2")
    ap.add_argument("--model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-unknown-model", action="store_true")
    ap.add_argument("--policy-weight", type=float, default=0.50)
    ap.add_argument("--servo-weight", type=float, default=0.75)
    ap.add_argument("--frame-mode", choices=["rebase", "translate", "identity"],
                    default="rebase")
    ap.add_argument("--trans-step-mm", type=float, default=None,
                    help="action scale for translation. The package default of "
                         "1.5 mm is R6's; the upstream SurgicAI Approach "
                         "checkpoint uses 0.5 mm. Recover it from any "
                         "checkpoint with tools/recover_step_size.py -- applying "
                         "a policy's actions at the wrong scale makes it track "
                         "for a few steps and then diverge.")
    ap.add_argument("--angle-step-deg", type=float, default=None,
                    help="action scale for rotation; R6 uses 3 deg, upstream 2")
    ap.add_argument("--jaw-step", type=float, default=None)
    ap.add_argument("--goal-rpy-train", nargs=3, type=float, default=None,
                    metavar=("ROLL", "PITCH", "YAW"),
                    help="the goal's RPY on the TRAINING 2*pi branch. Defaults "
                         "to the trained goal's own values in rebase mode.")
    ap.add_argument("--no-unwrap-rpy", action="store_true",
                    help="stop unwrapping observations onto the training RPY "
                         "branch. Only for reproducing the old, broken behaviour.")


    # limits
    ap.add_argument("--interface", choices=["servo_cp", "move_cp"], default="servo_cp")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--approach-max-steps", type=int, default=200)
    ap.add_argument("--lift-max-steps", type=int, default=120)
    ap.add_argument("--success-trans-cm", type=float, default=1.0)
    ap.add_argument("--success-rot-deg", type=float, default=10.0)
    ap.add_argument("--lift-success-trans-cm", type=float, default=0.2)
    ap.add_argument("--workspace-pad-cm", type=float, default=2.0)
    ap.add_argument("--max-step-translation-mm", type=float, default=2.5)
    ap.add_argument("--max-step-rotation-deg", type=float, default=5.0)
    ap.add_argument("--max-tracking-error-cm", type=float, default=1.5)
    ap.add_argument("--max-path-radius-cm", type=float, default=8.0)
    ap.add_argument("--limit-low", nargs=3, type=float, default=None,
                    help="hard positional box, metres, same frame as measured_cp")
    ap.add_argument("--limit-high", nargs=3, type=float, default=None)

    ap.add_argument("--expect-frame", default="ECM")
    ap.add_argument("--sub-reliability", choices=["best_effort", "reliable"],
                    default="best_effort",
                    help="QoS reliability requested for measured_cp and "
                         "jaw/measured_js. best_effort matches publishers of "
                         "either kind; reliable silently fails to match a "
                         "best_effort publisher, which is how a jaw topic can "
                         "echo fine on the command line and never reach a node.")
    ap.add_argument("--dry-run-static", action="store_true",
                    help="in a dry run, keep reading the real measured_cp. The "
                         "arm cannot move because nothing is published, so the "
                         "approach never converges and the episode always ends "
                         "at max_steps. Off by default: a dry run instead feeds "
                         "each command back as the next pose and walks the "
                         "whole sequence.")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="treat every precheck warning as a failure")
    ap.add_argument("--execute", action="store_true",
                    help="actually publish; without it this is a dry run")
    parsed = ap.parse_args(argv)
    parsed.dry_run_simulate = not parsed.dry_run_static
    return parsed


def main(argv=None) -> int:
    args = parse_args(argv)

    jaw_cal = JawCalibration(
        open_rad=float(np.deg2rad(args.jaw_open_deg)),
        closed_rad=float(np.deg2rad(args.jaw_closed_deg)),
        grip_rad=float(np.deg2rad(args.jaw_grip_deg)),
        approach_open_rad=float(np.deg2rad(args.jaw_approach_open_deg)),
    )
    baseline, problem = resolve_jaw_baseline(
        args.jaw_baseline, grip_deg=args.jaw_grip_deg, arm=args.arm
    )
    if problem:
        print(problem, file=sys.stderr)
        return 3

    rclpy.init()
    node_args = args

    rclpy.logging.get_logger("surgicai_grasp_lift").info("waiting for measured_cp...")

    # A throwaway node just to read the start pose, so the precheck can run on
    # real numbers before the real node publishes anything.
    probe = rclpy.create_node("surgicai_grasp_lift_probe")
    qos = QoSProfile(depth=10)
    qos.reliability = (
        ReliabilityPolicy.RELIABLE
        if args.sub_reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    holder = {}

    def _probe_cp(msg):
        holder["pose"] = (
            np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            ),
        )

    def _probe_jaw(msg):
        if msg.position:
            holder["jaw"] = float(msg.position[0])

    probe.create_subscription(
        PoseStamped, f"{args.arm.rstrip('/')}/measured_cp", _probe_cp, qos
    )
    probe.create_subscription(
        JointState, f"{args.arm.rstrip('/')}/jaw/measured_js", _probe_jaw, qos
    )
    # Wait for BOTH topics, not just the pose.  measured_cp usually arrives
    # first, and returning the moment it does used to leave the jaw looking
    # absent when it was merely a few milliseconds behind.
    deadline = time.monotonic() + 5.0
    while ("pose" not in holder or "jaw" not in holder) and time.monotonic() < deadline:
        rclpy.spin_once(probe, timeout_sec=0.1)
    probe.destroy_node()

    if "pose" not in holder:
        print(f"no messages on {args.arm}/measured_cp after 5 s", file=sys.stderr)
        _shutdown()
        return 2

    pos, quat = holder["pose"]
    start = Pose.from_pos_quat(
        pos, quat, jaw_cal.normalise(holder.get("jaw", jaw_cal.approach_open_rad))
    )

    lift = LiftSpec(
        axis=args.lift_axis,
        sign=int(args.lift_sign) if args.lift_sign is not None else 1,
        distance_m=args.lift_distance_cm / 100.0,
        frame=args.lift_frame,
        explicit=args.lift_sign is not None,
    )
    if args.suture_pos is not None and args.suture_quat is None:
        print("--suture-pos needs --suture-quat: the point of the leg is to "
              "present the needle at an angle, so the orientation is the "
              "payload.", file=sys.stderr)
        _shutdown()
        return 3

    try:
        controller, contract = build_controller(args)
        shadow_controller, shadow_contract = build_shadow(args)
    except (SystemExit, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        _shutdown()
        return 3

    if args.stage and contract is None:
        print("--stage needs a checkpoint contract to stage INTO. Pass --model "
              "with --controller rl, or name one with --contract.",
              file=sys.stderr)
        _shutdown()
        return 3

    standoff = (
        args.grasp_standoff_mm / 1000.0 if args.grasp_standoff_mm is not None
        else (contract.grasp_standoff_m if contract is not None else 0.0)
    )

    taught_grasp = None
    if args.taught_grasp_pos is not None:
        if args.taught_grasp_quat is None:
            print("--taught-grasp-pos needs --taught-grasp-quat", file=sys.stderr)
            _shutdown()
            return 3
        taught_grasp = Pose.from_pos_quat(
            args.taught_grasp_pos, args.taught_grasp_quat, 0.0
        )

    transport = None
    if args.suture_pos is not None:
        transport = TransportSpec(
            via=args.transport_via,
            via_clearance_m=(
                None if args.transport_clearance_cm is None
                else args.transport_clearance_cm / 100.0
            ),
        )

    plan = build_plan(
        start,
        args.grasp_pos,
        goal_orientation=args.goal_orientation
        or ("explicit" if args.grasp_quat else "hold"),
        goal_quat_xyzw=tuple(args.grasp_quat) if args.grasp_quat else None,
        lift=lift,
        jaw=jaw_cal,
        grasp_standoff_m=standoff,
        suture_position_m=args.suture_pos,
        suture_quat_xyzw=tuple(args.suture_quat) if args.suture_quat else None,
        transport=transport,
        taught_grasp_pose=taught_grasp,
        stage_contract=contract if args.stage else None,
        stage_offset_tool_cm=args.stage_offset_tool_cm,
        stage_rotation_deg=args.stage_rotation_deg,
    )

    report = precheck(
        plan,
        controller=args.controller,
        execute=args.execute,
        grasp_gate=args.grasp_gate,
        jaw_baseline=baseline,
        max_path_radius_cm=args.max_path_radius_cm,
        limit_low_m=args.limit_low,
        limit_high_m=args.limit_high,
        max_step_translation_mm=args.max_step_translation_mm,
        max_step_rotation_deg=args.max_step_rotation_deg,
        approach_max_steps=args.approach_max_steps,
        lift_max_steps=args.lift_max_steps,
        descend_max_steps=args.descend_max_steps,
        descend_step_mm=args.descend_step_mm,
        transport_max_steps=args.transport_max_steps,
        place_max_steps=args.place_max_steps,
        success_trans_cm=args.lift_success_trans_cm,
        success_rot_deg=args.success_rot_deg,
        suture_confirmed=args.suture_confirmed,
        stage_contract=contract,
        strict=args.strict,
    )
    print(report.render())
    print()
    if not report.ok:
        print("refusing to run. Nothing was published.", file=sys.stderr)
        _shutdown()
        return 3

    cfg = SequenceConfig(
        frame_mode=args.frame_mode,
        approach_contract=contract,
        on_approach_failure=args.on_approach_failure,
        descend_max_steps=args.descend_max_steps,
        descend_step_mm=args.descend_step_mm,
        stage_max_steps=args.stage_max_steps,
        transport_max_steps=args.transport_max_steps,
        transport_success_trans_cm=args.transport_success_trans_cm,
        transport_success_rot_deg=args.transport_success_rot_deg,
        place_max_steps=args.place_max_steps,
        place_success_trans_cm=args.place_success_trans_cm,
        place_success_rot_deg=args.place_success_rot_deg,
        approach_max_steps=args.approach_max_steps,
        approach_success_trans_cm=args.success_trans_cm,
        approach_success_rot_deg=args.success_rot_deg,
        lift_max_steps=args.lift_max_steps,
        lift_success_trans_cm=args.lift_success_trans_cm,
        settle_steps=args.settle_steps,
        settle_translation_tol_mm=args.settle_translation_tol_mm,
        settle_rotation_tol_deg=args.settle_rotation_tol_deg,
        settle_timeout_steps=args.settle_timeout_steps,
        observe_steps=args.observe_steps,
        hold_steps=args.hold_steps,
        residual_margin_deg=args.residual_margin_deg,
        grasp_gate=args.grasp_gate,
        operator_timeout_steps=args.operator_timeout_steps,
        on_slip=args.on_slip,
        step_size=_step_size_from(args),
        goal_rpy_train=tuple(args.goal_rpy_train) if args.goal_rpy_train else None,
        unwrap_rpy=not args.no_unwrap_rpy,
    )
    limits = build_limits(args)

    sequencer = GraspLiftSequencer(
        plan, controller, cfg, limits, baseline,
        shadow_controller=shadow_controller, shadow_contract=shadow_contract,
    )
    node = GraspLiftNode(node_args, plan, sequencer, jaw_cal)
    sequencer.confirm_callback = node._confirm_callback

    deadline = time.monotonic() + 5.0
    while (
        node._measured is None or node._jaw_rad is None
    ) and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node._measured is None:
        node.get_logger().error("lost measured_cp between the precheck and the run")
        node.destroy_node()
        _shutdown()
        return 2

    # Publishing into the void looks exactly like an arm that will not move,
    # and costs minutes to tell apart by eye. ROS knows the answer immediately.
    listeners = node.command_subscribers()
    deaf = [topic for topic, count in listeners.items() if count == 0]
    if deaf:
        detail = ", ".join(f"{t} ({listeners[t]} subscribers)" for t in listeners)
        message = (
            "nothing is subscribed to " + " and ".join(deaf) + ".\n"
            f"  {detail}\n"
            "  Every command would be published and discarded, which looks "
            "identical to an arm that refuses to move.\n"
            "  Check: is the PSM enabled and homed (ros2 topic echo "
            f"{args.arm}/operating_state --once), is the dVRK console running, "
            f"and is {args.arm} the right arm namespace?"
        )
        if args.execute:
            node.get_logger().error(message)
            node.destroy_node()
            _shutdown()
            return 4
        node.get_logger().warn("DRY RUN: " + message)
    else:
        node.get_logger().info(
            "command topics have listeners: "
            + ", ".join(f"{t}={c}" for t, c in listeners.items())
        )

    node.start_episode(report)
    node.create_timer(1.0 / max(args.rate, 0.1), node.tick)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.finish("abort: keyboard interrupt")
    except Exception as exc:  # including rclpy's ExternalShutdownException
        node.get_logger().error(f"stopping: {type(exc).__name__}: {exc}")
    finally:
        node.destroy_node()
        _shutdown()

    return 0 if sequencer.reason == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
