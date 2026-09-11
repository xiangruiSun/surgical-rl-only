"""ROS 2 node that runs the approach loop against a real dVRK PSM.

Defaults are deliberately timid:

* **dry run** -- nothing is published until you pass ``--execute``;
* the goal is **frozen** at the first measured pose, exactly like the training
  contract (one ``desired_goal`` per episode);
* every command is clamped to a box around start+goal and to a per-step
  translation/rotation cap;
* the loop aborts if the arm falls behind the command, if ``measured_cp`` goes
  stale, or if the arm leaves its operating state.

Topics (all under ``--arm``, default PSM1):

    subscribe  <arm>/measured_cp        geometry_msgs/PoseStamped
               <arm>/jaw/measured_js    sensor_msgs/JointState      (optional)
               <arm>/goal_reached       std_msgs/Bool               (move_cp only)
    publish    <arm>/servo_cp           geometry_msgs/PoseStamped
               <arm>/move_cp            geometry_msgs/PoseStamped
               <arm>/jaw/servo_jp       sensor_msgs/JointState      (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ROS 2 is not on the path. Run: source /opt/ros/humble/setup.bash"
    ) from exc

from .controllers import D2Controller, RLController, ResidualController
from .frames import Pose
from .loop import ApproachLoop, LoopConfig, SafetyLimits


class ApproachNode(Node):
    def __init__(self, args):
        super().__init__("surgicai_rl_approach")
        self.args = args
        self.arm = args.arm.rstrip("/")

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE

        self._measured = None
        self._measured_stamp = 0.0
        self._measured_frame = None
        self._jaw_rad = None
        self._goal_reached = None

        self.create_subscription(
            PoseStamped, f"{self.arm}/measured_cp", self._on_measured_cp, qos
        )
        self.create_subscription(
            JointState, f"{self.arm}/jaw/measured_js", self._on_jaw, qos
        )
        if args.interface == "move_cp":
            self.create_subscription(
                Bool, f"{self.arm}/goal_reached", self._on_goal_reached, qos
            )

        self.cmd_pub = self.create_publisher(PoseStamped, f"{self.arm}/{args.interface}", qos)
        self.jaw_pub = (
            self.create_publisher(JointState, f"{self.arm}/jaw/servo_jp", qos)
            if args.use_policy_jaw
            else None
        )

        self.loop = None
        self.trace_file = open(args.trace, "w") if args.trace else None
        self.finished = False
        self._started = False

    # -- callbacks ---------------------------------------------------------
    def _on_measured_cp(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self._measured = (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))
        self._measured_stamp = time.monotonic()
        self._measured_frame = msg.header.frame_id

    def _on_jaw(self, msg: JointState):
        if msg.position:
            self._jaw_rad = float(msg.position[0])

    def _on_goal_reached(self, msg: Bool):
        self._goal_reached = bool(msg.data)

    # -- helpers -----------------------------------------------------------
    def _jaw_norm(self) -> float:
        if self._jaw_rad is None:
            return 0.0
        return float(np.clip(self._jaw_rad / self.args.jaw_open_rad, 0.0, 1.0))

    def _current_pose(self):
        if self._measured is None:
            return None
        pos, quat = self._measured
        return Pose.from_pos_quat(pos, quat, self._jaw_norm())

    def _publish(self, pose: Pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._measured_frame or self.args.expect_frame
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose.p
        q = pose.quat_xyzw()
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = q
        self.cmd_pub.publish(msg)
        if self.jaw_pub is not None:
            js = JointState()
            js.header.stamp = msg.header.stamp
            js.position = [float(pose.jaw) * self.args.jaw_open_rad]
            self.jaw_pub.publish(js)

    def _log(self, record: dict):
        if self.trace_file:
            self.trace_file.write(json.dumps(record) + "\n")
            self.trace_file.flush()

    # -- main loop ---------------------------------------------------------
    def start_episode(self, controller, cfg: LoopConfig, limits: SafetyLimits):
        start = self._current_pose()
        if start is None:
            raise RuntimeError(f"no message on {self.arm}/measured_cp")
        if self.args.expect_frame and self._measured_frame != self.args.expect_frame:
            self.get_logger().warn(
                f"measured_cp frame_id is {self._measured_frame!r}, expected "
                f"{self.args.expect_frame!r}. The goal you passed must be in the "
                "same frame as measured_cp."
            )
        self.loop = ApproachLoop(controller, cfg, limits)
        report = self.loop.begin(start, self.args.goal_pos)

        self.get_logger().info(f"controller     : {controller.describe()}")
        self.get_logger().info(f"frame          : {self._measured_frame}")
        self.get_logger().info(f"start          : {np.round(start.p * 100, 3)} cm")
        self.get_logger().info(f"goal           : {np.round(np.asarray(self.args.goal_pos) * 100, 3)} cm")
        self.get_logger().info(f"path length    : {report['translation_cm']:.2f} cm")
        if report["in_distribution"]:
            self.get_logger().info("training support: INSIDE the R6 demonstration support")
        else:
            self.get_logger().warn("training support: OUTSIDE the R6 demonstration support")
            for line in report["out_of_distribution"]:
                self.get_logger().warn(f"  - {line}")
        if not self.args.execute:
            self.get_logger().info("DRY RUN: no command will be published")
        self._log({"event": "begin", "report": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in report.items()
        }, "start_cm": (start.p * 100).tolist(), "goal_cm": (np.asarray(self.args.goal_pos) * 100).tolist()})
        self._started = True

    def tick(self):
        if self.finished or not self._started:
            return
        age = time.monotonic() - self._measured_stamp
        if age > self.loop.limits.max_pose_age_s:
            self.get_logger().error(f"measured_cp stale ({age:.2f} s) - stopping")
            self.finish("abort: stale measured_cp")
            return

        measured = self._current_pose()
        result = self.loop.step(measured)

        self.get_logger().info(
            f"step {result.index:3d}  err {result.trans_err_cm:6.2f} cm / "
            f"{result.rot_err_deg:6.2f} deg  action {np.round(result.action, 2)}"
            + (f"  clamped[{','.join(c['kind'] for c in result.clamps)}]" if result.clamps else "")
        )
        self._log({
            "event": "step", "i": result.index,
            "measured_cm": (measured.p * 100).tolist(),
            "command_cm": (result.command.p * 100).tolist(),
            "action": result.action.tolist(),
            "trans_err_cm": result.trans_err_cm, "rot_err_deg": result.rot_err_deg,
            "clamps": result.clamps, "reason": result.reason,
        })

        if self.args.execute and not result.reason.startswith("abort"):
            self._publish(result.command)

        if result.done:
            self.finish(result.reason)

    def finish(self, reason: str):
        self.finished = True
        level = self.get_logger().info if reason == "success" else self.get_logger().warn
        level(f"episode finished: {reason}")
        self._log({"event": "finish", "reason": reason})
        if self.trace_file:
            self.trace_file.close()
            self.trace_file = None


def build_controller(args):
    if args.controller == "d2":
        return D2Controller(staged=True)
    from .policy import ApproachPolicy

    if not args.model:
        raise SystemExit("--model is required for the rl/residual controllers")
    policy = ApproachPolicy.load(args.model, device=args.device,
                                 verify=not args.allow_unknown_model)
    if args.controller == "rl":
        return RLController(policy)
    return ResidualController(policy, policy_weight=args.policy_weight,
                              servo_weight=args.servo_weight)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="/PSM1")
    ap.add_argument("--goal-pos", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"),
                    help="goal position in metres, in the SAME frame as measured_cp")
    ap.add_argument("--goal-quat", nargs=4, type=float, default=None,
                    metavar=("QX", "QY", "QZ", "QW"))
    ap.add_argument("--goal-orientation", choices=["hold", "trained_relative", "explicit"],
                    default=None)
    ap.add_argument("--controller", choices=["rl", "d2", "residual"], default="rl")
    ap.add_argument("--model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-unknown-model", action="store_true")
    ap.add_argument("--frame-mode", choices=["rebase", "translate", "identity"], default="rebase")
    ap.add_argument("--interface", choices=["servo_cp", "move_cp"], default="servo_cp")
    ap.add_argument("--rate", type=float, default=10.0, help="control rate in Hz")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--success-trans-cm", type=float, default=1.0)
    ap.add_argument("--success-rot-deg", type=float, default=10.0)
    ap.add_argument("--use-policy-jaw", action="store_true",
                    help="publish the policy's jaw command; off by default")
    ap.add_argument("--jaw-open-rad", type=float, default=1.0,
                    help="jaw angle that maps to normalised 1.0")
    ap.add_argument("--expect-frame", default="ECM")
    ap.add_argument("--policy-weight", type=float, default=0.50)
    ap.add_argument("--servo-weight", type=float, default=0.75)
    ap.add_argument("--workspace-pad-cm", type=float, default=2.0)
    ap.add_argument("--max-step-translation-mm", type=float, default=2.5)
    ap.add_argument("--max-step-rotation-deg", type=float, default=5.0)
    ap.add_argument("--max-tracking-error-cm", type=float, default=1.5)
    ap.add_argument("--trace", default=None, help="write a JSONL trace here")
    ap.add_argument("--execute", action="store_true",
                    help="actually publish commands; without it this is a dry run")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rclpy.init()
    node = ApproachNode(args)

    # wait for the first measured pose
    deadline = time.monotonic() + 5.0
    while node._measured is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node._measured is None:
        node.get_logger().error(f"no messages on {node.arm}/measured_cp after 5 s")
        node.destroy_node()
        rclpy.shutdown()
        return 2

    cfg = LoopConfig(
        frame_mode=args.frame_mode,
        goal_orientation=args.goal_orientation or ("explicit" if args.goal_quat else "hold"),
        goal_quat_xyzw=tuple(args.goal_quat) if args.goal_quat else None,
        use_policy_jaw=args.use_policy_jaw,
        max_steps=args.max_steps,
        success_trans_cm=args.success_trans_cm,
        success_rot_rad=float(np.deg2rad(args.success_rot_deg)),
    )
    limits = SafetyLimits(
        workspace_pad_cm=args.workspace_pad_cm,
        max_step_translation_mm=args.max_step_translation_mm,
        max_step_rotation_deg=args.max_step_rotation_deg,
        max_tracking_error_cm=args.max_tracking_error_cm,
    )
    node.start_episode(build_controller(args), cfg, limits)

    timer_period = 1.0 / max(args.rate, 0.1)
    node.create_timer(timer_period, node.tick)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.finish("abort: keyboard interrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
