#!/usr/bin/env python3
"""Does this arm accept a Cartesian setpoint at all?

The smallest possible question, asked with none of this package's machinery in
the way: read ``measured_cp``, publish ONE setpoint a couple of millimetres
away, and report whether the arm moved.

It exists because "the arm does not move" has several very different causes
that look identical from a control loop -- nothing subscribed to the command
topic, an arm that is enabled but will not take Cartesian commands, a frame
mismatch between what ``measured_cp`` reports and what ``servo_cp`` expects, or
a genuine controller problem -- and a loop that keeps issuing commands cannot
tell them apart.  One command, one measurement, one answer.

    # look only, publish nothing
    python3 tools/poke_arm.py --arm /PSM1

    # actually move 2 mm along +x of the reported frame
    python3 tools/poke_arm.py --arm /PSM1 --axis x --distance-mm 2 --execute

Nothing here is clamped by the deployment's safety envelope, because there is
no envelope: it is one small displacement of your choosing.  Keep it small.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import PoseStamped
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ROS 2 is not on the path. Run: source /opt/ros/humble/setup.bash"
    ) from exc

AXES = {"x": 0, "y": 1, "z": 2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="/PSM1")
    ap.add_argument("--interface", choices=["servo_cp", "move_cp"], default="servo_cp")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--distance-mm", type=float, default=2.0)
    ap.add_argument("--repeats", type=int, default=20,
                    help="how many times to republish the SAME setpoint; "
                         "servo_cp is a stream and one message may be missed")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--sub-reliability", choices=["best_effort", "reliable"],
                    default="best_effort")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    if abs(args.distance_mm) > 10.0:
        raise SystemExit("this is a poke, not a move; keep it under 10 mm")

    arm = args.arm.rstrip("/")
    rclpy.init()
    node = rclpy.create_node("surgicai_poke_arm")

    sub_qos = QoSProfile(depth=10)
    sub_qos.reliability = (
        ReliabilityPolicy.RELIABLE if args.sub_reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    pub_qos = QoSProfile(depth=10)
    pub_qos.reliability = ReliabilityPolicy.RELIABLE

    latest = {}

    def on_pose(msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        latest["p"] = np.array([p.x, p.y, p.z])
        latest["q"] = np.array([q.x, q.y, q.z, q.w])
        latest["frame"] = msg.header.frame_id

    node.create_subscription(PoseStamped, f"{arm}/measured_cp", on_pose, sub_qos)
    pub = node.create_publisher(PoseStamped, f"{arm}/{args.interface}", pub_qos)

    deadline = time.monotonic() + 5.0
    while "p" not in latest and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "p" not in latest:
        print(f"nothing on {arm}/measured_cp", file=sys.stderr)
        rclpy.shutdown()
        return 2

    # let discovery finish before asking who is listening
    deadline = time.monotonic() + 2.0
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    listeners = pub.get_subscription_count()
    start = latest["p"].copy()
    print(f"arm            : {arm}")
    print(f"command topic  : {arm}/{args.interface}")
    print(f"subscribers    : {listeners}"
          + ("   <-- NOBODY IS LISTENING. Commands go nowhere."
             if listeners == 0 else ""))
    print(f"measured frame : {latest['frame']!r}")
    print(f"measured cm    : {np.round(start * 100, 4)}")

    target = start.copy()
    target[AXES[args.axis]] += args.distance_mm / 1000.0
    print(f"target cm      : {np.round(target * 100, 4)}   "
          f"({args.distance_mm:+.1f} mm along {args.axis})")

    if not args.execute:
        print()
        print("look only; nothing was published. Add --execute to move.")
        rclpy.shutdown()
        return 0

    msg = PoseStamped()
    msg.header.frame_id = latest["frame"]
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = target
    (
        msg.pose.orientation.x, msg.pose.orientation.y,
        msg.pose.orientation.z, msg.pose.orientation.w,
    ) = latest["q"]

    period = 1.0 / max(args.rate, 0.1)
    for _ in range(max(args.repeats, 1)):
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        end = time.monotonic() + period
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.01)

    deadline = time.monotonic() + args.settle_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    moved = (latest["p"] - start) * 1000.0
    travelled = float(np.linalg.norm(moved))
    print()
    print(f"moved mm       : {np.round(moved, 3)}   |{travelled:.3f}| mm")
    print(f"asked for      : {args.distance_mm:.3f} mm")
    print()
    if travelled < 0.1:
        print("THE ARM DID NOT MOVE.")
        if listeners == 0:
            print("  Nothing is subscribed to the command topic. That is the")
            print("  whole answer -- find out what should be, or which")
            print("  namespace the arm really uses.")
        else:
            print("  Something is subscribed and the setpoint was still")
            print("  refused. Check the dVRK console output at the moment of")
            print("  the poke: a rejected Cartesian goal is usually logged")
            print("  there and nowhere else. Worth ruling out:")
            print("    - is the arm in teleoperation, so the console holds")
            print("      command authority?")
            print(f"    - does {arm}/local/measured_cp exist and differ from")
            print(f"      {arm}/measured_cp? Then a base frame is being applied")
            print("      on the way out, and the setpoint may be expected in")
            print("      the other frame.")
            print("    - does the tool need engaging before Cartesian motion?")
    elif abs(travelled - abs(args.distance_mm)) < 0.5:
        print("The arm accepts Cartesian setpoints on this topic and in this")
        print("frame. Whatever stops the pipeline is not this.")
    else:
        print("The arm moved, but not by what was asked. Check the frame the")
        print("setpoint is interpreted in before running anything longer.")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
