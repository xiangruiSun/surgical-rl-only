#!/usr/bin/env python3
"""Record what closing on *nothing* looks like on this particular arm.

Without this the jaw readings during a grasp have no reference: a residual of
two degrees means nothing until you know that an empty jaw on this arm settles
at zero point three.  Tendon tension, backlash and jaw zeroing all drift, so
this is per-arm and worth redoing after a tool change.

    source /opt/ros/humble/setup.bash
    python3 tools/calibrate_jaw.py --arm /PSM1 --out jaw_baseline.json --execute

**Make sure the gripper is empty and clear of everything before running.** The
tool commands the jaw closed and nothing else; the arm does not move in
Cartesian space.  Dry run by default.

The output feeds ``run_grasp_lift.py --jaw-baseline``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ROS 2 is not on the path. Run: source /opt/ros/humble/setup.bash"
    ) from exc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="/PSM1")
    ap.add_argument("--jaw-grip-deg", type=float, default=-15.0,
                    help="must match the grip angle used in the real run")
    ap.add_argument("--jaw-open-deg", type=float, default=40.0,
                    help="angle the jaw is opened to before each close")
    ap.add_argument("--jaw-interface", choices=["servo_jp", "move_jp"],
                    default="servo_jp")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="how long to hold each command before sampling")
    ap.add_argument("--sample-s", type=float, default=2.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="jaw_baseline.json")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    grip = float(np.deg2rad(args.jaw_grip_deg))
    open_angle = float(np.deg2rad(args.jaw_open_deg))
    arm = args.arm.rstrip("/")

    rclpy.init()
    node = rclpy.create_node("surgicai_jaw_calibration")
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.RELIABLE

    state = {"pos": None, "effort": None}

    def _on_jaw(msg: JointState):
        if msg.position:
            state["pos"] = float(msg.position[0])
        state["effort"] = float(msg.effort[0]) if msg.effort else None

    node.create_subscription(JointState, f"{arm}/jaw/measured_js", _on_jaw, qos)
    pub = node.create_publisher(JointState, f"{arm}/jaw/{args.jaw_interface}", qos)

    deadline = time.monotonic() + 5.0
    while state["pos"] is None and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if state["pos"] is None:
        print(f"no messages on {arm}/jaw/measured_js after 5 s", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 2

    if not args.execute:
        print("DRY RUN: no jaw command will be published. Add --execute.")
    print(f"jaw now at {np.degrees(state['pos']):.2f} deg, "
          f"effort {state['effort']}")
    print(f"will close to {args.jaw_grip_deg:.1f} deg, {args.repeats} time(s).")
    print("THE GRIPPER MUST BE EMPTY.")

    def command(angle, seconds):
        end = time.monotonic() + seconds
        period = 1.0 / max(args.rate, 0.1)
        while time.monotonic() < end:
            if args.execute:
                msg = JointState()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.position = [float(angle)]
                pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=period)

    def sample(angle, seconds):
        positions, efforts = [], []
        end = time.monotonic() + seconds
        period = 1.0 / max(args.rate, 0.1)
        while time.monotonic() < end:
            if args.execute:
                msg = JointState()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.position = [float(angle)]
                pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=period)
            if state["pos"] is not None:
                positions.append(state["pos"])
            if state["effort"] is not None:
                efforts.append(abs(state["effort"]))
        return positions, efforts

    all_pos, all_effort = [], []
    for repeat in range(args.repeats):
        print(f"--- repeat {repeat + 1}/{args.repeats}: opening")
        command(open_angle, args.settle_s)
        print("    closing")
        command(grip, args.settle_s)
        print("    sampling")
        positions, efforts = sample(grip, args.sample_s)
        if positions:
            print(
                f"    settled at {np.degrees(statistics.fmean(positions)):.3f} deg"
                + (f", effort {statistics.fmean(efforts):.4f}" if efforts else
                   ", no effort field")
            )
        all_pos += positions
        all_effort += efforts

    node.destroy_node()
    rclpy.shutdown()

    if not all_pos:
        print("no jaw samples collected", file=sys.stderr)
        return 2

    payload = {
        "empty_close_rad": float(statistics.fmean(all_pos)),
        "empty_close_rad_noise": float(
            max(statistics.pstdev(all_pos) if len(all_pos) > 1 else 0.0,
                np.deg2rad(0.1))
        ),
        "empty_close_effort": (
            float(statistics.fmean(all_effort)) if all_effort else None
        ),
        "empty_close_effort_noise": (
            float(statistics.pstdev(all_effort)) if len(all_effort) > 1 else None
        ),
        "source": (
            f"{arm} empty-jaw close to {args.jaw_grip_deg:.1f} deg, "
            f"{args.repeats} repeats, {len(all_pos)} samples, "
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}"
            + ("" if args.execute else "  [DRY RUN - NOT A REAL MEASUREMENT]")
        ),
        "grip_command_deg": args.jaw_grip_deg,
        "executed": bool(args.execute),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print()
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    if not args.execute:
        print(
            "\nThis was a DRY RUN: the jaw never moved, so the numbers above "
            "describe wherever the jaw already was. Do not use this file as a "
            "baseline.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
