"""Reset-only calibration of commanded needle ranges to settled poses and goals."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PyKDL import Frame, Vector

from RL.utils.utils import frame_to_vector, rotation_geodesic_rad


DEFAULT_RANGES = (
    ("current_narrow", 0.3, 0.2, 30.0),
    ("assumed_real_target", 3.0, 3.0, 30.0),
    ("proposed_sim_end", 3.0, 3.0, 45.0),
    ("legacy_cl_start", 4.0, 35.0, 45.0),
    ("legacy_cl_end", 8.0, 70.0, 90.0),
)


def parse_range(value: str):
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "range must be name,x_mm,y_mm,yaw_deg"
        )
    name = parts[0].strip()
    try:
        x_mm, y_mm, yaw_deg = (float(item) for item in parts[1:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not name or min(x_mm, y_mm, yaw_deg) <= 0.0:
        raise argparse.ArgumentTypeError(
            "range name must be non-empty and magnitudes must be positive"
        )
    return name, x_mm, y_mm, yaw_deg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate reset command ranges using reset() only"
    )
    parser.add_argument("--resets-per-range", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--ros-domain-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument(
        "--range",
        dest="ranges",
        action="append",
        type=parse_range,
        help="name,x_mm,y_mm,yaw_deg; repeat for multiple ranges",
    )
    return parser.parse_args()


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summary(values):
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def pairwise_pose_spread(pose_vectors):
    xyz_max_cm = 0.0
    rotation_max_deg = 0.0
    for index, pose_a in enumerate(pose_vectors):
        for pose_b in pose_vectors[index + 1 :]:
            xyz_max_cm = max(
                xyz_max_cm,
                float(np.linalg.norm(pose_a[:3] - pose_b[:3]) * 100.0),
            )
            rotation_max_deg = max(
                rotation_max_deg,
                float(np.degrees(rotation_geodesic_rad(pose_a, pose_b))),
            )
    return {
        "xyz_pairwise_max_cm": xyz_max_cm,
        "rotation_pairwise_max_deg": rotation_max_deg,
    }


def pose_transfer_error(actual, target):
    return (
        float(np.linalg.norm(actual[:3] - target[:3]) * 100.0),
        float(np.degrees(rotation_geodesic_rad(actual, target))),
    )


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    args = parse_args()
    if args.resets_per_range < 20:
        raise ValueError("resets-per-range must be >= 20 for R2 calibration")
    actual_domain = os.environ.get("ROS_DOMAIN_ID")
    if actual_domain != str(args.ros_domain_id):
        raise RuntimeError(
            f"ROS_DOMAIN_ID={actual_domain!r}; expected {args.ros_domain_id}"
        )
    ranges = tuple(args.ranges or DEFAULT_RANGES)
    if len(ranges) < 4:
        raise ValueError("R2 calibration requires at least four ranges")

    from RL.Approach_env import NeedleResetValidityError, SRC_approach
    from RL.utils.seed import seed_everything

    seed_everything(args.seed)
    env = SRC_approach(
        seed=args.seed,
        reward_type="sparse",
        threshold=np.array([1.0, np.deg2rad(10.0)], dtype=np.float32),
        max_episode_step=200,
        measured_success_reward=True,
        needle_settle_steps=60,
        needle_settle_interval_s=0.1,
        randomize_psm_reset=False,
        randomize_needle_reset=True,
        needle_reset_validity_max_attempts=1,
    )

    scene_manager = env.scene_manager
    rest_frame = Frame(
        scene_manager.needle_rest_rotation,
        Vector(
            scene_manager.needle_init_pos.x(),
            scene_manager.needle_init_pos.y(),
            scene_manager.needle_init_pos.z(),
        ),
    )
    rest_vector = frame_to_vector(rest_frame).astype(np.float64)
    groups = []

    try:
        for group_index, (name, x_mm, y_mm, yaw_deg) in enumerate(ranges):
            range_vector = np.array(
                [x_mm * 1.0e-3, y_mm * 1.0e-3, np.deg2rad(yaw_deg)],
                dtype=np.float32,
            )
            env.random_range = range_vector
            rows = []
            reason_counter: Counter[str] = Counter()
            print(
                "RANGE_START:",
                {
                    "name": name,
                    "range_m_m_rad": range_vector.astype(float).tolist(),
                    "resets": args.resets_per_range,
                },
                flush=True,
            )

            for reset_index in range(args.resets_per_range):
                reset_seed = (
                    args.seed + group_index * 1000 + reset_index
                )
                valid = False
                error = None
                env.reset_validity_audit = None
                env.reset_settle_audit = None
                try:
                    env.reset(seed=reset_seed)
                    valid = True
                except NeedleResetValidityError as exc:
                    error = str(exc)
                except RuntimeError as exc:
                    # Wide legacy ranges can ask the contact-constrained needle
                    # to cross farther than the production 5 mm resolved-pose
                    # guard permits.  That is a measured transfer failure, not
                    # a reason to abort the rest of the calibration sweep.
                    if not str(exc).startswith(
                        "Needle pose randomization did not reach"
                    ):
                        raise
                    error = str(exc)

                audit = getattr(env, "reset_validity_audit", None) or {}
                settle = (
                    audit.get("settle")
                    or getattr(env, "reset_settle_audit", None)
                    or {}
                )
                reason = (
                    "ok"
                    if valid
                    else str(
                        audit.get(
                            "reason",
                            (
                                "transport_failure"
                                if error
                                and error.startswith(
                                    "Needle pose randomization did not reach"
                                )
                                else "reset_invalid"
                            ),
                        )
                    )
                )
                if not valid:
                    for item in reason.split(","):
                        if item:
                            reason_counter[item] += 1

                actual_vector = frame_to_vector(
                    scene_manager.needle.get_pose()
                ).astype(np.float64)
                target_vector = frame_to_vector(
                    scene_manager.last_needle_target_frame
                ).astype(np.float64)
                command_offset = target_vector[:3] - rest_vector[:3]
                actual_offset = actual_vector[:3] - rest_vector[:3]
                transfer_xyz_cm, transfer_so3_deg = pose_transfer_error(
                    actual_vector, target_vector
                )
                actual_radius_cm = float(
                    np.linalg.norm(actual_offset) * 100.0
                )
                actual_so3_from_rest_deg = float(
                    np.degrees(
                        rotation_geodesic_rad(actual_vector, rest_vector)
                    )
                )
                row = {
                    "reset": reset_index,
                    "seed": reset_seed,
                    "valid": valid,
                    "reason": reason,
                    "error": error,
                    "settled": bool(settle.get("settled", False)),
                    "settle_steps": settle.get("steps"),
                    "translation_drift_cm": settle.get(
                        "translation_drift_cm"
                    ),
                    "command_offset_m_m_rad": [
                        float(command_offset[0]),
                        float(command_offset[1]),
                        float(scene_manager.last_needle_target_rz),
                    ],
                    "target_pose_world_m_rad": target_vector.tolist(),
                    "actual_pose_world_m_rad": actual_vector.tolist(),
                    "actual_offset_xyz_m": actual_offset.tolist(),
                    "actual_radius_cm": actual_radius_cm,
                    "actual_so3_from_rest_deg": actual_so3_from_rest_deg,
                    "command_to_actual_xyz_cm": transfer_xyz_cm,
                    "command_to_actual_so3_deg": transfer_so3_deg,
                    "goal_pose_m_rad": (
                        np.asarray(env.needle_obs[:6], dtype=np.float64).tolist()
                        if valid
                        else None
                    ),
                }
                rows.append(row)
                print(
                    f"RANGE_RESET name={name} i={reset_index:02d} "
                    f"valid={valid} settled={row['settled']} "
                    f"radius_cm={actual_radius_cm:.4f} "
                    f"so3_rest_deg={actual_so3_from_rest_deg:.3f} "
                    f"transfer_cm={transfer_xyz_cm:.4f} "
                    f"transfer_deg={transfer_so3_deg:.3f} "
                    f"reason={reason}",
                    flush=True,
                )

            settled_rows = [row for row in rows if row["settled"]]
            valid_rows = [row for row in rows if row["valid"]]
            actual_poses = [
                np.asarray(row["actual_pose_world_m_rad"], dtype=np.float64)
                for row in settled_rows
            ]
            goal_poses = [
                np.asarray(row["goal_pose_m_rad"], dtype=np.float64)
                for row in valid_rows
            ]
            actual_dx_abs_mm = [
                abs(row["actual_offset_xyz_m"][0]) * 1000.0
                for row in settled_rows
            ]
            actual_dy_abs_mm = [
                abs(row["actual_offset_xyz_m"][1]) * 1000.0
                for row in settled_rows
            ]
            actual_dz_abs_mm = [
                abs(row["actual_offset_xyz_m"][2]) * 1000.0
                for row in settled_rows
            ]
            metrics = {
                "requested_resets": len(rows),
                "settled_placements": len(settled_rows),
                "valid_resets": len(valid_rows),
                "invalid_resets": len(rows) - len(valid_rows),
                "invalid_reasons": dict(reason_counter),
                "actual_pairwise": pairwise_pose_spread(actual_poses),
                "actual_radius_cm": summary(
                    [row["actual_radius_cm"] for row in settled_rows]
                ),
                "actual_so3_from_rest_deg": summary(
                    [
                        row["actual_so3_from_rest_deg"]
                        for row in settled_rows
                    ]
                ),
                "actual_abs_x_mm": summary(actual_dx_abs_mm),
                "actual_abs_y_mm": summary(actual_dy_abs_mm),
                "actual_abs_z_mm": summary(actual_dz_abs_mm),
                "command_to_actual_xyz_cm": summary(
                    [
                        row["command_to_actual_xyz_cm"]
                        for row in settled_rows
                    ]
                ),
                "command_to_actual_so3_deg": summary(
                    [
                        row["command_to_actual_so3_deg"]
                        for row in settled_rows
                    ]
                ),
                "translation_drift_cm": summary(
                    [
                        float(row["translation_drift_cm"])
                        for row in settled_rows
                        if row["translation_drift_cm"] is not None
                    ]
                ),
                "goal_pairwise": pairwise_pose_spread(goal_poses),
            }
            groups.append(
                {
                    "name": name,
                    "command_range": {
                        "x_mm": x_mm,
                        "y_mm": y_mm,
                        "yaw_deg": yaw_deg,
                        "range_m_m_rad": range_vector.astype(float).tolist(),
                    },
                    "metrics": metrics,
                    "rows": rows,
                }
            )
            print(
                "RANGE_DONE:",
                json.dumps(json_ready({"name": name, **metrics})),
                flush=True,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
        for psm in getattr(scene_manager, "psm_list", []):
            with psm._cmd_lock:
                psm._cmd = None
            with psm._actuator_cmd_lock:
                psm._actuator_cmd = None
        time.sleep(0.1)
        ral_instance = getattr(env, "ral_instance", None)
        if ral_instance is not None and hasattr(ral_instance, "shutdown"):
            ral_instance.shutdown()

    payload = {
        "date": datetime.now().isoformat(),
        "ros_domain_id": args.ros_domain_id,
        "resets_per_range": args.resets_per_range,
        "total_reset_calls": args.resets_per_range * len(groups),
        "r1_settling": {
            "stable_steps": 5,
            "translation_step_threshold_mm": 0.2,
            "rotation_step_threshold_deg": 0.3,
            "max_steps": 60,
        },
        "groups": groups,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(json_ready(payload), indent=2),
        encoding="utf-8",
    )

    lines = [
        "# SurgicAI R2 needle-range transfer calibration",
        "",
        (
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"ROS_DOMAIN_ID: {args.ros_domain_id} (isolated)"
        ),
        "",
        "## Protocol",
        "",
        (
            f"- {len(groups)} command ranges × "
            f"{args.resets_per_range} reset calls = "
            f"**{args.resets_per_range * len(groups)} reset-only trials**."
        ),
        (
            "- Every reset used the R1 quasi-static hold/release mechanism: "
            "five stable held samples, then five stable free-body samples; "
            "0.2 mm/step, 0.3°/step, maximum 60 samples."
        ),
        (
            "- Validity attempts were fixed to one so every reset call "
            "represents exactly one commanded draw; invalid placements are "
            "reported rather than silently resampled."
        ),
        "- No policy, `learn()`, training, or demo collection was invoked.",
        "",
        "## Command → settled pose → goal",
        "",
        (
            "| range | command ±x/±y/±yaw | resets/stable/valid | "
            "actual radius p50/p95 (cm) | actual SO(3) p50/p95 (°) | "
            "actual pairwise xyz/SO(3) | transfer error p50/p95 "
            "xyz (cm) / SO(3) (°) | goal pairwise xyz/SO(3) |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for group in groups:
        command = group["command_range"]
        metrics = group["metrics"]
        actual_radius = metrics["actual_radius_cm"]
        actual_so3 = metrics["actual_so3_from_rest_deg"]
        actual_pairwise = metrics["actual_pairwise"]
        transfer_xyz = metrics["command_to_actual_xyz_cm"]
        transfer_so3 = metrics["command_to_actual_so3_deg"]
        goal_pairwise = metrics["goal_pairwise"]
        lines.append(
            f"| `{group['name']}` | "
            f"{command['x_mm']:.1f}/{command['y_mm']:.1f}/"
            f"{command['yaw_deg']:.0f} | "
            f"{metrics['requested_resets']}/"
            f"{metrics['settled_placements']}/{metrics['valid_resets']} | "
            f"{fmt(actual_radius['p50'])}/{fmt(actual_radius['p95'])} | "
            f"{fmt(actual_so3['p50'])}/{fmt(actual_so3['p95'])} | "
            f"{fmt(actual_pairwise['xyz_pairwise_max_cm'])} cm / "
            f"{fmt(actual_pairwise['rotation_pairwise_max_deg'])}° | "
            f"{fmt(transfer_xyz['p50'])}/{fmt(transfer_xyz['p95'])} / "
            f"{fmt(transfer_so3['p50'])}/{fmt(transfer_so3['p95'])} | "
            f"{fmt(goal_pairwise['xyz_pairwise_max_cm'])} cm / "
            f"{fmt(goal_pairwise['rotation_pairwise_max_deg'])}° |"
        )

    lines.extend(
        [
            "",
            "## Settled coordinate magnitudes",
            "",
            (
                "| range | |x| p50/p95 (mm) | |y| p50/p95 (mm) | "
                "|z| p50/p95 (mm) | invalid reasons |"
            ),
            "|---|---:|---:|---:|---|",
        ]
    )
    for group in groups:
        metrics = group["metrics"]
        x_stats = metrics["actual_abs_x_mm"]
        y_stats = metrics["actual_abs_y_mm"]
        z_stats = metrics["actual_abs_z_mm"]
        lines.append(
            f"| `{group['name']}` | "
            f"{fmt(x_stats['p50'])}/{fmt(x_stats['p95'])} | "
            f"{fmt(y_stats['p50'])}/{fmt(y_stats['p95'])} | "
            f"{fmt(z_stats['p50'])}/{fmt(z_stats['p95'])} | "
            f"`{json.dumps(metrics['invalid_reasons'], sort_keys=True)}` |"
        )

    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            (
                "- `actual radius` is translation from the one-time natural "
                "rest position; actual SO(3) is geodesic rotation from "
                "`R_rest`."
            ),
            (
                "- `actual pairwise` is the maximum pairwise distance among "
                "all placements that reached the R1 steady-state condition."
            ),
            (
                "- `transfer error` compares each settled pose with its own "
                "commanded `R_rest * Rz(rz)` target."
            ),
            (
                "- `goal pairwise` uses reset-time `needle_obs` goals from "
                "valid resets only."
            ),
            f"- Raw evidence: `{args.json_output.name}`.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REPORT_WRITTEN:", args.output, flush=True)
    print("RAW_WRITTEN:", args.json_output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
