"""Reset-only validation for quasi-static needle placement and validity gating.

Run from the SurgicAI root with AMBF on the same isolated ROS domain:

    python3 -m RL.validate_reset_gate --episodes 30 --ros-domain-id 77 \
        --output records/logs/SurgicAI_reset_settle_validation_20260727.md

The harness calls reset() only.  It never invokes a policy, learn(), or demo
collection, and it leaves random_range and the task success contract unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from RL.utils.utils import frame_to_vector, rotation_geodesic_rad


BEFORE = {
    "translation_drift_p95_cm": 5.93,
    "rejection_rate": 0.608,
    "hard_failures": 1,
    "goal_xyz_pairwise_max_cm": 5.656,
    "goal_rotation_pairwise_max_deg": 179.775,
    "gate_so3_max_deg": 60.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Needle reset settling and validity-gate validation"
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--needle-settle-steps", type=int, default=60)
    parser.add_argument("--needle-settle-interval-s", type=float, default=0.1)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--ros-domain-id", type=int, required=True)
    return parser.parse_args()


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def pairwise_pose_spread(pose_vectors):
    xyz_max_cm = 0.0
    rotation_max_deg = 0.0
    for i, pose_a in enumerate(pose_vectors):
        for pose_b in pose_vectors[i + 1 :]:
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


def main() -> int:
    args = parse_args()
    actual_domain = os.environ.get("ROS_DOMAIN_ID")
    if actual_domain != str(args.ros_domain_id):
        raise RuntimeError(
            "Isolated-domain check failed: "
            f"ROS_DOMAIN_ID={actual_domain!r}, expected {args.ros_domain_id}"
        )

    from RL.Approach_env import NeedleResetValidityError, SRC_approach
    from RL.utils.seed import seed_everything

    seed_everything(args.seed)
    env = SRC_approach(
        seed=args.seed,
        reward_type="sparse",
        threshold=np.array([1.0, np.deg2rad(10.0)], dtype=np.float32),
        max_episode_step=200,
        measured_success_reward=True,
        needle_settle_steps=args.needle_settle_steps,
        needle_settle_interval_s=args.needle_settle_interval_s,
        randomize_psm_reset=True,
        randomize_needle_reset=True,
    )

    gate_cfg = {
        "gate_on": bool(env.needle_reset_validity_gate),
        "xy_max_cm": env.needle_valid_xy_max_m * 100.0,
        "z_pad_m": env.needle_valid_z_pad_m,
        "z_tol_cm": env.needle_valid_z_tol_m * 100.0,
        "so3_max_deg": float(np.degrees(env.needle_valid_so3_max_rad)),
        "max_attempts": int(env.needle_reset_validity_max_attempts),
        "needle_random_range_m_m_rad": np.asarray(
            env.random_range, dtype=float
        ).tolist(),
        "needle_settle_max_steps": int(env.needle_settle_steps),
        "needle_settle_interval_s": float(env.needle_settle_interval_s),
        "stable_steps_required": int(env.needle_settle_stable_steps),
        "translation_step_threshold_mm": (
            env.needle_settle_translation_tol_m * 1000.0
        ),
        "rotation_step_threshold_deg": float(
            np.degrees(env.needle_settle_rotation_tol_rad)
        ),
        "rest_quaternion_xyzw": list(
            env.scene_manager.needle_rest_quaternion
        ),
        "orientation_reference": "R_rest * Rz(commanded_rz)",
        "placement_mode": (
            "continuous Cartesian position hold, release, then free-body "
            "steady-state verification"
        ),
    }
    print("RESET_CONFIG:", json.dumps(gate_cfg), flush=True)

    records = []
    attempt_audits = []
    accepted_goal_poses = []
    accepted_needle_poses = []
    reason_counter: Counter[str] = Counter()
    total_rejections = 0
    total_attempts = 0
    resets_with_retry = 0
    hard_failures = 0

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            try:
                env.reset(seed=episode_seed)
                attempts = int(getattr(env, "reset_attempts_used", 1))
                rejections = list(
                    getattr(env, "reset_validity_rejections", [])
                )
                final = getattr(env, "reset_validity_audit", None)
                hard_fail = False
                accepted_goal_poses.append(
                    np.asarray(env.needle_obs[:6], dtype=np.float64)
                )
                accepted_needle_poses.append(
                    frame_to_vector(
                        env.scene_manager.needle.get_pose()
                    ).astype(np.float64)
                )
            except NeedleResetValidityError as exc:
                attempts = int(
                    getattr(
                        env,
                        "reset_attempts_used",
                        env.needle_reset_validity_max_attempts,
                    )
                )
                rejections = list(
                    getattr(env, "reset_validity_rejections", [])
                )
                final = getattr(env, "reset_validity_audit", None)
                hard_fail = True
                hard_failures += 1
                print(f"HARD_FAIL episode={episode}: {exc}", flush=True)

            total_attempts += attempts
            total_rejections += len(rejections)
            if attempts > 1 or hard_fail:
                resets_with_retry += 1
            for rejection in rejections:
                for reason in str(rejection.get("reason", "")).split(","):
                    if reason and reason != "ok":
                        reason_counter[reason] += 1
                attempt_audits.append(rejection)
            if not hard_fail and final is not None:
                attempt_audits.append(final)

            settle = (
                final.get("settle")
                if final is not None
                else getattr(env, "reset_settle_audit", None)
            )
            row = {
                "episode": episode,
                "seed": episode_seed,
                "attempts": attempts,
                "rejections": len(rejections),
                "hard_fail": hard_fail,
                "final_xy_dev_cm": (
                    round(final["xy_dev_cm"], 6) if final else None
                ),
                "final_z_dev_cm": (
                    round(final["z_dev_cm"], 6) if final else None
                ),
                "final_so3_deg": (
                    round(final["so3_to_target_deg"], 6) if final else None
                ),
                "translation_drift_cm": (
                    round(settle["translation_drift_cm"], 6)
                    if settle
                    else None
                ),
                "settle_steps": settle.get("steps") if settle else None,
                "release_step": settle.get("release_step") if settle else None,
                "settled": settle.get("settled") if settle else None,
                "reject_reasons": [
                    rejection.get("reason") for rejection in rejections
                ],
            }
            records.append(row)
            print(
                f"RESET {episode:02d}: attempts={attempts} "
                f"rejections={len(rejections)} hard_fail={hard_fail} "
                f"drift={row['translation_drift_cm']}cm "
                f"so3={row['final_so3_deg']}deg "
                f"settle_steps={row['settle_steps']} "
                f"reasons={row['reject_reasons']}",
                flush=True,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass
        # PSM command threads are daemon loops.  Clear their pending messages
        # before shutting down the shared ROS context so they cannot publish
        # once the context is invalid.
        scene_manager = getattr(env, "scene_manager", None)
        for psm in getattr(scene_manager, "psm_list", []):
            with psm._cmd_lock:
                psm._cmd = None
            with psm._actuator_cmd_lock:
                psm._actuator_cmd = None
        time.sleep(0.1)
        ral_instance = getattr(env, "ral_instance", None)
        if ral_instance is not None and hasattr(ral_instance, "shutdown"):
            ral_instance.shutdown()

    attempt_settles = [
        audit["settle"]
        for audit in attempt_audits
        if audit.get("settle") is not None
    ]
    translation_drifts_cm = [
        float(settle["translation_drift_cm"]) for settle in attempt_settles
    ]
    final_so3_deg = [
        float(record["final_so3_deg"])
        for record in records
        if not record["hard_fail"] and record["final_so3_deg"] is not None
    ]
    settle_steps = [
        int(settle["steps"]) for settle in attempt_settles
    ]
    n = len(records)
    rejection_rate = (
        total_rejections / total_attempts if total_attempts else 0.0
    )
    retry_rate = resets_with_retry / n if n else 0.0
    goal_spread = pairwise_pose_spread(accepted_goal_poses)
    needle_spread = pairwise_pose_spread(accepted_needle_poses)
    metrics = {
        "resets": n,
        "total_attempts": total_attempts,
        "total_rejections": total_rejections,
        "rejection_rate": rejection_rate,
        "resets_with_retry": resets_with_retry,
        "retry_rate": retry_rate,
        "hard_failures": hard_failures,
        "translation_drift_cm": {
            "count": len(translation_drifts_cm),
            "p50": percentile(translation_drifts_cm, 50),
            "p95": percentile(translation_drifts_cm, 95),
            "max": max(translation_drifts_cm)
            if translation_drifts_cm
            else None,
        },
        "settle_steps": {
            "p50": percentile(settle_steps, 50),
            "p95": percentile(settle_steps, 95),
            "max": max(settle_steps) if settle_steps else None,
        },
        "so3_to_rest_target_deg": {
            "p50": percentile(final_so3_deg, 50),
            "p95": percentile(final_so3_deg, 95),
            "max": max(final_so3_deg) if final_so3_deg else None,
        },
        "goal_distribution": goal_spread,
        "needle_body_distribution": needle_spread,
        "criteria": {
            "translation_drift_p95_lt_0_2_cm": (
                percentile(translation_drifts_cm, 95) is not None
                and percentile(translation_drifts_cm, 95) < 0.2
            ),
            "rejection_rate_lt_10pct": rejection_rate < 0.10,
            "hard_failures_eq_0": hard_failures == 0,
            "so3_clustered_near_zero": (
                percentile(final_so3_deg, 95) is not None
                and percentile(final_so3_deg, 95) < 15.0
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = (
        Path(args.json_output)
        if args.json_output
        else output.with_suffix(".json")
    )
    raw_payload = {
        "date": datetime.now().isoformat(),
        "ros_domain_id": args.ros_domain_id,
        "configuration": gate_cfg,
        "before": BEFORE,
        "metrics": metrics,
        "rejection_reasons": dict(reason_counter),
        "records": records,
        "attempt_audits": attempt_audits,
    }
    raw_output.write_text(
        json.dumps(json_ready(raw_payload), indent=2),
        encoding="utf-8",
    )

    drift = metrics["translation_drift_cm"]
    so3 = metrics["so3_to_rest_target_deg"]
    now = datetime.now()
    lines = [
        "# SurgicAI needle reset settling — validation",
        "",
        (
            f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"ROS_DOMAIN_ID: {args.ros_domain_id} (isolated)"
        ),
        "",
        "## Placement and gate configuration",
        "",
        "```json",
        json.dumps(gate_cfg, indent=2),
        "```",
        "",
        "## Before / after",
        "",
        "| Metric | Before (2026-07-21) | After (this 30-reset run) | Target |",
        "|---|---:|---:|---:|",
        (
            "| settle `translation_drift_cm` p95 | "
            f"{BEFORE['translation_drift_p95_cm']:.3f} cm | "
            f"{drift['p95']:.6f} cm | < 0.2 cm |"
        ),
        (
            f"| gate rejection rate | {BEFORE['rejection_rate']:.1%} | "
            f"{rejection_rate:.1%} | < 10% |"
        ),
        (
            f"| hard failures | {BEFORE['hard_failures']}/30 | "
            f"{hard_failures}/{n} | 0/30 |"
        ),
        (
            "| goal xyz pairwise max | "
            f"{BEFORE['goal_xyz_pairwise_max_cm']:.3f} cm | "
            f"{goal_spread['xyz_pairwise_max_cm']:.6f} cm | — |"
        ),
        (
            "| goal SO(3) pairwise max | "
            f"{BEFORE['goal_rotation_pairwise_max_deg']:.3f}° | "
            f"{goal_spread['rotation_pairwise_max_deg']:.6f}° | — |"
        ),
        (
            f"| gate SO(3) threshold | {BEFORE['gate_so3_max_deg']:.1f}° | "
            f"{gate_cfg['so3_max_deg']:.1f}° | 15° |"
        ),
        "",
        "## Summary",
        "",
        f"- Resets performed: **{n}**",
        f"- Total placement attempts: **{total_attempts}**",
        f"- Total rejections: **{total_rejections}** ({rejection_rate:.1%})",
        f"- Resets needing retry: **{resets_with_retry}** ({retry_rate:.1%})",
        f"- Hard failures: **{hard_failures}/{n}**",
        (
            "- Translation drift p50 / p95 / max: "
            f"**{drift['p50']:.6f} / {drift['p95']:.6f} / "
            f"{drift['max']:.6f} cm**"
        ),
        (
            "- SO(3) to `R_rest * Rz(rz)` p50 / p95 / max: "
            f"**{so3['p50']:.6f} / {so3['p95']:.6f} / "
            f"{so3['max']:.6f}°**"
        ),
        (
            "- Goal pairwise max: "
            f"**{goal_spread['xyz_pairwise_max_cm']:.6f} cm / "
            f"{goal_spread['rotation_pairwise_max_deg']:.6f}°**"
        ),
        (
            "- Settled needle-body pairwise max: "
            f"**{needle_spread['xyz_pairwise_max_cm']:.6f} cm / "
            f"{needle_spread['rotation_pairwise_max_deg']:.6f}°**"
        ),
        "",
        "### Rejection reason distribution",
        "",
    ]
    if reason_counter:
        for reason, count in reason_counter.most_common():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None; every placement passed on its first attempt.")

    lines.extend(
        [
            "",
            "## Completion criteria",
            "",
        ]
    )
    for criterion, passed in metrics["criteria"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{criterion}`")

    lines.extend(
        [
            "",
            "## Per-reset detail",
            "",
            (
                "| ep | seed | attempts | rejects | hard fail | drift (cm) | "
                "settle steps | release step | final SO(3) (deg) | reasons |"
            ),
            (
                "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---|"
            ),
        ]
    )
    for record in records:
        lines.append(
            f"| {record['episode']} | {record['seed']} | "
            f"{record['attempts']} | {record['rejections']} | "
            f"{'Y' if record['hard_fail'] else ''} | "
            f"{record['translation_drift_cm']} | {record['settle_steps']} | "
            f"{record['release_step']} | {record['final_so3_deg']} | "
            f"{', '.join(str(x) for x in record['reject_reasons'])} |"
        )

    lines.extend(
        [
            "",
            "## Method and scope",
            "",
            (
                "- Chosen placement mechanism: continuously republish the "
                "target Cartesian pose until five held samples are steady, "
                "release once, then require five free-body steady samples. "
                "This reuses the project-validated Stage-C command path and "
                "avoids adding 2 mm of gravitational impact energy."
            ),
            (
                "- A timeout at 60 samples is a rejected placement and triggers "
                "a full reset retry; it is never accepted as a goal."
            ),
            (
                "- `R_rest` is captured once from the measured natural startup "
                "pose. Placement and the gate both use `R_rest * Rz(rz)`."
            ),
            (
                "- This harness calls `reset()` only. It does not invoke a "
                "policy, `learn()`, retraining, or demo collection."
            ),
            (
                "- `random_range`, `trans_tolerance`, `angle_tolerance`, and "
                "the pose-close success contract were not changed."
            ),
            f"- Raw machine-readable evidence: `{raw_output.name}`.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REPORT_WRITTEN:", str(output), flush=True)
    print("RAW_WRITTEN:", str(raw_output), flush=True)
    print(
        f"DONE resets={n} attempts={total_attempts} "
        f"rejections={total_rejections} hard_fails={hard_failures} "
        f"goal_pairwise_cm={goal_spread['xyz_pairwise_max_cm']:.6f} "
        f"goal_pairwise_deg={goal_spread['rotation_pairwise_max_deg']:.6f}",
        flush=True,
    )
    return 0 if all(metrics["criteria"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
