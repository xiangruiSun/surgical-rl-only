"""Collect diverse Approach demonstrations directly from measured AMBF state.

This collector is deliberately separate from the image recorder.  It produces
GoalEnv transitions for TD3+HER+BC, uses a staged measured feedback expert, and
accepts an episode only under an independently recomputed pose-close contract.
Physical needle grasp confirmation is optional and is reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from RL.needle_reset_ranges import (
    ASSUMED_REAL_RZ_DEG,
    ASSUMED_REAL_X_MM,
    ASSUMED_REAL_Y_MM,
)


STEP_SIZE_RAW = np.array(
    [
        1.5e-3,
        1.5e-3,
        1.5e-3,
        np.deg2rad(3.0),
        np.deg2rad(3.0),
        np.deg2rad(3.0),
        0.05,
    ],
    dtype=np.float32,
)
STEP_SIZE_NORMALIZED = STEP_SIZE_RAW * np.array(
    [100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
)


def wrap_to_pi(value):
    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def rpy_matrix(rpy):
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_geodesic_deg(actual, desired) -> float:
    relative = rpy_matrix(actual[3:6]).T @ rpy_matrix(desired[3:6])
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def pose_close_errors(actual, desired) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    return {
        "translation_cm": float(np.linalg.norm(actual[:3] - desired[:3])),
        "rotation_geodesic_deg": rotation_geodesic_deg(actual, desired),
        "measured_jaw": float(actual[6]),
    }


def latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Deterministic stratified design with one sample per marginal bin."""
    if count <= 0 or dimensions <= 0:
        raise ValueError("count and dimensions must be positive")
    rng = np.random.default_rng(seed)
    design = np.empty((count, dimensions), dtype=np.float64)
    centers = (np.arange(count, dtype=np.float64) + 0.5) / count
    for axis in range(dimensions):
        design[:, axis] = centers[rng.permutation(count)]
    return design


def build_reset_design(
    count: int,
    seed: int,
    needle_range: np.ndarray,
    grasp_range_deg: tuple[float, float],
) -> list[dict[str, Any]]:
    unit = latin_hypercube(count, 4, seed)
    low_grasp, high_grasp = grasp_range_deg
    result = []
    for index, point in enumerate(unit):
        offset = (2.0 * point[:3] - 1.0) * needle_range
        grasp = low_grasp + point[3] * (high_grasp - low_grasp)
        result.append(
            {
                "episode": index,
                "unit_sample": point.tolist(),
                "needle_offset_m_m_rad": offset.tolist(),
                "grasp_angle_deg": float(grasp),
            }
        )
    return result


@dataclass
class ExpertDiagnostic:
    phase: str
    pose_ready: bool
    pose_dwell: int
    translation_cm: float
    rotation_geodesic_deg: float
    measured_jaw: float


class MeasuredApproachExpert:
    """Proportional approach, pose dwell, jaw close, and stop controller."""

    def __init__(
        self,
        step_size_raw=STEP_SIZE_RAW,
        translation_threshold_cm=0.1,
        rotation_threshold_deg=10.0,
        pose_dwell_steps=3,
        open_jaw=0.8,
        closed_jaw=0.0,
        max_action=0.8,
        gain=0.7,
    ):
        self.step_size_raw = np.asarray(step_size_raw, dtype=np.float64)
        self.translation_threshold_cm = float(translation_threshold_cm)
        self.rotation_threshold_deg = float(rotation_threshold_deg)
        self.pose_dwell_steps = int(pose_dwell_steps)
        self.open_jaw = float(open_jaw)
        self.closed_jaw = float(closed_jaw)
        self.max_action = float(max_action)
        self.gain = float(gain)
        if self.step_size_raw.shape != (7,) or np.any(self.step_size_raw <= 0):
            raise ValueError("step_size_raw must be a positive seven-vector")
        if self.pose_dwell_steps <= 0:
            raise ValueError("pose_dwell_steps must be positive")
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.pose_dwell = 0

    def action(self, measured, desired) -> tuple[np.ndarray, ExpertDiagnostic]:
        measured = np.asarray(measured, dtype=np.float64)
        desired = np.asarray(desired, dtype=np.float64)
        errors = pose_close_errors(measured, desired)
        pose_ready = bool(
            errors["translation_cm"] <= self.translation_threshold_cm
            and errors["rotation_geodesic_deg"] <= self.rotation_threshold_deg
        )
        if self.phase == "approach":
            self.pose_dwell = self.pose_dwell + 1 if pose_ready else 0
            if self.pose_dwell >= self.pose_dwell_steps:
                self.phase = "close_jaw"

        delta = desired - measured
        delta[3:6] = wrap_to_pi(delta[3:6])
        delta_raw = delta.copy()
        delta_raw[:3] /= 100.0
        action = np.clip(
            self.gain * delta_raw / self.step_size_raw,
            -self.max_action,
            self.max_action,
        )
        jaw_target = self.open_jaw if self.phase == "approach" else self.closed_jaw
        jaw_error = jaw_target - measured[6]
        action[6] = (
            0.0
            if abs(jaw_error) <= 1.0e-6
            else np.clip(
                self.gain * jaw_error / self.step_size_raw[6],
                -self.max_action,
                self.max_action,
            )
        )
        diagnostic = ExpertDiagnostic(
            phase=self.phase,
            pose_ready=pose_ready,
            pose_dwell=self.pose_dwell,
            translation_cm=errors["translation_cm"],
            rotation_geodesic_deg=errors["rotation_geodesic_deg"],
            measured_jaw=errors["measured_jaw"],
        )
        return action.astype(np.float32), diagnostic


def copy_observation(observation):
    return {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in observation.items()
    }


def atomic_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def pairwise_goal_summary(goals: np.ndarray) -> dict[str, Any]:
    xyz = goals[:, :3]
    max_translation = 0.0
    max_translation_pair = [0, 0]
    max_rotation = 0.0
    max_rotation_pair = [0, 0]
    for left in range(len(goals)):
        for right in range(left + 1, len(goals)):
            translation = float(np.linalg.norm(xyz[left] - xyz[right]))
            rotation = rotation_geodesic_deg(goals[left], goals[right])
            if translation > max_translation:
                max_translation = translation
                max_translation_pair = [left, right]
            if rotation > max_rotation:
                max_rotation = rotation
                max_rotation_pair = [left, right]
    return {
        "xyz_mean_cm": xyz.mean(axis=0).tolist(),
        "xyz_std_population_cm": xyz.std(axis=0).tolist(),
        "xyz_min_cm": xyz.min(axis=0).tolist(),
        "xyz_max_cm": xyz.max(axis=0).tolist(),
        "xyz_axis_range_cm": np.ptp(xyz, axis=0).tolist(),
        "xyz_pairwise_max_cm": max_translation,
        "xyz_pairwise_max_pair": max_translation_pair,
        "rotation_pairwise_max_deg": max_rotation,
        "rotation_pairwise_max_pair": max_rotation_pair,
        "episode_goals_xyz_cm_rpy_rad": goals[:, :6].tolist(),
    }


def dataset_summary(episodes, step_size_normalized) -> dict[str, Any]:
    transitions = [transition for episode in episodes for transition in episode]
    goals = np.stack([episode[0]["obs"]["desired_goal"] for episode in episodes])
    residuals = []
    goal_changes = []
    finals = []
    for episode in episodes:
        initial_goal = episode[0]["obs"]["desired_goal"]
        for transition in episode:
            current = transition["obs"]["achieved_goal"]
            predicted = current + transition["action"] * step_size_normalized
            actual = transition["next_obs"]["achieved_goal"]
            residuals.append(float(np.linalg.norm(actual - predicted)))
            goal_changes.append(
                float(
                    np.max(
                        np.abs(transition["next_obs"]["desired_goal"] - initial_goal)
                    )
                )
            )
        finals.append(
            pose_close_errors(
                episode[-1]["next_obs"]["achieved_goal"],
                episode[-1]["next_obs"]["desired_goal"],
            )
        )
    residuals = np.asarray(residuals, dtype=np.float64)
    return {
        "episodes": len(episodes),
        "transitions": len(transitions),
        "episode_lengths": [len(episode) for episode in episodes],
        "goal_distribution": pairwise_goal_summary(goals),
        "within_episode_goal_max_abs_change": float(max(goal_changes, default=0.0)),
        "integrator_residual_norm": {
            "median": float(np.median(residuals)),
            "max": float(np.max(residuals)),
            "fraction_le_1e-5": float(np.mean(residuals <= 1.0e-5)),
        },
        "final_measured_errors": finals,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect measured, stratified, multi-goal Approach demos"
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--max-attempts-per-goal", type=int, default=5)
    parser.add_argument("--needle-random-x-mm", type=float, default=ASSUMED_REAL_X_MM)
    parser.add_argument("--needle-random-y-mm", type=float, default=ASSUMED_REAL_Y_MM)
    parser.add_argument("--needle-random-rz-deg", type=float, default=ASSUMED_REAL_RZ_DEG)
    parser.add_argument("--grasp-start-deg", type=float, default=5.0)
    parser.add_argument("--grasp-end-deg", type=float, default=20.0)
    parser.add_argument("--needle-settle-steps", type=int, default=60)
    parser.add_argument("--needle-settle-interval-s", type=float, default=0.1)
    parser.add_argument("--translation-threshold-mm", type=float, default=1.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=10.0)
    parser.add_argument("--jaw-threshold", type=float, default=0.1)
    parser.add_argument("--pose-dwell-steps", type=int, default=3)
    parser.add_argument("--strict-sync", action="store_true")
    parser.add_argument("--physics-steps-per-action", type=int, default=10)
    parser.add_argument("--physics-barrier-timeout-s", type=float, default=2.0)
    parser.add_argument("--require-grasp-confirmation", action="store_true")
    parser.add_argument("--grasp-confirmation-steps", type=int, default=3)
    parser.add_argument(
        "--pose-close-only",
        action="store_true",
        help=(
            "Benchmark contract: accept on measured pose-close (trans/rot) plus "
            "scripted actuate attach; drop the physically-unattainable measured "
            "jaw<=0.1 requirement (probe 2026-07-21 proved the jaws close on the "
            "phantom, not the needle). Matches original SurgicAI grasp semantics."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path(__file__).resolve().parent / "Expert_traj" / "Approach" / (
        "measured_multigoal_" + timestamp
    )


def main() -> int:
    args = parse_args()
    if args.episodes <= 0 or args.max_attempts_per_goal <= 0:
        raise ValueError("episodes and max-attempts-per-goal must be positive")
    output_dir = (args.output_dir or default_output_dir()).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir()

    needle_range = np.array(
        [
            args.needle_random_x_mm * 1.0e-3,
            args.needle_random_y_mm * 1.0e-3,
            np.deg2rad(args.needle_random_rz_deg),
        ],
        dtype=np.float32,
    )
    design = build_reset_design(
        args.episodes,
        args.seed,
        needle_range,
        (args.grasp_start_deg, args.grasp_end_deg),
    )

    from RL.Approach_env import SRC_approach, NeedleResetValidityError
    from RL.utils.seed import seed_everything

    seed_everything(args.seed)
    threshold_cm = args.translation_threshold_mm / 10.0
    env = SRC_approach(
        seed=args.seed,
        reward_type="sparse",
        threshold=np.array(
            [threshold_cm, np.deg2rad(args.rotation_threshold_deg)],
            dtype=np.float32,
        ),
        max_episode_step=args.max_steps,
        step_size=STEP_SIZE_RAW,
        stepDR=False,
        command_integrated=False,
        command_state_clamp=True,
        measured_success_reward=True,
        needle_settle_steps=args.needle_settle_steps,
        needle_settle_interval_s=args.needle_settle_interval_s,
        synchronous_physics=args.strict_sync,
        physics_steps_per_action=args.physics_steps_per_action,
        physics_barrier_timeout_s=args.physics_barrier_timeout_s,
        randomize_psm_reset=True,
        randomize_needle_reset=True,
        freeze_live_goal=True,
        needle_random_range=needle_range,
        # Pose-close-only benchmark contract drops the measured-jaw grasp gate and
        # restores the scripted actuate attach on pose success.
        require_closed_jaw=not args.pose_close_only,
        jaw_success_source="measured",
        attach_on_pose_success=bool(args.pose_close_only),
        require_grasp_confirmation=args.require_grasp_confirmation,
        grasp_confirmation_steps=args.grasp_confirmation_steps,
        grasp_start_deg=args.grasp_start_deg,
        grasp_end_deg=args.grasp_end_deg,
        goal_source_audit=True,
    )
    expert = MeasuredApproachExpert(
        translation_threshold_cm=threshold_cm,
        rotation_threshold_deg=args.rotation_threshold_deg,
        pose_dwell_steps=args.pose_dwell_steps,
    )
    accepted_episodes = []
    episode_audit = []
    status = "failed"
    failure = None
    try:
        for target in design:
            accepted = None
            attempts = []
            for attempt in range(args.max_attempts_per_goal):
                attempt_seed = args.seed + target["episode"] * 1000 + attempt
                env.needle_reset_offset = np.asarray(
                    target["needle_offset_m_m_rad"], dtype=np.float32
                )
                env.config_grasp_angle_deg = float(target["grasp_angle_deg"])
                try:
                    observation, _ = env.reset(seed=attempt_seed)
                except NeedleResetValidityError as exc:
                    # A needle-reset hard failure just wastes this attempt; record
                    # it and move to the next attempt instead of crashing the run.
                    attempts.append({
                        "attempt": attempt,
                        "seed": attempt_seed,
                        "steps": 0,
                        "stop_reason": "reset_invalid",
                        "reset_invalid_reason": str(exc),
                        "final_errors": None,
                    })
                    print(
                        "COLLECT_RESET_INVALID:",
                        {"episode": target["episode"], "attempt": attempt, "reason": str(exc)},
                        flush=True,
                    )
                    continue
                expert.reset()
                trajectory = []
                stop_reason = "max_steps"
                for step_index in range(1, args.max_steps + 1):
                    measured = observation["achieved_goal"]
                    desired = observation["desired_goal"]
                    action, diagnostic = expert.action(measured, desired)
                    next_observation, reward, terminated, truncated, info = env.step(
                        action
                    )
                    errors = pose_close_errors(
                        next_observation["achieved_goal"],
                        next_observation["desired_goal"],
                    )
                    pose_close_success = bool(
                        errors["translation_cm"] <= threshold_cm
                        and errors["rotation_geodesic_deg"]
                        <= args.rotation_threshold_deg
                        and (
                            args.pose_close_only
                            or errors["measured_jaw"] <= args.jaw_threshold
                        )
                    )
                    grasp_status = getattr(env, "last_grasp_status", None)
                    grasp_confirmed = bool(
                        grasp_status and grasp_status.get("needle_grasped", False)
                    )
                    accepted_success = bool(
                        pose_close_success
                        and (
                            grasp_confirmed
                            if args.require_grasp_confirmation
                            else True
                        )
                    )
                    command_raw = np.asarray(
                        env.scene_manager.psm_goal_list[env.psm_idx - 1],
                        dtype=np.float32,
                    )
                    command_normalized = command_raw * np.array(
                        [100, 100, 100, 1, 1, 1, 1], dtype=np.float32
                    )
                    transition_info = dict(info or {})
                    transition_info.update(
                        {
                            "contract": "measured_pose_close_multigoal_v1",
                            "episode_index": int(target["episode"]),
                            "attempt": attempt,
                            "attempt_seed": attempt_seed,
                            "step_index": step_index,
                            "expert": diagnostic.__dict__,
                            "measured_errors": errors,
                            "command_pose_cm_rad_jaw": command_normalized.tolist(),
                            "measured_pose_cm_rad_jaw": next_observation[
                                "achieved_goal"
                            ].tolist(),
                            "desired_goal_cm_rad_jaw": next_observation[
                                "desired_goal"
                            ].tolist(),
                            "pose_close_success": pose_close_success,
                            "grasp_confirmed": grasp_confirmed,
                            "is_success": accepted_success,
                        }
                    )
                    trajectory.append(
                        {
                            "obs": copy_observation(observation),
                            "next_obs": copy_observation(next_observation),
                            "action": np.asarray(action, dtype=np.float32).copy(),
                            "reward": np.array(
                                [float(np.asarray(reward).reshape(-1)[0])],
                                dtype=np.float32,
                            ),
                            "done": np.array(
                                [float(accepted_success)], dtype=np.float32
                            ),
                            "info": transition_info,
                        }
                    )
                    observation = next_observation
                    if accepted_success:
                        stop_reason = "grasp_confirmed" if args.require_grasp_confirmation else "measured_pose_close"
                        accepted = trajectory
                        break
                    if terminated:
                        stop_reason = "environment_terminated_without_contract"
                        break
                    if truncated:
                        stop_reason = "environment_truncated"
                        break

                attempts.append(
                    {
                        "attempt": attempt,
                        "seed": attempt_seed,
                        "steps": len(trajectory),
                        "stop_reason": stop_reason,
                        "final_errors": (
                            trajectory[-1]["info"]["measured_errors"]
                            if trajectory
                            else None
                        ),
                    }
                )
                if accepted is not None:
                    break

            audit = {
                **target,
                "accepted": accepted is not None,
                "attempts": attempts,
                "reset_goal_audit": getattr(env, "reset_goal_audit", None),
            }
            episode_audit.append(audit)
            if accepted is None:
                failure = f"goal {target['episode']} failed all attempts"
                break
            accepted_episodes.append(accepted)
            atomic_pickle(
                episodes_dir / f"episode_{target['episode']:04d}.pkl", accepted
            )
            print(
                "ACCEPTED_DEMO",
                target["episode"],
                "steps=",
                len(accepted),
                "attempts=",
                len(attempts),
                flush=True,
            )

        if len(accepted_episodes) == args.episodes:
            merged = [item for episode in accepted_episodes for item in episode]
            dataset_path = output_dir / "all_episodes_merged.pkl"
            atomic_pickle(dataset_path, merged)
            summary = dataset_summary(accepted_episodes, STEP_SIZE_NORMALIZED)
            status = "complete"
            report = {
                "status": status,
                "contract": {
                    "observation": "measured PSM2 pose; xyz cm, RPY rad, measured jaw",
                    "desired_goal": "post-settle live needle goal frozen per episode",
                    "success": (
                        f"measured trans<={args.translation_threshold_mm}mm, "
                        f"SO(3)<={args.rotation_threshold_deg}deg, "
                        + (
                            "pose-close-only + scripted actuate attach "
                            "(measured-jaw gate dropped: jaws close on phantom, "
                            "not needle)"
                            if args.pose_close_only
                            else f"measured jaw<={args.jaw_threshold}"
                        )
                    ),
                    "physical_grasp_confirmation_required": bool(
                        args.require_grasp_confirmation
                    ),
                    "synthetic_attachment": False,
                },
                "sampling": {
                    "method": "latin_hypercube_4d",
                    "seed": args.seed,
                    "needle_random_range_m_m_rad": needle_range.tolist(),
                    "grasp_angle_range_deg": [
                        args.grasp_start_deg,
                        args.grasp_end_deg,
                    ],
                    "design": design,
                },
                "runtime": {
                    "strict_sync": bool(args.strict_sync),
                    "needle_settle_steps": args.needle_settle_steps,
                    "needle_settle_interval_s": args.needle_settle_interval_s,
                    "max_steps": args.max_steps,
                    "max_attempts_per_goal": args.max_attempts_per_goal,
                },
                "summary": summary,
                "episode_audit": episode_audit,
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": sha256_file(dataset_path),
                },
            }
            atomic_json(output_dir / "collection_report.json", report)
            atomic_json(
                output_dir / "FINAL_DATASET_STATUS.json",
                {
                    "status": "complete",
                    "episodes": args.episodes,
                    "transitions": len(merged),
                    "dataset": str(dataset_path),
                    "sha256": report["dataset"]["sha256"],
                    "integrator_exact_fraction_le_1e-5": summary[
                        "integrator_residual_norm"
                    ]["fraction_le_1e-5"],
                },
            )
            print(json.dumps(report["summary"], indent=2), flush=True)
            return 0

        atomic_json(
            output_dir / "collection_report.json",
            {
                "status": status,
                "failure": failure,
                "accepted_episodes": len(accepted_episodes),
                "requested_episodes": args.episodes,
                "sampling_design": design,
                "episode_audit": episode_audit,
            },
        )
        return 2
    finally:
        env.close()
        ral_instance = getattr(env, "ral_instance", None)
        if ral_instance is not None and hasattr(ral_instance, "shutdown"):
            ral_instance.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
