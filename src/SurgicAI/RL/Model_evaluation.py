import os
import argparse
import json
import zipfile
import numpy as np
import gymnasium as gym
import importlib
from RL.algorithm_configs_online import get_algorithm_config
import gc
import torch
import time
from pathlib import Path

from RL.controllers.grasp_servo import MeasuredGraspServo
from RL.Approach_env import NeedleResetValidityError
from RL.needle_reset_ranges import (
    ASSUMED_REAL_RZ_DEG,
    ASSUMED_REAL_X_MM,
    ASSUMED_REAL_Y_MM,
)
from RL.rl_paths import ExperimentKey, ensure_dir, experiment_dir
from RL.utils.cli_args import add_common_logging_args, add_experiment_variant_arg, add_threshold_args
from RL.utils.logging_utils import get_logger, setup_logging
from RL.utils.seed import seed_everything
from RL.utils.checkpoint_io import load_sb3_checkpoint, resolve_checkpoint_path
from RL.utils.utils import (
    convert_mat_to_frame,
    default_step_size,
    experiment_variant,
    frame_to_vector,
    resolve_src_env,
    rotation_geodesic_rad,
    threshold_from_args,
)

gc.collect()
torch.cuda.empty_cache()
logger = get_logger(__name__)


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def rpy_to_quaternion_xyzw(rpy):
    """Convert roll/pitch/yaw radians to a normalized xyzw quaternion."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quat = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float64)
    norm = np.linalg.norm(quat)
    return quat / norm if norm > 0.0 else np.array([0.0, 0.0, 0.0, 1.0])


def canonicalize_quaternion(quaternion, reference=None):
    """Choose a stable q/-q sign, optionally continuous with a reference q."""
    quat = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quat = quat / norm
    if reference is not None:
        if np.dot(quat, np.asarray(reference, dtype=np.float64)) < 0.0:
            quat = -quat
    elif quat[3] < 0.0:
        quat = -quat
    return quat


def quaternion_geodesic_deg(first, second):
    """Return the sign-invariant SO(3) geodesic angle between two xyzw quaternions."""
    first = canonicalize_quaternion(first)
    second = canonicalize_quaternion(second)
    dot = float(np.clip(abs(np.dot(first, second)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def normalized_pose_snapshot(env):
    """Read commanded and measured PSM state without changing environment state."""
    base_env = env.unwrapped
    scene_manager = base_env.scene_manager
    psm_index = int(base_env.psm_idx) - 1

    command_raw = np.asarray(scene_manager.psm_goal_list[psm_index], dtype=np.float64).copy()
    command_normalized = command_raw.copy()
    command_normalized[:3] *= 100.0
    command_quaternion = canonicalize_quaternion(
        rpy_to_quaternion_xyzw(command_normalized[3:6])
    )

    psm = scene_manager.psm_list[psm_index]
    measured_mat = psm.measured_cp()
    if measured_mat is None:
        return {
            "command_state_cm_rad_jaw": command_normalized.tolist(),
            "command_quaternion_xyzw": command_quaternion.tolist(),
            "measured_available": False,
            "measured_state_cm_rad_jaw": None,
            "measured_quaternion_xyzw": None,
        }

    measured_frame = convert_mat_to_frame(measured_mat)
    measured_jaw = (
        psm.measured_jaw_angle()
        if hasattr(psm, "measured_jaw_angle")
        else None
    )
    measured_raw = np.append(
        frame_to_vector(measured_frame),
        np.nan if measured_jaw is None else float(measured_jaw),
    ).astype(np.float64)
    measured_normalized = measured_raw.copy()
    measured_normalized[:3] *= 100.0
    measured_quaternion = canonicalize_quaternion(measured_frame.M.GetQuaternion())
    return {
        "command_state_cm_rad_jaw": command_normalized.tolist(),
        "command_quaternion_xyzw": command_quaternion.tolist(),
        "measured_available": True,
        "measured_state_cm_rad_jaw": measured_normalized.tolist(),
        "measured_quaternion_xyzw": measured_quaternion.tolist(),
        "measured_jaw_source": "RigidBodyState.joint_positions[6]",
        "measured_jaw_available": measured_jaw is not None,
    }


def measured_contract_status(env, measured_state):
    """Evaluate both requested measured-pose contracts over Approach multigoals."""
    if measured_state is None or not np.isfinite(measured_state[6]):
        return {
            "jaw_source": "RigidBodyState.joint_positions[6]",
            "jaw": None,
            "pose_1mm_rot10": False,
            "pose_1cm_rot10": False,
            "paper_1mm_rot10_jaw01": False,
            "eval_1cm_rot10_jaw01": False,
        }
    measured_state = np.asarray(measured_state, dtype=np.float64)
    goals = getattr(env.unwrapped, "multigoal_obs", [env.unwrapped.goal_obs])
    goal_scale = np.array([100, 100, 100, 1, 1, 1, 1], dtype=np.float64)
    errors = []
    for raw_goal in goals:
        goal = np.asarray(raw_goal, dtype=np.float64) * goal_scale
        errors.append({
            "translation_cm": float(np.linalg.norm(goal[:3] - measured_state[:3])),
            "rotation_geodesic_deg": quaternion_geodesic_deg(
                rpy_to_quaternion_xyzw(goal[3:6]),
                rpy_to_quaternion_xyzw(measured_state[3:6]),
            ),
        })
    jaw_ready = bool(measured_state[6] <= 0.1)
    pose_1mm = any(
        error["translation_cm"] <= 0.1
        and error["rotation_geodesic_deg"] <= 10.0
        for error in errors
    )
    pose_1cm = any(
        error["translation_cm"] <= 1.0
        and error["rotation_geodesic_deg"] <= 10.0
        for error in errors
    )
    paper = any(
        error["translation_cm"] <= 0.1
        and error["rotation_geodesic_deg"] <= 10.0
        and jaw_ready
        for error in errors
    )
    current_eval = any(
        error["translation_cm"] <= 1.0
        and error["rotation_geodesic_deg"] <= 10.0
        and jaw_ready
        for error in errors
    )
    return {
        "jaw_source": "RigidBodyState.joint_positions[6]",
        "jaw": float(measured_state[6]),
        "pose_1mm_rot10": bool(pose_1mm),
        "pose_1cm_rot10": bool(pose_1cm),
        "paper_1mm_rot10_jaw01": bool(paper),
        "eval_1cm_rot10_jaw01": bool(current_eval),
    }


def append_jsonl(path, payload):
    """Durably append one record so an interrupted unattended run stays auditable."""
    if path is None:
        return
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path, payload):
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, output_path)


def live_fp_goal_update(env, pose_file, episode, last_sequence, timeout_s=0.0):
    """Read one episode-matched FP pose and rebuild the single-goal observation."""
    if not pose_file:
        return None
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        try:
            payload = json.loads(Path(pose_file).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = None
        if (
            payload is not None
            and int(payload.get("episode", -1)) == int(episode)
            and int(payload.get("sequence", -1)) > int(last_sequence)
        ):
            from p9a_goal_geometry import needle_pose_to_goal

            pose = np.asarray(payload["fp_pose_world"], dtype=np.float64)
            base_env = env.unwrapped
            world_to_base = kdl_frame_to_matrix(
                base_env.scene_manager.psm_list[
                    base_env.psm_idx - 1
                ].get_T_w_b()
            )
            goal = needle_pose_to_goal(
                pose,
                world_to_base,
                base_env.grasp_angle,
                base_env.lift_height,
            ).astype(np.float32)
            base_env.fixed_episode_goal = goal.copy()
            base_env.goal_obs = goal.copy()
            base_env.multigoal_obs = [goal.copy()]
            if base_env.command_integrated:
                achieved = np.asarray(
                    base_env.scene_manager.psm_goal_list[
                        base_env.psm_idx - 1
                    ],
                    dtype=np.float32,
                )
            else:
                psm = base_env.scene_manager.psm_list[
                    base_env.psm_idx - 1
                ]
                measured_mat = psm.measured_cp()
                if measured_mat is None:
                    achieved = np.asarray(
                        base_env.scene_manager.psm_goal_list[
                            base_env.psm_idx - 1
                        ],
                        dtype=np.float32,
                    )
                else:
                    achieved = np.append(
                        frame_to_vector(convert_mat_to_frame(measured_mat)),
                        float(psm.get_jaw_angle()),
                    ).astype(np.float32)
            observation, _, _, _, _ = (
                base_env.gym_manager.update_observation(achieved)
            )
            age_ms = (
                (time.time_ns() - int(payload["capture_time_ns"])) / 1.0e6
                if payload.get("capture_time_ns") is not None
                else None
            )
            return observation, payload, age_ms
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.01)


def compact_incremental_episode_record(episode_record):
    """Keep the crash-safe stream small while preserving every R5 analysis field."""
    compact_keys = (
        "episode",
        "success",
        "steps",
        "termination_reason",
        "trajectory_length_mm",
        "final_achieved_goal",
        "final_desired_goal",
        "final_trans_error_cm",
        "final_criterion_angle_error_deg",
        "final_achieved_jaw",
        "measured_contract_pose_1mm_rot10_ever",
        "measured_contract_pose_1cm_rot10_ever",
        "measured_contract_paper_1mm_rot10_jaw01_ever",
        "measured_contract_eval_1cm_rot10_jaw01_ever",
        "command_clamp_event_count",
        "eval_seed",
        "train_seed",
        "freeze_live_goal",
        "model_path",
        "external_goal_bank",
        "external_goal_source",
        "external_goal_entry_index",
        "external_goal_reset_seed",
        "reset_pairing_translation_mm",
        "reset_pairing_rotation_deg",
        "reset_pairing_valid",
        "snapshot_restore_reached",
        "fp_translation_error_mm",
        "fp_rotation_error_deg",
        "fp_flip_gt_90_deg",
        "live_fp_updates",
        "live_fp_age_ms_p50",
        "live_fp_age_ms_p95",
        "live_fp_track_rotation_error_deg_p95",
        "live_fp_track_translation_error_mm_p95",
    )
    compact = {
        key: episode_record[key]
        for key in compact_keys
        if key in episode_record
    }
    goal_audit = episode_record.get("goal_source_audit") or {}
    settle = goal_audit.get("needle_settle") or {}
    compact.update({
        "reset_attempts_used": goal_audit.get("reset_attempts_used"),
        "needle_settle_translation_drift_cm": settle.get("translation_drift_cm"),
        "needle_settle_steps": settle.get("steps"),
        "needle_settle_timed_out": settle.get("timed_out"),
    })
    return compact


def load_external_goal_bank(path, source):
    """Load a P9a bank without changing the historical checkpoint-bank path."""
    bank_path = Path(path).expanduser().resolve()
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise ValueError(f"External goal bank is incomplete: {bank_path}")
    entries = payload.get("entries") or []
    if not entries:
        raise ValueError(f"External goal bank has no entries: {bank_path}")
    goal_key = f"{source}_goal_raw"
    goals = []
    for index, entry in enumerate(entries):
        if goal_key not in entry:
            raise ValueError(
                f"External goal bank entry {index} lacks {goal_key}"
            )
        goal = np.asarray(entry[goal_key], dtype=np.float32)
        if goal.shape != (7,) or not np.all(np.isfinite(goal)):
            raise ValueError(
                f"External goal bank entry {index} has invalid {goal_key}"
            )
        goals.append(goal)
    return bank_path, payload, entries, goals


def kdl_frame_to_matrix(frame):
    matrix = np.eye(4, dtype=np.float64)
    for row in range(3):
        for column in range(3):
            matrix[row, column] = frame.M[row, column]
        matrix[row, 3] = frame.p[row]
    return matrix


def hold_and_settle_wrist(
    env,
    timeout_s,
    tolerance_rad=1.0e-4,
    stable_s=3.0,
    poll_s=0.1,
):
    """Hold the commanded PSM pose until the measured jaw joints stop moving.

    P7f measurement: a constant normalized jaw command does NOT hold the AMBF
    gripper joints.  With the Approach default command frozen at 0.8 the
    measured jaw creeps monotonically from ~0.80 rad to the ~1.5707 rad joint
    limit over roughly fifty seconds and only then stays put.  A composite
    tracking mesh that welds the jaws onto the wrist is therefore not a rigid
    body during that creep.

    This helper holds the *already commanded* pose and jaw -- it never picks a
    new jaw angle -- and simply waits for the creep to finish before the policy
    starts.  Reward, observation, action and success criteria are untouched;
    only the episode's initial condition changes.
    """
    scene = env.unwrapped.scene_manager
    psm = scene.psm_list[env.unwrapped.psm_idx - 1]
    started = time.monotonic()
    needle_before = kdl_frame_to_matrix(scene.needle.get_pose())
    previous = None
    stable_since = None
    samples = 0
    settled = False
    while time.monotonic() - started < float(timeout_s):
        scene.psm_step(scene.psm_goal_list[0], 1)
        scene.psm_step(scene.psm_goal_list[1], 2)
        scene.world_manager.update()
        time.sleep(poll_s)
        samples += 1
        measured = psm.measured_jaw_angle()
        if measured is None:
            continue
        measured = float(measured)
        if previous is not None and abs(measured - previous) <= tolerance_rad:
            if stable_since is None:
                stable_since = time.monotonic()
            elif time.monotonic() - stable_since >= stable_s:
                settled = True
                previous = measured
                break
        else:
            stable_since = None
        previous = measured
    needle_after = kdl_frame_to_matrix(scene.needle.get_pose())
    return {
        "requested_timeout_s": float(timeout_s),
        "elapsed_s": float(time.monotonic() - started),
        "samples": int(samples),
        "settled": bool(settled),
        "tolerance_rad": float(tolerance_rad),
        "stable_s": float(stable_s),
        "final_measured_jaw_rad": None if previous is None else float(previous),
        "commanded_jaw_normalized": float(
            scene.psm_goal_list[env.unwrapped.psm_idx - 1][6]
        ),
        "needle_drift_mm": float(
            np.linalg.norm(needle_after[:3, 3] - needle_before[:3, 3]) * 1000.0
        ),
        "needle_drift_deg": float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (
                            np.trace(
                                needle_before[:3, :3].T @ needle_after[:3, :3]
                            )
                            - 1.0
                        )
                        / 2.0,
                        -1.0,
                        1.0,
                    )
                )
            )
        ),
    }


def external_reset_pairing_record(env, entry, index):
    expected = np.asarray(entry["expected_T_Wneedle"], dtype=np.float64)
    actual_frame = env.unwrapped.scene_manager.needle.get_pose()
    actual = kdl_frame_to_matrix(actual_frame)
    expected_vec = np.r_[
        expected[:3, 3],
        convert_mat_to_frame(expected).M.GetRPY(),
    ]
    actual_vec = frame_to_vector(actual_frame).astype(np.float64)
    trans_mm = float(
        np.linalg.norm(actual[:3, 3] - expected[:3, 3]) * 1000.0
    )
    rot_deg = float(
        np.degrees(rotation_geodesic_rad(expected_vec, actual_vec))
    )
    fp_metrics = entry.get("fp_metrics") or {}
    return {
        "external_goal_entry_index": int(index),
        "external_goal_reset_seed": int(entry["reset_seed"]),
        "reset_pairing_translation_mm": trans_mm,
        "reset_pairing_rotation_deg": rot_deg,
        "reset_pairing_valid": bool(
            trans_mm <= env.unwrapped.external_pairing_trans_tol_mm
            and rot_deg <= env.unwrapped.external_pairing_rot_tol_deg
        ),
        "reset_needle_pose_world": actual.tolist(),
        "expected_reset_needle_pose_world": expected.tolist(),
        "fp_translation_error_mm": fp_metrics.get(
            "fp_translation_error_mm"
        ),
        "fp_rotation_error_deg": fp_metrics.get("fp_rotation_error_deg"),
        "fp_flip_gt_90_deg": fp_metrics.get("fp_flip_gt_90_deg"),
    }


def restore_external_needle_snapshot(env, entry):
    """Restore and hold a previously settled compatible pose for paired P9a."""
    base_env = env.unwrapped
    expected = np.asarray(entry["expected_T_Wneedle"], dtype=np.float64)
    expected_frame = convert_mat_to_frame(expected)
    needle = base_env.scene_manager.needle
    reached = needle.set_pose(
        expected_frame,
        timeout=1.0,
        position_tolerance=2.5e-5,
        step_callback=base_env.scene_manager.step,
        release=False,
    )
    for _ in range(5):
        needle.hold_pose(expected_frame)
        base_env.scene_manager.step()
        time.sleep(0.02)
    base_env.external_locked_needle_frame = expected_frame
    return bool(reached)


def pose_error_record(goal, actual, goal_quaternion, actual_quaternion):
    if actual is None or actual_quaternion is None:
        return None
    goal = np.asarray(goal, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    return {
        "translation_cm": float(np.linalg.norm(goal[:3] - actual[:3])),
        "rotation_geodesic_deg": quaternion_geodesic_deg(goal_quaternion, actual_quaternion),
        "jaw": float(goal[6] - actual[6]),
    }


def load_checkpoint_demo_transitions(model_path):
    if not model_path:
        raise ValueError("checkpoint demo goal modes require --model-path")
    with zipfile.ZipFile(resolve_checkpoint_path(model_path)) as archive:
        saved = json.loads(archive.read("data"))
    demo_text = saved["replay_buffer_kwargs"]["demo_transitions"]
    return eval(
        demo_text,
        {"__builtins__": {}},
        {"array": np.array, "float32": np.float32},
    )


def load_checkpoint_goal_bank(model_path):
    transitions = load_checkpoint_demo_transitions(model_path)
    done = np.array([
        bool(np.asarray(transition["done"]).reshape(-1)[0])
        for transition in transitions
    ])
    starts = np.r_[0, np.flatnonzero(done)[:-1] + 1]
    normalized_goals = np.stack([
        np.asarray(transitions[index]["obs"]["desired_goal"], dtype=np.float32)
        for index in starts
    ])
    scale = np.array([100, 100, 100, 1, 1, 1, 1], dtype=np.float32)
    return normalized_goals / scale


def load_checkpoint_goal_random_goals(model_path, count, seed):
    if count <= 0:
        raise ValueError("--checkpoint-goal-random-count must be positive")
    transitions = load_checkpoint_demo_transitions(model_path)
    normalized_goals = np.stack([
        np.asarray(transition["obs"]["desired_goal"], dtype=np.float32)
        for transition in transitions
    ])
    low = np.quantile(normalized_goals, 0.05, axis=0).astype(np.float32)
    high = np.quantile(normalized_goals, 0.95, axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    sampled_normalized = rng.uniform(low, high, size=(count, normalized_goals.shape[1])).astype(np.float32)
    scale = np.array([100, 100, 100, 1, 1, 1, 1], dtype=np.float32)
    sampled_raw = sampled_normalized / scale
    print(
        "CHECKPOINT_RANDOM_GOAL_DIAG:",
        "count=", count,
        "seed=", seed,
        "desired_goal_p05_normalized=", np.round(low, 6).tolist(),
        "desired_goal_p95_normalized=", np.round(high, 6).tolist(),
        "first_goal_raw=", np.round(sampled_raw[0], 6).tolist(),
    )
    return sampled_raw


def load_checkpoint_goal_jitter_goals(model_path, count, seed, jitter_scale):
    if count <= 0:
        raise ValueError("--checkpoint-goal-jitter-count must be positive")
    if jitter_scale < 0:
        raise ValueError("--checkpoint-goal-jitter-scale must be non-negative")
    transitions = load_checkpoint_demo_transitions(model_path)
    done = np.array([
        bool(np.asarray(transition["done"]).reshape(-1)[0])
        for transition in transitions
    ])
    starts = np.r_[0, np.flatnonzero(done)[:-1] + 1]
    all_normalized_goals = np.stack([
        np.asarray(transition["obs"]["desired_goal"], dtype=np.float32)
        for transition in transitions
    ])
    reset_normalized_goals = np.stack([
        np.asarray(transitions[index]["obs"]["desired_goal"], dtype=np.float32)
        for index in starts
    ])
    low = np.quantile(all_normalized_goals, 0.05, axis=0).astype(np.float32)
    high = np.quantile(all_normalized_goals, 0.95, axis=0).astype(np.float32)
    q25 = np.quantile(reset_normalized_goals, 0.25, axis=0).astype(np.float32)
    q75 = np.quantile(reset_normalized_goals, 0.75, axis=0).astype(np.float32)
    noise_std = (q75 - q25) * np.float32(jitter_scale)
    rng = np.random.default_rng(seed)
    anchor_indices = rng.integers(0, len(reset_normalized_goals), size=count)
    sampled_normalized = reset_normalized_goals[anchor_indices].copy()
    if jitter_scale > 0:
        sampled_normalized += rng.normal(
            loc=0.0,
            scale=noise_std,
            size=sampled_normalized.shape,
        ).astype(np.float32)
    sampled_normalized = np.clip(sampled_normalized, low, high)
    scale = np.array([100, 100, 100, 1, 1, 1, 1], dtype=np.float32)
    sampled_raw = sampled_normalized / scale
    print(
        "CHECKPOINT_JITTER_GOAL_DIAG:",
        "count=", count,
        "seed=", seed,
        "jitter_scale=", jitter_scale,
        "anchor_indices=", anchor_indices[: min(10, len(anchor_indices))].tolist(),
        "noise_std_normalized=", np.round(noise_std, 6).tolist(),
        "desired_goal_p05_normalized=", np.round(low, 6).tolist(),
        "desired_goal_p95_normalized=", np.round(high, 6).tolist(),
        "first_goal_raw=", np.round(sampled_raw[0], 6).tolist(),
    )
    return sampled_raw


def checkpoint_goal_dim_index(dim):
    names = {
        "x": 0,
        "y": 1,
        "z": 2,
        "roll": 3,
        "pitch": 4,
        "yaw": 5,
        "jaw": 6,
    }
    key = str(dim).lower()
    if key in names:
        return names[key]
    try:
        idx = int(key)
    except ValueError as exc:
        raise ValueError(f"unknown goal dimension {dim!r}") from exc
    if idx < 0 or idx > 6:
        raise ValueError(f"goal dimension index must be in [0, 6], got {idx}")
    return idx


def load_checkpoint_goal_offset_sweep_goals(model_path, dim, offset_values):
    if not offset_values:
        raise ValueError("--checkpoint-goal-offset-values must include at least one value")
    transitions = load_checkpoint_demo_transitions(model_path)
    done = np.array([
        bool(np.asarray(transition["done"]).reshape(-1)[0])
        for transition in transitions
    ])
    starts = np.r_[0, np.flatnonzero(done)[:-1] + 1]
    reset_normalized_goals = np.stack([
        np.asarray(transitions[index]["obs"]["desired_goal"], dtype=np.float32)
        for index in starts
    ])
    scale = np.array([100, 100, 100, 1, 1, 1, 1], dtype=np.float32)
    base_raw = np.mean(reset_normalized_goals, axis=0).astype(np.float32) / scale
    idx = checkpoint_goal_dim_index(dim)
    goals = []
    for value in offset_values:
        goal = base_raw.copy()
        if idx < 3:
            goal[idx] += np.float32(value * 1.0e-3)
        elif idx < 6:
            goal[idx] += np.float32(np.deg2rad(value))
        else:
            goal[idx] += np.float32(value)
        goals.append(goal)
    goals = np.stack(goals).astype(np.float32)
    print(
        "CHECKPOINT_OFFSET_SWEEP_GOAL_DIAG:",
        "dim=", dim,
        "dim_index=", idx,
        "offset_values=", [float(value) for value in offset_values],
        "offset_units=", "mm" if idx < 3 else ("deg" if idx < 6 else "raw"),
        "base_goal_raw=", np.round(base_raw, 6).tolist(),
        "first_goal_raw=", np.round(goals[0], 6).tolist(),
    )
    return goals


def load_checkpoint_goal_audit_stats(model_path):
    transitions = load_checkpoint_demo_transitions(model_path)
    done = np.array([
        bool(np.asarray(transition["done"]).reshape(-1)[0])
        for transition in transitions
    ])
    starts = np.r_[0, np.flatnonzero(done)[:-1] + 1]
    all_desired = np.stack([
        np.asarray(transition["obs"]["desired_goal"], dtype=np.float32)
        for transition in transitions
    ])
    reset_desired = np.stack([
        np.asarray(transitions[index]["obs"]["desired_goal"], dtype=np.float32)
        for index in starts
    ])
    return {
        "desired_p05": np.quantile(all_desired, 0.05, axis=0).astype(np.float32),
        "desired_p95": np.quantile(all_desired, 0.95, axis=0).astype(np.float32),
        "desired_mean": np.mean(all_desired, axis=0).astype(np.float32),
        "reset_desired": reset_desired.astype(np.float32),
    }


def setup_environment(args, test_env):
    max_episode_steps = 1000
    trans_step_mm = args.trans_step_mm
    angle_step_deg = args.angle_step_deg
    if args.checkpoint_compat:
        trans_step_mm = 0.5 if trans_step_mm is None else trans_step_mm
        angle_step_deg = 2.0 if angle_step_deg is None else angle_step_deg
    else:
        trans_step_mm = 1.5 if trans_step_mm is None else trans_step_mm
        angle_step_deg = 3.0 if angle_step_deg is None else angle_step_deg
    step_size = default_step_size(
        trans_step=trans_step_mm * 1.0e-3,
        angle_step_deg=angle_step_deg,
        jaw_step=0.05,
    )
    threshold = threshold_from_args(args.trans_error, args.angle_error)
    SRC_class = resolve_src_env(args.task_name)
    
    if test_env == "stepDR_env":
        stepDR = True
    else:
        stepDR = False

    gym.envs.register(id=f"{args.algorithm}_{args.reward_type}", entry_point=SRC_class, max_episode_steps=max_episode_steps)
    env_kwargs = {
        "render_mode": "human",
        "reward_type": args.reward_type,
        "max_episode_step": max_episode_steps,
        "seed": args.eval_seed,
        "step_size": step_size,
        "threshold": threshold,
        "stepDR": stepDR,
    }
    if args.task_name.lower() == "approach":
        external_bank_path = None
        external_bank_payload = None
        external_bank_entries = None
        external_goal_bank = None
        external_modes = (
            args.checkpoint_goal_bank,
            args.checkpoint_goal_random,
            args.checkpoint_goal_jitter,
            args.checkpoint_goal_offset_sweep,
        )
        if args.external_goal_bank:
            if any(external_modes):
                raise ValueError(
                    "--external-goal-bank cannot be combined with checkpoint "
                    "goal-bank/random/jitter/offset modes"
                )
            (
                external_bank_path,
                external_bank_payload,
                external_bank_entries,
                external_goal_bank,
            ) = load_external_goal_bank(
                args.external_goal_bank,
                args.external_goal_source,
            )
        goal_bank = (
            external_goal_bank
            if external_goal_bank is not None
            else load_checkpoint_goal_bank(
                args.checkpoint_goal_bank_model_path or args.model_path
            )
            if args.checkpoint_goal_bank
            else None
        )
        random_goals = (
            load_checkpoint_goal_random_goals(
                args.model_path,
                args.checkpoint_goal_random_count,
                args.checkpoint_goal_random_seed,
            )
            if args.checkpoint_goal_random else None
        )
        jitter_goals = (
            load_checkpoint_goal_jitter_goals(
                args.model_path,
                args.checkpoint_goal_jitter_count,
                args.checkpoint_goal_jitter_seed,
                args.checkpoint_goal_jitter_scale,
            )
            if args.checkpoint_goal_jitter else None
        )
        offset_sweep_goals = (
            load_checkpoint_goal_offset_sweep_goals(
                args.model_path,
                args.checkpoint_goal_offset_dim,
                args.checkpoint_goal_offset_values,
            )
            if args.checkpoint_goal_offset_sweep else None
        )
        needle_random_range = np.array([
            args.needle_random_x_mm * 1.0e-3,
            args.needle_random_y_mm * 1.0e-3,
            np.deg2rad(args.needle_random_rz_deg),
        ], dtype=np.float32)
        psm_reset_random_range = np.array([
            args.psm_reset_noise_xyz_mm * 1.0e-3,
            args.psm_reset_noise_xyz_mm * 1.0e-3,
            args.psm_reset_noise_xyz_mm * 1.0e-3,
            np.deg2rad(args.psm_reset_noise_rpy_deg),
            np.deg2rad(args.psm_reset_noise_rpy_deg),
            np.deg2rad(args.psm_reset_noise_rpy_deg),
            args.psm_reset_noise_jaw,
        ], dtype=np.float32)
        psm2_reset_offset = np.array([
            args.psm2_reset_offset_xyz_mm[0] * 1.0e-3,
            args.psm2_reset_offset_xyz_mm[1] * 1.0e-3,
            args.psm2_reset_offset_xyz_mm[2] * 1.0e-3,
            np.deg2rad(args.psm2_reset_offset_rpy_deg[0]),
            np.deg2rad(args.psm2_reset_offset_rpy_deg[1]),
            np.deg2rad(args.psm2_reset_offset_rpy_deg[2]),
            args.psm2_reset_offset_jaw,
        ], dtype=np.float32)
        goal_offset = np.array([
            args.goal_offset_xyz_mm[0] * 1.0e-3,
            args.goal_offset_xyz_mm[1] * 1.0e-3,
            args.goal_offset_xyz_mm[2] * 1.0e-3,
            np.deg2rad(args.goal_offset_rpy_deg[0]),
            np.deg2rad(args.goal_offset_rpy_deg[1]),
            np.deg2rad(args.goal_offset_rpy_deg[2]),
            0.0,
        ], dtype=np.float32)
        env_kwargs.update({
            "checkpoint_compat": args.checkpoint_compat,
            "command_integrated": args.command_integrated,
            "command_state_clamp": args.command_state_clamp,
            "measured_success_reward": args.measured_success_reward,
            "needle_settle_steps": args.needle_settle_steps,
            "needle_settle_interval_s": args.needle_settle_interval_s,
            "synchronous_physics": args.synchronous_physics,
            "physics_steps_per_action": args.physics_steps_per_action,
            "physics_barrier_timeout_s": args.physics_barrier_timeout_s,
            "randomize_psm_reset": not args.fixed_psm_reset,
            "randomize_needle_reset": not args.fixed_needle_reset,
            "fixed_historical_goal": args.fixed_historical_goal,
            "raw_rpy_contract": args.raw_rpy_contract,
            "require_closed_jaw": args.require_closed_jaw,
            "require_grasp_confirmation": (
                args.require_grasp_confirmation or args.controller == "grasp-servo"
            ),
            "grasp_confirmation_steps": args.grasp_confirmation_steps,
            "checkpoint_goal_bank": goal_bank,
            "checkpoint_goal_random": args.checkpoint_goal_random,
            "checkpoint_goal_random_goals": random_goals,
            "checkpoint_goal_jitter": args.checkpoint_goal_jitter,
            "checkpoint_goal_jitter_goals": jitter_goals,
            "checkpoint_goal_offset_sweep": args.checkpoint_goal_offset_sweep,
            "checkpoint_goal_offset_goals": offset_sweep_goals,
            "freeze_live_goal": args.freeze_live_goal,
            "needle_random_range": needle_random_range,
            "psm_reset_random_range": psm_reset_random_range,
            "psm2_reset_offset": psm2_reset_offset,
            "goal_offset": goal_offset,
            "grasp_start_deg": args.grasp_start_deg,
            "grasp_end_deg": args.grasp_end_deg,
            "grasp_angle_deg": args.grasp_angle_deg,
            "lift_height": args.lift_height_mm * 1.0e-3,
            "goal_source_audit": args.goal_source_audit,
        })
        # Optional reset-gate threshold overrides (None -> keep env defaults).
        if args.needle_valid_so3_max_deg is not None:
            env_kwargs["needle_valid_so3_max_deg"] = args.needle_valid_so3_max_deg
        if args.needle_valid_xy_max_cm is not None:
            env_kwargs["needle_valid_xy_max_cm"] = args.needle_valid_xy_max_cm
        if args.needle_valid_z_tol_cm is not None:
            env_kwargs["needle_valid_z_tol_cm"] = args.needle_valid_z_tol_cm
        if args.needle_reset_validity_max_attempts is not None:
            env_kwargs["needle_reset_validity_max_attempts"] = args.needle_reset_validity_max_attempts
    env = gym.make(f"{args.algorithm}_{args.reward_type}", **env_kwargs)
    env.unwrapped.goal_transform_diag = args.goal_transform_diag
    env.unwrapped.wrap_obs_rpy_delta = args.wrap_obs_rpy_delta
    env.unwrapped.align_obs_rpy_branch = args.align_obs_rpy_branch
    if args.task_name.lower() == "approach":
        env.unwrapped.external_goal_bank_path = (
            str(external_bank_path) if external_bank_path is not None else None
        )
        env.unwrapped.external_goal_bank_payload = external_bank_payload
        env.unwrapped.external_goal_bank_entries = external_bank_entries or []
        env.unwrapped.external_goal_source = (
            args.external_goal_source if external_bank_entries else None
        )
        env.unwrapped.external_pairing_trans_tol_mm = (
            args.external_pairing_trans_tol_mm
        )
        env.unwrapped.external_pairing_rot_tol_deg = (
            args.external_pairing_rot_tol_deg
        )

    effective_env_max = int(getattr(env.unwrapped, "max_timestep", -1))
    effective_wrapper_max = int(getattr(env.spec, "max_episode_steps", -1))
    effective_closed_jaw = bool(getattr(env.unwrapped, "require_closed_jaw", False))
    effective_grasp_confirmation = bool(
        getattr(env.unwrapped, "require_grasp_confirmation", False)
    )
    print(
        "EVAL_CONTRACT_PREFLIGHT:",
        f"env_max_episode_steps={effective_env_max}",
        f"wrapper_max_episode_steps={effective_wrapper_max}",
        f"require_closed_jaw={effective_closed_jaw}",
        f"require_grasp_confirmation={effective_grasp_confirmation}",
        f"threshold_trans_cm={float(env.unwrapped.threshold_trans)}",
        f"threshold_angle_deg={float(np.rad2deg(env.unwrapped.threshold_angle))}",
        f"random_range={np.asarray(env.unwrapped.random_range, dtype=float).tolist()}",
    )
    if effective_env_max != max_episode_steps or effective_wrapper_max != max_episode_steps:
        raise RuntimeError(
            "Evaluation episode limit mismatch: "
            f"expected {max_episode_steps}, env={effective_env_max}, wrapper={effective_wrapper_max}"
        )
    if args.controller == "grasp-servo" and not effective_grasp_confirmation:
        raise RuntimeError("grasp-servo requires sensor-based grasp confirmation")
    if args.task_name.lower() == "approach" and args.goal_source_audit:
        env.unwrapped.checkpoint_goal_audit_stats = load_checkpoint_goal_audit_stats(args.model_path)
    return env, step_size, threshold, max_episode_steps

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate trained RL models.")
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the RL algorithm to evaluate')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='sparse', help='Reward type')
    add_threshold_args(parser)
    parser.add_argument('--eval_seed', type=int, default=42, help='Fixed seed for evaluation')
    # Backwards-compatible flags (kept), plus canonical --variant.
    parser.add_argument('--randomized', action='store_true', help='Model was trained with world randomization enabled')
    parser.add_argument('--stepDR', action='store_true', help='Model was trained with stepDR enabled')
    add_experiment_variant_arg(parser)
    parser.add_argument('--model-path', type=str, default=None, help='Explicit path to model (overrides derived path)')
    parser.add_argument('--train-seeds', type=int, nargs='*', default=None, help='Seeds to evaluate (default: a standard list)')
    parser.add_argument('--num-episodes', type=int, default=20, help='Evaluation episodes per seed')
    parser.add_argument(
        '--checkpoint-compat',
        action='store_true',
        help='Use the fixed historical Approach reset/goal and command-integrated transition contract.',
    )
    parser.add_argument(
        '--command-integrated',
        action='store_true',
        help='Observe the commanded Cartesian state after integrating each action, while keeping dynamic goals.',
    )
    parser.add_argument(
        '--command-state-clamp',
        action='store_true',
        help='Clamp command-integrated XYZ/jaw to physical limits and wrap RPY to [-pi, pi], recording every trigger.',
    )
    parser.add_argument(
        '--measured-success-reward',
        action='store_true',
        help='Use measured Cartesian pose, SO(3) error, and measured jaw feedback for online reward and success.',
    )
    parser.add_argument('--needle-settle-steps', type=int, default=60, help='Maximum reset settling samples (0.1 s each by default); acceptance requires held then released steady-state windows.')
    parser.add_argument('--needle-settle-interval-s', type=float, default=0.1, help='Wall-clock interval between reset settling cycles.')
    parser.add_argument('--synchronous-physics', action='store_true', help='Enable AMBF step throttling and wait for a fixed-step barrier after every command.')
    parser.add_argument('--physics-steps-per-action', type=int, default=10, help='Fixed AMBF physics steps released per environment action.')
    parser.add_argument('--physics-barrier-timeout-s', type=float, default=2.0, help='Timeout waiting for WorldState.sim_step barrier.')
    parser.add_argument('--deterministic-eval', action='store_true', help='Enable deterministic Torch algorithms and explicitly seed Gym spaces/environment.')
    parser.add_argument('--divergence-abort-cm', type=float, default=None, help='Diagnostic early abort when command/measured translation gap first exceeds this many centimeters.')
    parser.add_argument('--stall-abort-step', type=int, default=None, help='Diagnostic early abort after this step when the episode has not succeeded (step > value).')
    parser.add_argument('--fixed-psm-reset', action='store_true', help='Disable PSM reset randomization without fixing the goal.')
    parser.add_argument('--fixed-needle-reset', action='store_true', help='Disable needle reset randomization without fixing the goal.')
    parser.add_argument('--fixed-historical-goal', action='store_true', help='Use the expert dataset Approach goal without changing reset or dynamics.')
    parser.add_argument('--raw-rpy-contract', action='store_true', help='Use the historical unwrapped RPY distance and reward contract.')
    parser.add_argument('--wrap-obs-rpy-delta', action='store_true', help='Wrap the observation RPY delta (goal-achieved) to [-pi, pi] so measured_cp orientation is re-aligned onto the policy training branch.')
    parser.add_argument('--align-obs-rpy-branch', action='store_true', help='Snap the measured achieved RPY onto the desired goal 2*pi branch so both the absolute orientation block and the delta match training (supersedes --wrap-obs-rpy-delta).')
    parser.add_argument('--require-closed-jaw', action='store_true', help='Require jaw <= 0.1 for Approach success, matching the original environment.')
    parser.add_argument('--require-grasp-confirmation', action='store_true', help='Require PSM finger-sensor Needle attachment before Approach success; never unconditionally attach in criteria().')
    parser.add_argument('--grasp-confirmation-steps', type=int, default=3, help='Consecutive pose/jaw/Needle-grasp cycles required before success.')
    parser.add_argument('--grasp-approach-dwell-steps', type=int, default=3, help='Consecutive in-pose cycles before the grasp servo starts closing the jaw.')
    parser.add_argument('--grasp-open-jaw', type=float, default=0.8, help='Measured jaw target during the grasp-servo approach phase.')
    parser.add_argument('--trans-step-mm', type=float, default=None, help='Override Cartesian translation step in millimeters.')
    parser.add_argument('--angle-step-deg', type=float, default=None, help='Override RPY step in degrees.')
    parser.add_argument('--checkpoint-goal-bank', action='store_true', help='Cycle through reset goals embedded in the checkpoint demonstrations.')
    parser.add_argument('--checkpoint-goal-bank-model-path', type=str, default=None, help='Optional common checkpoint whose embedded demonstrations define --checkpoint-goal-bank. This permits exact paired policy evaluation without requiring both evaluated policies to embed the same demonstrations.')
    parser.add_argument('--checkpoint-goal-random', action='store_true', help='Cycle through new random goals sampled from checkpoint demo desired_goal 5%-95% ranges.')
    parser.add_argument('--checkpoint-goal-random-count', type=int, default=20, help='Number of checkpoint-distribution random goals to generate.')
    parser.add_argument('--checkpoint-goal-random-seed', type=int, default=42, help='Seed for checkpoint-distribution random goal generation.')
    parser.add_argument('--checkpoint-goal-jitter', action='store_true', help='Cycle through reset demo desired_goals with small jitter, clamped to checkpoint desired_goal 5%-95% ranges.')
    parser.add_argument('--checkpoint-goal-jitter-count', type=int, default=20, help='Number of jittered checkpoint reset goals to generate.')
    parser.add_argument('--checkpoint-goal-jitter-seed', type=int, default=42, help='Seed for jittered checkpoint reset goal generation.')
    parser.add_argument('--checkpoint-goal-jitter-scale', type=float, default=0.10, help='Gaussian jitter scale as a fraction of reset desired_goal IQR.')
    parser.add_argument('--checkpoint-goal-offset-sweep', action='store_true', help='Cycle through checkpoint reset-mean goals with a single-dimension offset for capability boundary tests.')
    parser.add_argument('--checkpoint-goal-offset-dim', type=str, default='x', help='Goal dimension to offset: x/y/z/roll/pitch/yaw/jaw or 0..6.')
    parser.add_argument('--checkpoint-goal-offset-values', type=float, nargs='*', default=[0.0], help='Offsets to apply. Units are mm for xyz, degrees for rpy, raw units for jaw.')
    parser.add_argument('--external-goal-bank', type=str, default=None, help='P9a JSON goal bank containing paired reset poses plus gt_goal_raw/fp_goal_raw.')
    parser.add_argument('--external-goal-source', choices=['gt', 'fp'], default='gt', help='Select gt_goal_raw or fp_goal_raw from --external-goal-bank.')
    parser.add_argument('--external-pairing-trans-tol-mm', type=float, default=1.0, help='Maximum replay-vs-bank needle translation mismatch for a valid paired episode.')
    parser.add_argument('--external-pairing-rot-tol-deg', type=float, default=0.5, help='Maximum replay-vs-bank needle SO(3) mismatch for a valid paired episode.')
    parser.add_argument('--no-external-snapshot-lock', action='store_true', help='Restore the paired reset snapshot, then release needle pose control for live FP-in-loop evaluation.')
    parser.add_argument('--live-fp-pose-file', type=str, default=None, help='Atomic JSON pose stream written by the P9b FoundationPose tracker.')
    parser.add_argument('--live-fp-control-file', type=str, default=None, help='Atomic episode-boundary request JSON consumed by the P9b camera/tracker.')
    parser.add_argument('--live-fp-initial-wait-s', type=float, default=30.0, help='Maximum wait for the episode-matched initial register result.')
    parser.add_argument('--freeze-jaw-command', action='store_true', help='P7f audit: zero the jaw action channel so the commanded jaw stays at its reset value for the whole episode. Pair with --psm-reset-noise-jaw 0 to hold one constant jaw command across the whole run.')
    parser.add_argument('--pre-episode-settle-s', type=float, default=0.0, help='P7f audit: before the policy starts, hold the commanded PSM pose/jaw and wait up to this many seconds for the measured jaw joints to stop creeping. Does not choose a new jaw angle and changes no reward/observation/success criterion.')
    parser.add_argument('--freeze-live-goal', dest='freeze_live_goal', action='store_true', default=True, help='Freeze the reset-time calibrated live desired_goal as a fixed episode goal (default: ON in evaluation for reproducibility).')
    parser.add_argument('--no-freeze-live-goal', dest='freeze_live_goal', action='store_false', help='Recompute the live desired_goal every step instead of freezing it at reset (disables the evaluation default).')
    parser.add_argument('--max-consecutive-reset-invalid', type=int, default=5, help='Abort the run only after this many back-to-back reset_invalid (needle-reset hard-fail) episodes; each is otherwise skipped and excluded from the success denominator.')
    # Optional overrides of the needle-reset validity gate (default None -> keep the
    # env-side conservative defaults; used e.g. to force hard-fails in validation).
    parser.add_argument('--needle-valid-so3-max-deg', type=float, default=None, help='Override reset-gate SO(3) max degrees (default: env default 60).')
    parser.add_argument('--needle-valid-xy-max-cm', type=float, default=None, help='Override reset-gate xy max cm (default: env default 5).')
    parser.add_argument('--needle-valid-z-tol-cm', type=float, default=None, help='Override reset-gate z tolerance cm (default: env default 1).')
    parser.add_argument('--needle-reset-validity-max-attempts', type=int, default=None, help='Override reset-gate max attempts before hard-fail (default: env default 8).')
    parser.add_argument('--goal-source-audit', action='store_true', help='Record live desired_goal source and compare it with checkpoint demo desired_goal distribution.')
    parser.add_argument('--needle-random-x-mm', type=float, default=ASSUMED_REAL_X_MM, help='Needle reset randomization x range in millimeters.')
    parser.add_argument('--needle-random-y-mm', type=float, default=ASSUMED_REAL_Y_MM, help='Needle reset randomization y range in millimeters.')
    parser.add_argument('--needle-random-rz-deg', type=float, default=ASSUMED_REAL_RZ_DEG, help='Needle reset randomization z-rotation range in degrees.')
    parser.add_argument('--psm-reset-noise-xyz-mm', type=float, default=5.0, help='PSM reset randomization xyz range in millimeters.')
    parser.add_argument('--psm-reset-noise-rpy-deg', type=float, default=float(np.rad2deg(0.5)), help='PSM reset randomization RPY range in degrees.')
    parser.add_argument('--psm-reset-noise-jaw', type=float, default=0.3, help='PSM reset randomization jaw range.')
    parser.add_argument('--psm2-reset-offset-xyz-mm', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Deterministic PSM2 reset xyz offset in millimeters.')
    parser.add_argument('--psm2-reset-offset-rpy-deg', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Deterministic PSM2 reset RPY offset in degrees.')
    parser.add_argument('--psm2-reset-offset-jaw', type=float, default=0.0, help='Deterministic PSM2 reset jaw offset.')
    parser.add_argument('--goal-offset-xyz-mm', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Diagnostic offset applied to live desired_goal xyz in millimeters.')
    parser.add_argument('--goal-offset-rpy-deg', type=float, nargs=3, default=[0.0, 0.0, 0.0], help='Diagnostic offset applied to live desired_goal RPY in degrees.')
    parser.add_argument('--grasp-start-deg', type=float, default=5.0, help='Start of live needle grasp-angle range in degrees.')
    parser.add_argument('--grasp-end-deg', type=float, default=20.0, help='End of live needle grasp-angle range in degrees.')
    parser.add_argument('--grasp-angle-deg', type=float, default=None, help='Fixed live needle grasp angle in degrees.')
    parser.add_argument('--lift-height-mm', type=float, default=7.0, help='Live desired_goal lift height in millimeters.')
    parser.add_argument('--goal-transform-diag', action='store_true', help='Print the needle/world/PSM goal transform chain.')
    parser.add_argument(
        '--controller',
        choices=['policy', 'goal-servo', 'grasp-servo', 'policy-residual'],
        default='policy',
        help='Use the checkpoint policy, pose-only goal servo, phased measured grasp servo, or checkpoint plus residual correction.',
    )
    parser.add_argument(
        '--residual-policy-weight',
        type=float,
        default=0.25,
        help='Policy contribution for --controller policy-residual.',
    )
    parser.add_argument(
        '--residual-servo-weight',
        type=float,
        default=1.0,
        help='Servo correction contribution for --controller policy-residual.',
    )
    parser.add_argument(
        '--disable-residual-direction-guard',
        action='store_true',
        help='Do not replace policy dimensions that move away from the current goal.',
    )
    parser.add_argument('--min-success-rate', type=float, default=None, help='Exit non-zero when the aggregate success rate is below this value.')
    parser.add_argument('--results-json', type=str, default=None, help='Optional path for a machine-readable aggregate result.')
    parser.add_argument('--episode-jsonl', type=str, default=None, help='Optional durable JSONL path appended after every completed or reset-invalid episode.')
    parser.add_argument(
        '--structured-step-trace',
        action='store_true',
        help='Embed a read-only per-step command/measured/goal pose trace in --results-json.',
    )
    add_common_logging_args(parser)
    return parser.parse_args()

def load_model(algorithm, env, task_name, reward_type, seed, randomized, stepDR, model_path: str | None, variant: str | None):
    randomization_str = experiment_variant(
        variant=variant,
        randomized=bool(randomized),
        stepDR=bool(stepDR),
    )

    if model_path is not None:
        resolved_model_path = resolve_checkpoint_path(model_path)
    else:
        # Default: reuse the same directory structure as training scripts.
        candidate_variants = [randomization_str]
        if randomization_str == "base_env":
            candidate_variants = ["base_env", "no_randomization"]

        resolved_model_path = None
        for variant in candidate_variants:
            candidate = experiment_dir(ExperimentKey(
                task_name=task_name,
                algorithm=algorithm,
                reward_type=reward_type,
                seed=seed,
                variant=variant,
            )) / "final_model"
            if candidate.exists() or candidate.with_suffix(".zip").exists():
                resolved_model_path = candidate
                break
        if resolved_model_path is None:
            # Last resort: return the derived path even if it doesn't exist,
            # so the error message is actionable.
            resolved_model_path = experiment_dir(ExperimentKey(
                task_name=task_name,
                algorithm=algorithm,
                reward_type=reward_type,
                seed=seed,
                variant=candidate_variants[0],
            )) / "final_model"

    algorithm_config = get_algorithm_config(algorithm, env, task_name, reward_type, seed, None, True)
    model_class = algorithm_config['class']
    return load_sb3_checkpoint(model_class, resolved_model_path, env=env)

def goal_servo_action(obs, step_size, raw_rpy_contract=False):
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    delta = desired - achieved
    delta_env = delta.copy()
    delta_env[:3] /= 100.0
    if not raw_rpy_contract:
        delta_env[3:6] = wrap_to_pi(delta_env[3:6])
    return np.clip(delta_env / np.asarray(step_size, dtype=np.float32), -1.0, 1.0)


def policy_residual_action(
    obs,
    policy_action,
    step_size,
    raw_rpy_contract=False,
    policy_weight=0.25,
    servo_weight=1.0,
    direction_guard=True,
):
    policy = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    servo = goal_servo_action(
        obs,
        step_size,
        raw_rpy_contract=raw_rpy_contract,
    ).astype(np.float32)

    guarded_policy = policy.copy()
    if direction_guard:
        achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
        desired = np.asarray(obs["desired_goal"], dtype=np.float32)
        delta = desired - achieved
        if not raw_rpy_contract:
            delta[3:6] = wrap_to_pi(delta[3:6])

        policy_step = policy * np.asarray(step_size, dtype=np.float32)
        # Observations store xyz in cm, while step_size xyz is in meters.
        policy_step_obs_units = policy_step.copy()
        policy_step_obs_units[:3] *= 100.0

        for dim in range(min(len(guarded_policy), len(delta))):
            if abs(float(delta[dim])) > 1e-6 and float(policy_step_obs_units[dim] * delta[dim]) <= 0.0:
                guarded_policy[dim] = servo[dim]

    action = policy_weight * guarded_policy + servo_weight * servo
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def goal_source_audit_record(env, desired_goal):
    stats = getattr(env.unwrapped, "checkpoint_goal_audit_stats", None)
    source = getattr(env.unwrapped, "reset_goal_audit", None)
    if stats is None:
        return source

    desired = np.asarray(desired_goal, dtype=np.float64)
    low = np.asarray(stats["desired_p05"], dtype=np.float64)
    high = np.asarray(stats["desired_p95"], dtype=np.float64)
    reset_desired = np.asarray(stats["reset_desired"], dtype=np.float64)
    inside_dims = (desired >= low) & (desired <= high)
    nearest_idx = int(np.argmin(np.linalg.norm(reset_desired[:, :6] - desired[:6], axis=1)))
    nearest = reset_desired[nearest_idx]
    audit = {
        "checkpoint_desired_p05": low.tolist(),
        "checkpoint_desired_p95": high.tolist(),
        "inside_checkpoint_5_95_by_dim": [bool(value) for value in inside_dims],
        "inside_checkpoint_5_95_all": bool(np.all(inside_dims)),
        "nearest_reset_goal_index": nearest_idx,
        "nearest_reset_goal_l2_first6": float(np.linalg.norm(nearest[:6] - desired[:6])),
        "nearest_reset_goal_delta": (desired - nearest).tolist(),
    }
    if source is not None:
        audit["source"] = source
    return audit


def run_evaluation(
    env,
    model,
    num_episodes,
    max_episode_steps,
    controller="policy",
    residual_policy_weight=0.25,
    residual_servo_weight=1.0,
    residual_direction_guard=True,
    structured_step_trace=False,
    divergence_abort_cm=None,
    stall_abort_step=None,
    grasp_approach_dwell_steps=3,
    grasp_confirmation_steps=3,
    grasp_open_jaw=0.8,
    max_consecutive_reset_invalid=5,
    episode_jsonl_path=None,
    episode_record_context=None,
    external_snapshot_lock=True,
    live_fp_pose_file=None,
    live_fp_control_file=None,
    live_fp_initial_wait_s=30.0,
    freeze_jaw_command=False,
    pre_episode_settle_s=0.0,
):
    total_length = 0
    total_timecost = 0
    total_success = 0
    all_lengths = []
    all_timecosts = []
    episode_records = []
    reset_invalid_records = []
    consecutive_reset_invalid = 0
    external_entries = list(
        getattr(env.unwrapped, "external_goal_bank_entries", [])
    )
    external_valid_index = 0

    # --- C. checkpoint / model / space diagnostics (once) ---
    print("MODEL_CLASS_DIAG:", type(model))
    print("MODEL_POLICY_DIAG:", type(getattr(model, "policy", None)))
    print("MODEL_OBS_SPACE_DIAG:", getattr(model, "observation_space", None))
    print("MODEL_ACTION_SPACE_DIAG:", getattr(model, "action_space", None))
    print("ENV_OBS_SPACE_DIAG:", env.observation_space)
    print("ENV_ACTION_SPACE_DIAG:", env.action_space)
    print("MODEL_N_ENV_DIAG:", getattr(model, "n_envs", None))

    # --- D/E. accumulate eval obs distribution over first steps ---
    eval_obs_buf = {"ach": [], "des": [], "obs": [], "act": []}

    for episode in range(num_episodes):
        external_entry = None
        external_entry_index = None
        reset_seed = None
        if external_entries:
            external_entry_index = external_valid_index % len(external_entries)
            external_entry = external_entries[external_entry_index]
            reset_seed = int(external_entry["reset_seed"])
        try:
            obs, _ = env.reset(seed=reset_seed)
        except NeedleResetValidityError as exc:
            # A needle-reset hard failure must not crash the whole evaluation.
            # Record the episode as reset_invalid, keep it OUT of the success
            # denominator, and continue. If the simulator is truly dead, resets
            # will fail back-to-back, so abort only after too many consecutive
            # invalid resets.
            consecutive_reset_invalid += 1
            reason = str(exc)
            reset_invalid_record = {
                "episode": int(episode),
                "status": "reset_invalid",
                "reason": reason,
                "consecutive": int(consecutive_reset_invalid),
            }
            if episode_record_context:
                reset_invalid_record.update(episode_record_context)
            reset_invalid_records.append(reset_invalid_record)
            append_jsonl(episode_jsonl_path, reset_invalid_record)
            if external_entry is not None:
                raise RuntimeError(
                    "External paired reset became invalid for bank entry "
                    f"{external_entry_index} seed={reset_seed}: {reason}"
                ) from exc
            print(
                "EPISODE_RESET_INVALID:",
                {"episode": episode, "consecutive": consecutive_reset_invalid,
                 "limit": max_consecutive_reset_invalid, "reason": reason},
                flush=True,
            )
            if consecutive_reset_invalid > max_consecutive_reset_invalid:
                raise RuntimeError(
                    f"Aborting evaluation: {consecutive_reset_invalid} consecutive "
                    f"reset_invalid episodes exceed the limit "
                    f"({max_consecutive_reset_invalid}); the simulator may be dead."
                )
            continue
        consecutive_reset_invalid = 0
        grasp_servo = None
        if controller == "grasp-servo":
            grasp_servo = MeasuredGraspServo(
                env.unwrapped.step_size,
                translation_threshold_cm=float(env.unwrapped.threshold_trans),
                rotation_threshold_rad=float(env.unwrapped.threshold_angle),
                approach_dwell_steps=grasp_approach_dwell_steps,
                confirmation_steps=grasp_confirmation_steps,
                open_jaw=grasp_open_jaw,
            )
        reset_obs = {k: np.array(v, dtype=np.float64, copy=True) for k, v in obs.items()}
        reset_goal_source_audit = goal_source_audit_record(env, reset_obs["desired_goal"])
        external_pairing = None
        if external_entry is not None:
            snapshot_restore_reached = restore_external_needle_snapshot(
                env, external_entry
            )
            external_pairing = external_reset_pairing_record(
                env,
                external_entry,
                external_entry_index,
            )
            external_pairing["snapshot_restore_reached"] = (
                snapshot_restore_reached
            )
            if not external_pairing["reset_pairing_valid"]:
                raise RuntimeError(
                    "External goal bank replay mismatch for entry "
                    f"{external_entry_index}: "
                    f"{external_pairing['reset_pairing_translation_mm']:.6f} mm, "
                    f"{external_pairing['reset_pairing_rotation_deg']:.6f} deg"
                )
            if not external_snapshot_lock:
                env.unwrapped.scene_manager.needle.release_pose_control(
                    step_callback=env.unwrapped.scene_manager.step
                )
                env.unwrapped.external_locked_needle_frame = None
            external_valid_index += 1
        wrist_settle_record = None
        if pre_episode_settle_s and pre_episode_settle_s > 0.0:
            wrist_settle_record = hold_and_settle_wrist(
                env, pre_episode_settle_s
            )
            print(
                "PRE_EPISODE_WRIST_SETTLE:",
                json.dumps(wrist_settle_record),
                flush=True,
            )
        live_fp_records = []
        live_fp_sequence = -1
        if live_fp_pose_file and not live_fp_control_file:
            raise ValueError(
                "--live-fp-pose-file requires --live-fp-control-file"
            )
        if live_fp_control_file:
            # The control file is the episode-boundary handshake consumed by an
            # external capture process.  P7f records RGB-D offline and does not
            # run FoundationPose in the loop, so the write must not be gated on
            # --live-fp-pose-file.  When a pose file IS supplied the behaviour
            # below is byte-identical to the pre-P7f control flow.
            atomic_json(
                live_fp_control_file,
                {
                    "episode": int(episode),
                    "reset_seed": reset_seed,
                    "request_time_ns": time.time_ns(),
                    "expected_T_Wneedle": (
                        external_entry["expected_T_Wneedle"]
                        if external_entry is not None else None
                    ),
                    "commanded_jaw": float(
                        env.unwrapped.scene_manager.psm_goal_list[
                            env.unwrapped.psm_idx - 1
                        ][6]
                    ),
                    "freeze_jaw_command": bool(freeze_jaw_command),
                    "wrist_settle": wrist_settle_record,
                },
            )
        if live_fp_pose_file:
            live_update = live_fp_goal_update(
                env,
                live_fp_pose_file,
                episode,
                live_fp_sequence,
                timeout_s=live_fp_initial_wait_s,
            )
            if live_update is None:
                raise RuntimeError(
                    f"Timed out waiting for FP register for episode {episode}"
                )
            obs, live_payload, live_age_ms = live_update
            live_fp_sequence = int(live_payload["sequence"])
            live_fp_records.append(
                {**live_payload, "goal_age_ms": live_age_ms}
            )
        fixed_episode_goal = getattr(env.unwrapped, "fixed_episode_goal", None)
        raw_episode_goal = (
            np.array(fixed_episode_goal, dtype=np.float64).tolist()
            if fixed_episode_goal is not None else None
        )
        trajectory_length = 0
        episode_success = False
        episode_steps = max_episode_steps
        termination_reason = "max_steps_exhausted"
        min_achieved_jaw = float(reset_obs["achieved_goal"][6])
        min_action_jaw = float("inf")
        closed_jaw_observed = min_achieved_jaw <= 0.1
        first_step_goal_delta = None
        max_goal_jump_trans_cm = 0.0
        max_goal_jump_angle_deg = 0.0
        max_goal_jump_step = None
        max_goal_jump_before = None
        max_goal_jump_after = None
        max_goal_rotation_geodesic_deg = 0.0
        max_goal_rotation_geodesic_step = None
        step_trace = []
        episode_clamp_events = []
        episode_measured_contract_paper = False
        episode_measured_contract_eval = False
        episode_measured_contract_pose_1mm = False
        episode_measured_contract_pose_1cm = False
        reset_pose_snapshot = normalized_pose_snapshot(env) if structured_step_trace else None
        previous_desired = np.asarray(obs["desired_goal"], dtype=np.float64).copy()
        servo_phase_history = []
        for timestep in range(max_episode_steps):
            live_update = live_fp_goal_update(
                env,
                live_fp_pose_file,
                episode,
                live_fp_sequence,
            )
            if live_update is not None:
                obs, live_payload, live_age_ms = live_update
                live_fp_sequence = int(live_payload["sequence"])
                live_fp_records.append(
                    {**live_payload, "goal_age_ms": live_age_ms}
                )
            servo_diagnostic = None
            if controller == "grasp-servo":
                measured = env.unwrapped.measured_achieved_goal()
                if measured is None:
                    raise RuntimeError("grasp-servo requires measured PSM pose and jaw")
                psm = env.unwrapped.scene_manager.psm_list[env.unwrapped.psm_idx - 1]
                grasp_status_before = psm.grasp_status()
                action, servo_diagnostic = grasp_servo.action(
                    measured,
                    obs["desired_goal"],
                    grasp_status_before,
                )
                servo_phase_history.append(dict(servo_diagnostic))
            elif controller == "goal-servo":
                action = goal_servo_action(
                    obs,
                    env.unwrapped.step_size,
                    raw_rpy_contract=bool(getattr(env.unwrapped, "raw_rpy_contract", False)),
                )
            elif controller == "policy-residual":
                policy_action, _ = model.predict(obs, deterministic=True)
                action = policy_residual_action(
                    obs,
                    policy_action,
                    env.unwrapped.step_size,
                    raw_rpy_contract=bool(getattr(env.unwrapped, "raw_rpy_contract", False)),
                    policy_weight=residual_policy_weight,
                    servo_weight=residual_servo_weight,
                    direction_guard=residual_direction_guard,
                )
            else:
                action, _ = model.predict(obs, deterministic=True)

            action = np.asarray(action, dtype=np.float32)
            if freeze_jaw_command:
                # P7f audit contract: the composite tracking mesh is only a
                # rigid body if the jaw opening command never changes.  Zero
                # the jaw action channel so the commanded jaw stays at its
                # reset value for the whole episode.  Nothing else about the
                # policy, reward, observation or success criterion changes.
                action[6] = 0.0
            min_action_jaw = min(min_action_jaw, float(action[6]))

            # accumulate distribution stats (policy input is normalized dict)
            if len(eval_obs_buf["obs"]) < 200:
                eval_obs_buf["ach"].append(np.array(obs["achieved_goal"], dtype=np.float64))
                eval_obs_buf["des"].append(np.array(obs["desired_goal"], dtype=np.float64))
                eval_obs_buf["obs"].append(np.array(obs["observation"], dtype=np.float64))
                eval_obs_buf["act"].append(np.array(action, dtype=np.float64))

            if timestep < 5 or timestep % 20 == 0:
                ach = np.array(obs["achieved_goal"], dtype=np.float64)
                des = np.array(obs["desired_goal"], dtype=np.float64)
                obs_vec = np.array(obs["observation"], dtype=np.float64)

                raw_rpy_delta = des[3:6] - ach[3:6]
                wrapped_rpy_delta = wrap_to_pi(raw_rpy_delta)

                print(
                    "MODEL_ACTION_DIAG:",
                    "episode=", episode,
                    "t=", timestep,
                    "action=", action,
                    "obs_achieved=", obs["achieved_goal"],
                    "obs_desired=", obs["desired_goal"],
                    "obs_delta=", obs["desired_goal"] - obs["achieved_goal"],
                )

                # --- A. obs RPY input diagnosis ---
                print(
                    "OBS_RPY_INPUT_DIAG:",
                    "episode=", episode,
                    "t=", timestep,
                    "achieved_rpy=", ach[3:6],
                    "desired_rpy=", des[3:6],
                    "obs_delta_slice_17_20=", obs_vec[17:20],
                    "raw_delta_deg=", np.degrees(raw_rpy_delta),
                    "wrapped_delta_deg=", np.degrees(wrapped_rpy_delta),
                    "raw_l2_deg=", np.degrees(np.linalg.norm(raw_rpy_delta)),
                    "wrapped_l2_deg=", np.degrees(np.linalg.norm(wrapped_rpy_delta)),
                )

                # --- B. offline obs-variant action comparison (NOT executed) ---
                action_raw = action
                obs_wrap_delta = {k: np.array(v, copy=True) for k, v in obs.items()}
                obs_wrap_delta["observation"] = np.array(obs["observation"], copy=True)
                obs_wrap_delta["observation"][17:20] = wrapped_rpy_delta
                action_wrap_delta, _ = model.predict(obs_wrap_delta, deterministic=True)
                print(
                    "POLICY_OBS_VARIANT_DIAG:",
                    "episode=", episode,
                    "t=", timestep,
                    "action_raw=", action_raw,
                    "action_wrap_delta=", action_wrap_delta,
                    "action_diff=", np.array(action_wrap_delta) - np.array(action_raw),
                )

            external_locked_frame = getattr(
                env.unwrapped, "external_locked_needle_frame", None
            )
            if external_locked_frame is not None:
                env.unwrapped.scene_manager.needle.hold_pose(
                    external_locked_frame
                )
            next_obs, reward, terminated, truncated, info = env.unwrapped.step(action)
            measured_contract = measured_contract_status(
                env,
                env.unwrapped.measured_achieved_goal(),
            )
            episode_measured_contract_pose_1mm |= measured_contract["pose_1mm_rot10"]
            episode_measured_contract_pose_1cm |= measured_contract["pose_1cm_rot10"]
            episode_measured_contract_paper |= measured_contract["paper_1mm_rot10_jaw01"]
            episode_measured_contract_eval |= measured_contract["eval_1cm_rot10_jaw01"]
            step_clamp_events = [
                {**event, "episode": int(episode)}
                for event in getattr(env.unwrapped, "last_command_clamp_events", [])
            ]
            episode_clamp_events.extend(step_clamp_events)
            trajectory_length += np.linalg.norm(action[0:3] * env.unwrapped.step_size[0:3] * 1000)
            next_desired = np.asarray(next_obs["desired_goal"], dtype=np.float64)
            desired_delta = next_desired - previous_desired
            goal_jump_trans_cm = float(np.linalg.norm(desired_delta[:3]))
            goal_jump_angle_deg = float(np.degrees(np.linalg.norm(wrap_to_pi(desired_delta[3:6]))))
            desired_quaternion_before = canonicalize_quaternion(
                rpy_to_quaternion_xyzw(previous_desired[3:6])
            )
            desired_quaternion_after = canonicalize_quaternion(
                rpy_to_quaternion_xyzw(next_desired[3:6]),
                reference=desired_quaternion_before,
            )
            goal_rotation_geodesic_deg = quaternion_geodesic_deg(
                desired_quaternion_before,
                desired_quaternion_after,
            )
            max_goal_jump_trans_cm = max(max_goal_jump_trans_cm, goal_jump_trans_cm)
            if goal_jump_angle_deg > max_goal_jump_angle_deg:
                max_goal_jump_angle_deg = goal_jump_angle_deg
                max_goal_jump_step = timestep + 1
                max_goal_jump_before = previous_desired.tolist()
                max_goal_jump_after = next_desired.tolist()
            if goal_rotation_geodesic_deg > max_goal_rotation_geodesic_deg:
                max_goal_rotation_geodesic_deg = goal_rotation_geodesic_deg
                max_goal_rotation_geodesic_step = timestep + 1
            if timestep == 0:
                first_step_goal_delta = desired_delta.tolist()

            abort_reason = None
            pose_snapshot = None
            command_measured_error = None
            if structured_step_trace or divergence_abort_cm is not None:
                pose_snapshot = normalized_pose_snapshot(env)
                command_state = pose_snapshot["command_state_cm_rad_jaw"]
                command_quaternion = pose_snapshot["command_quaternion_xyzw"]
                measured_state = pose_snapshot["measured_state_cm_rad_jaw"]
                measured_quaternion = pose_snapshot["measured_quaternion_xyzw"]
                command_measured_error = pose_error_record(
                    command_state,
                    measured_state,
                    command_quaternion,
                    measured_quaternion,
                )
            success_now = bool(terminated and info.get("is_success", False))
            if not success_now:
                if (
                    divergence_abort_cm is not None
                    and command_measured_error is not None
                    and command_measured_error["translation_cm"] > divergence_abort_cm
                ):
                    abort_reason = "diverged_abort"
                elif stall_abort_step is not None and (timestep + 1) > stall_abort_step:
                    abort_reason = "stall_abort"

            if structured_step_trace:
                observed_achieved = np.asarray(next_obs["achieved_goal"], dtype=np.float64)
                observed_quaternion = canonicalize_quaternion(
                    rpy_to_quaternion_xyzw(observed_achieved[3:6])
                )
                step_trace.append({
                    "step": int(timestep + 1),
                    "action": action.astype(np.float64).tolist(),
                    "desired_before_cm_rad_jaw": previous_desired.tolist(),
                    "desired_after_cm_rad_jaw": next_desired.tolist(),
                    "desired_delta_xyz_cm": desired_delta[:3].tolist(),
                    "desired_delta_rpy_raw_deg": np.degrees(desired_delta[3:6]).tolist(),
                    "desired_delta_rpy_wrapped_deg": np.degrees(
                        wrap_to_pi(desired_delta[3:6])
                    ).tolist(),
                    "desired_euler_jump_l2_deg": goal_jump_angle_deg,
                    "desired_quaternion_before_xyzw": desired_quaternion_before.tolist(),
                    "desired_quaternion_after_xyzw_sign_continuous": desired_quaternion_after.tolist(),
                    "desired_rotation_geodesic_deg": goal_rotation_geodesic_deg,
                    "observed_achieved_after_cm_rad_jaw": observed_achieved.tolist(),
                    "observed_achieved_quaternion_xyzw": observed_quaternion.tolist(),
                    **pose_snapshot,
                    "desired_vs_observed_achieved": pose_error_record(
                        next_desired,
                        observed_achieved,
                        desired_quaternion_after,
                        observed_quaternion,
                    ),
                    "desired_vs_command": pose_error_record(
                        next_desired,
                        command_state,
                        desired_quaternion_after,
                        command_quaternion,
                    ),
                    "desired_vs_measured": pose_error_record(
                        next_desired,
                        measured_state,
                        desired_quaternion_after,
                        measured_quaternion,
                    ),
                    "command_vs_measured": command_measured_error,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "is_success": bool(info.get("is_success", False)),
                    "termination_reason": (
                        abort_reason
                        if abort_reason is not None
                        else "success"
                        if success_now
                        else "truncation"
                        if truncated
                        else "terminated_other"
                        if terminated
                        else "continuing"
                    ),
                    "command_clamp_events": step_clamp_events,
                    "measured_contracts": measured_contract,
                    "physics_step_barrier": getattr(
                        env.unwrapped.scene_manager.world_manager,
                        "last_step_barrier",
                        None,
                    ),
                    "grasp_servo": servo_diagnostic,
                    "grasp_status": getattr(env.unwrapped, "last_grasp_status", None),
                })
            previous_desired = next_desired.copy()

            achieved_jaw = float(next_obs["achieved_goal"][6])
            min_achieved_jaw = min(min_achieved_jaw, achieved_jaw)
            closed_jaw_observed = closed_jaw_observed or achieved_jaw <= 0.1
            obs = next_obs
            if abort_reason is not None:
                episode_steps = timestep + 1
                termination_reason = abort_reason
                break
            if terminated or truncated:
                episode_steps = timestep + 1
                if terminated and info.get("is_success", False):
                    total_success += 1
                    total_length += trajectory_length
                    total_timecost += timestep + 1
                    all_lengths.append(trajectory_length)
                    all_timecosts.append(timestep + 1)
                    episode_success = True
                    termination_reason = (
                        "grasp_confirmed" if controller == "grasp-servo" else "success"
                    )
                elif truncated:
                    termination_reason = "truncation"
                else:
                    termination_reason = "terminated_other"
                break

        final_achieved = np.array(obs["achieved_goal"], dtype=np.float64)
        final_desired = np.array(obs["desired_goal"], dtype=np.float64)
        final_raw_rpy_delta = final_desired[3:6] - final_achieved[3:6]
        final_wrapped_rpy_delta = wrap_to_pi(final_raw_rpy_delta)
        criterion_angle_error_deg = float(np.degrees(np.linalg.norm(
            final_raw_rpy_delta
            if bool(getattr(env.unwrapped, "raw_rpy_contract", False))
            else final_wrapped_rpy_delta
        )))
        episode_record = {
            "episode": episode,
            "success": bool(episode_success),
            "steps": int(episode_steps),
            "termination_reason": termination_reason,
            "trajectory_length_mm": float(trajectory_length),
            "raw_goal": raw_episode_goal,
            "reset_achieved_goal": reset_obs["achieved_goal"].tolist(),
            "reset_desired_goal": reset_obs["desired_goal"].tolist(),
            "reset_delta": (reset_obs["desired_goal"] - reset_obs["achieved_goal"]).tolist(),
            "goal_source_audit": reset_goal_source_audit,
            "final_achieved_goal": final_achieved.tolist(),
            "final_desired_goal": final_desired.tolist(),
            "final_trans_error_cm": float(np.linalg.norm(final_desired[:3] - final_achieved[:3])),
            "final_raw_rpy_l2_deg": float(np.degrees(np.linalg.norm(final_raw_rpy_delta))),
            "final_wrapped_rpy_l2_deg": float(np.degrees(np.linalg.norm(final_wrapped_rpy_delta))),
            "final_criterion_angle_error_deg": criterion_angle_error_deg,
            "final_jaw_error": float(final_desired[6] - final_achieved[6]),
            "final_achieved_jaw": float(final_achieved[6]),
            "min_achieved_jaw": float(min_achieved_jaw),
            "min_action_jaw": None if np.isinf(min_action_jaw) else float(min_action_jaw),
            "jaw_close_action_issued": bool(not np.isinf(min_action_jaw) and min_action_jaw < 0.0),
            "closed_jaw_observed": bool(closed_jaw_observed),
            "first_step_desired_goal_delta": first_step_goal_delta,
            "max_goal_jump_trans_cm": float(max_goal_jump_trans_cm),
            "max_goal_jump_angle_deg": float(max_goal_jump_angle_deg),
            "max_goal_jump_step": max_goal_jump_step,
            "max_goal_jump_before_cm_rad_jaw": max_goal_jump_before,
            "max_goal_jump_after_cm_rad_jaw": max_goal_jump_after,
            "max_goal_rotation_geodesic_deg": float(max_goal_rotation_geodesic_deg),
            "max_goal_rotation_geodesic_step": max_goal_rotation_geodesic_step,
            "command_clamp_event_count": len(episode_clamp_events),
            "command_clamp_events": episode_clamp_events,
            "grasp_servo_final_phase": (
                grasp_servo.phase if grasp_servo is not None else None
            ),
            "grasp_servo_phase_history": servo_phase_history,
            "final_grasp_status": getattr(env.unwrapped, "last_grasp_status", None),
            "measured_contract_pose_1mm_rot10_ever": bool(
                episode_measured_contract_pose_1mm
            ),
            "measured_contract_pose_1cm_rot10_ever": bool(
                episode_measured_contract_pose_1cm
            ),
            "measured_contract_paper_1mm_rot10_jaw01_ever": bool(episode_measured_contract_paper),
            "measured_contract_eval_1cm_rot10_jaw01_ever": bool(episode_measured_contract_eval),
        }
        if live_fp_pose_file:
            ages = [
                row["goal_age_ms"]
                for row in live_fp_records
                if row.get("goal_age_ms") is not None
            ]
            rotations = [
                row["fp_rotation_error_deg"]
                for row in live_fp_records
                if row.get("fp_rotation_error_deg") is not None
            ]
            translations = [
                row["fp_translation_error_mm"]
                for row in live_fp_records
                if row.get("fp_translation_error_mm") is not None
            ]
            episode_record.update({
                "live_fp_updates": len(live_fp_records),
                "live_fp_age_ms_p50": float(np.percentile(ages, 50)) if ages else None,
                "live_fp_age_ms_p95": float(np.percentile(ages, 95)) if ages else None,
                "live_fp_track_rotation_error_deg_p95": (
                    float(np.percentile(rotations, 95)) if rotations else None
                ),
                "live_fp_track_translation_error_mm_p95": (
                    float(np.percentile(translations, 95)) if translations else None
                ),
                "live_fp_records": live_fp_records,
            })
        if external_pairing is not None:
            episode_record.update(external_pairing)
            episode_record.update({
                "external_goal_bank": getattr(
                    env.unwrapped, "external_goal_bank_path", None
                ),
                "external_goal_source": getattr(
                    env.unwrapped, "external_goal_source", None
                ),
            })
        if episode_record_context:
            episode_record.update(episode_record_context)
        if structured_step_trace:
            episode_record["reset_pose_snapshot"] = reset_pose_snapshot
            episode_record["step_trace"] = step_trace
        episode_records.append(episode_record)
        append_jsonl(
            episode_jsonl_path,
            compact_incremental_episode_record(episode_record),
        )
        logger.info(
            "Episode %s: reason=%s steps=%s length=%0.2fmm trans=%0.3fcm angle=%0.2fdeg "
            "closed_jaw=%s min_jaw=%0.3f",
            episode,
            termination_reason,
            episode_steps,
            trajectory_length,
            episode_record["final_trans_error_cm"],
            criterion_angle_error_deg,
            closed_jaw_observed,
            min_achieved_jaw,
        )
    
    # --- E. eval obs distribution summary (same format as TRAIN_OBS_DIST_DIAG) ---
    eval_obs_summary = {}
    if eval_obs_buf["obs"]:
        ach = np.stack(eval_obs_buf["ach"])
        des = np.stack(eval_obs_buf["des"])
        ob = np.stack(eval_obs_buf["obs"])
        act = np.stack(eval_obs_buf["act"])
        raw_rpy = des[:, 3:6] - ach[:, 3:6]
        wr_rpy = wrap_to_pi(raw_rpy)

        def mmm(name, arr):
            arr = np.asarray(arr)
            print(f"  {name} min={np.round(arr.min(0),3).tolist()} "
                  f"max={np.round(arr.max(0),3).tolist()} "
                  f"mean={np.round(arr.mean(0),3).tolist()}")

        print("EVAL_OBS_DIST_DIAG:")
        print(f"  n={ob.shape[0]}")
        mmm("achieved_xyz_cm", ach[:, 0:3])
        mmm("desired_xyz_cm", des[:, 0:3])
        mmm("obs_delta_trans_cm(obs[14:17])", ob[:, 14:17])
        mmm("obs_delta_rpy_raw_deg(obs[17:20])", np.degrees(ob[:, 17:20]))
        mmm("desire_minus_achieved_rpy_raw_deg", np.degrees(raw_rpy))
        mmm("desire_minus_achieved_rpy_wrapped_deg", np.degrees(wr_rpy))
        raw_l2 = np.degrees(np.linalg.norm(raw_rpy, axis=1))
        wr_l2 = np.degrees(np.linalg.norm(wr_rpy, axis=1))
        print(f"  raw_rpy_l2_deg min/max/mean={raw_l2.min():.1f}/{raw_l2.max():.1f}/{raw_l2.mean():.1f}")
        print(f"  wrapped_rpy_l2_deg min/max/mean={wr_l2.min():.1f}/{wr_l2.max():.1f}/{wr_l2.mean():.1f}")
        mmm("action", act)
        print(f"  action_saturation_ratio(|a|>0.99)={np.mean(np.abs(act) > 0.99):.3f}")
        eval_obs_summary = {
            "n": int(ob.shape[0]),
            "achieved_xyz_cm_min": ach[:, 0:3].min(0).tolist(),
            "achieved_xyz_cm_max": ach[:, 0:3].max(0).tolist(),
            "achieved_xyz_cm_mean": ach[:, 0:3].mean(0).tolist(),
            "desired_xyz_cm_min": des[:, 0:3].min(0).tolist(),
            "desired_xyz_cm_max": des[:, 0:3].max(0).tolist(),
            "desired_xyz_cm_mean": des[:, 0:3].mean(0).tolist(),
            "obs_delta_trans_cm_min": ob[:, 14:17].min(0).tolist(),
            "obs_delta_trans_cm_max": ob[:, 14:17].max(0).tolist(),
            "obs_delta_trans_cm_mean": ob[:, 14:17].mean(0).tolist(),
            "obs_delta_rpy_raw_deg_min": np.degrees(ob[:, 17:20]).min(0).tolist(),
            "obs_delta_rpy_raw_deg_max": np.degrees(ob[:, 17:20]).max(0).tolist(),
            "obs_delta_rpy_raw_deg_mean": np.degrees(ob[:, 17:20]).mean(0).tolist(),
            "desired_minus_achieved_rpy_raw_deg_min": np.degrees(raw_rpy).min(0).tolist(),
            "desired_minus_achieved_rpy_raw_deg_max": np.degrees(raw_rpy).max(0).tolist(),
            "desired_minus_achieved_rpy_raw_deg_mean": np.degrees(raw_rpy).mean(0).tolist(),
            "desired_minus_achieved_rpy_wrapped_deg_min": np.degrees(wr_rpy).min(0).tolist(),
            "desired_minus_achieved_rpy_wrapped_deg_max": np.degrees(wr_rpy).max(0).tolist(),
            "desired_minus_achieved_rpy_wrapped_deg_mean": np.degrees(wr_rpy).mean(0).tolist(),
            "raw_rpy_l2_deg_min": float(raw_l2.min()),
            "raw_rpy_l2_deg_max": float(raw_l2.max()),
            "raw_rpy_l2_deg_mean": float(raw_l2.mean()),
            "wrapped_rpy_l2_deg_min": float(wr_l2.min()),
            "wrapped_rpy_l2_deg_max": float(wr_l2.max()),
            "wrapped_rpy_l2_deg_mean": float(wr_l2.mean()),
            "action_min": act.min(0).tolist(),
            "action_max": act.max(0).tolist(),
            "action_mean": act.mean(0).tolist(),
            "action_saturation_ratio": float(np.mean(np.abs(act) > 0.99)),
            "action_saturation_ratio_by_dim": np.mean(np.abs(act) > 0.99, axis=0).tolist(),
        }

    # reset_invalid episodes are excluded from the success-rate denominator.
    valid_episodes = num_episodes - len(reset_invalid_records)
    success_rate = (total_success / valid_episodes) if valid_episodes > 0 else 0.0
    avg_length = total_length / total_success if total_success > 0 else 0
    avg_timecost = total_timecost / total_success if total_success > 0 else 0
    if reset_invalid_records:
        print(
            "RESET_INVALID_SUMMARY:",
            {"reset_invalid": len(reset_invalid_records),
             "valid_episodes": valid_episodes,
             "num_episodes": num_episodes},
            flush=True,
        )

    return success_rate, avg_length, avg_timecost, all_lengths, all_timecosts, episode_records, eval_obs_summary, reset_invalid_records

def save_results(args, results, train_seeds, test_env):
    variant = experiment_variant(variant=args.variant, randomized=args.randomized, stepDR=args.stepDR)
    out_dir = ensure_dir(experiment_dir(ExperimentKey(
        task_name=args.task_name,
        algorithm=args.algorithm,
        reward_type=args.reward_type,
        seed=args.eval_seed,
        variant=f"{variant}_evaluation",
    )))
    results_dir = ensure_dir(out_dir / "evaluation_results" / str(test_env))

    # Save detailed results to txt file (include test environment to avoid overwriting)
    safe_test_env = str(test_env)
    #txt_file = os.path.join(results_dir, f"{args.task_name}_{args.algorithm}_{args.reward_type}_{randomization_str}_{safe_test_env}_results.txt")
    txt_file = results_dir / "results.txt"
    with open(txt_file, 'w') as f:
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Reward Type: {args.reward_type}\n")
        f.write(f"Number of seeds: {len(train_seeds)}\n")
        f.write(f"Evaluation seed: {args.eval_seed}\n")
        f.write(f"Test Environment: {test_env}\n")
        f.write("Results:\n")
        f.write(f"Success Rate: {results['mean_success_rate']:.2%} ± {results['std_success_rate']:.2%}\n")
        f.write(f"Average Trajectory Length: {results['mean_avg_length']:.2f} ± {results['std_avg_length']:.2f} mm\n")
        f.write(f"Average Time Cost: {results['mean_avg_timecost']:.2f} ± {results['std_avg_timecost']:.2f} steps\n\n\n")
    
    logger.info("Detailed results saved to %s", txt_file)

    numbers_file = results_dir / "numbers.txt"
    with open(numbers_file, 'w') as f:
        f.write(f"{results['mean_success_rate']} {results['std_success_rate']} ")
        f.write(f"{results['mean_avg_length']} {results['std_avg_length']} ")
        f.write(f"{results['mean_avg_timecost']} {results['std_avg_timecost']}")
    
    logger.info("Numeric results saved to %s", numbers_file)

def main():
    args = parse_arguments()
    setup_logging(level=args.log_level, log_file=args.log_file)
    seed_everything(args.eval_seed, deterministic_torch=args.deterministic_eval)
    
    test_envs = ["stepDR_env"] if args.stepDR else ["base_env"]
    
    for test_env in test_envs:
        logger.info("Evaluating in environment: %s", test_env)

        env, step_size, threshold, max_episode_steps = setup_environment(args, test_env)
        if args.deterministic_eval:
            seed_everything(args.eval_seed, deterministic_torch=True, env=env)
        
        train_seeds = args.train_seeds if args.train_seeds is not None else [1, 5, 10, 15, 100, 150, 1000, 1500, 10000, 15000]
        all_success_rates = []
        all_lengths = []
        all_timecosts = []
        all_episode_records = []
        all_reset_invalid_records = []
        eval_obs_summaries = []
        
        for train_seed in train_seeds:
            model = load_model(
                args.algorithm,
                env,
                args.task_name,
                args.reward_type,
                train_seed,
                args.randomized,
                args.stepDR,
                args.model_path,
                args.variant,
            )
            
            success_rate, avg_length, avg_timecost, lengths, timecosts, episode_records, eval_obs_summary, reset_invalid_records = run_evaluation(
                env,
                model,
                args.num_episodes,
                max_episode_steps,
                controller=args.controller,
                residual_policy_weight=args.residual_policy_weight,
                residual_servo_weight=args.residual_servo_weight,
                residual_direction_guard=not args.disable_residual_direction_guard,
                structured_step_trace=args.structured_step_trace,
                divergence_abort_cm=args.divergence_abort_cm,
                stall_abort_step=args.stall_abort_step,
                grasp_approach_dwell_steps=args.grasp_approach_dwell_steps,
                grasp_confirmation_steps=args.grasp_confirmation_steps,
                grasp_open_jaw=args.grasp_open_jaw,
                max_consecutive_reset_invalid=args.max_consecutive_reset_invalid,
                episode_jsonl_path=args.episode_jsonl,
                external_snapshot_lock=not args.no_external_snapshot_lock,
                live_fp_pose_file=args.live_fp_pose_file,
                live_fp_control_file=args.live_fp_control_file,
                live_fp_initial_wait_s=args.live_fp_initial_wait_s,
                freeze_jaw_command=args.freeze_jaw_command,
                pre_episode_settle_s=args.pre_episode_settle_s,
                episode_record_context={
                    "train_seed": int(train_seed),
                    "eval_seed": int(args.eval_seed),
                    "freeze_live_goal": bool(args.freeze_live_goal),
                    "model_path": str(Path(args.model_path).expanduser().resolve())
                    if args.model_path else None,
                },
            )
            for record in reset_invalid_records:
                record["train_seed"] = int(train_seed)
            all_reset_invalid_records.extend(reset_invalid_records)

            all_success_rates.append(success_rate)
            all_lengths.extend(lengths)
            all_timecosts.extend(timecosts)
            for record in episode_records:
                record["train_seed"] = int(train_seed)
            all_episode_records.extend(episode_records)
            eval_obs_summary["train_seed"] = int(train_seed)
            eval_obs_summaries.append(eval_obs_summary)
            
            logger.info(
                "Seed %s: success=%0.2f%%, avg_len=%0.2fmm, avg_steps=%0.2f",
                train_seed,
                success_rate * 100.0,
                avg_length,
                avg_timecost,
            )
        
        # Calculate mean and standard deviation across all seeds
        mean_success_rate = np.mean(all_success_rates)
        std_success_rate = np.std(all_success_rates)
        mean_avg_length = np.mean(all_lengths) if all_lengths else 0.0
        std_avg_length = np.std(all_lengths) if all_lengths else 0.0
        mean_avg_timecost = np.mean(all_timecosts) if all_timecosts else 0.0
        std_avg_timecost = np.std(all_timecosts) if all_timecosts else 0.0
        
        logger.info(
            "Final across seeds: success=%0.2f%%±%0.2f%%, avg_len=%0.2f±%0.2fmm, avg_steps=%0.2f±%0.2f",
            mean_success_rate * 100.0,
            std_success_rate * 100.0,
            mean_avg_length,
            std_avg_length,
            mean_avg_timecost,
            std_avg_timecost,
        )
        
        results = {
            'mean_success_rate': mean_success_rate,
            'std_success_rate': std_success_rate,
            'mean_avg_length': mean_avg_length,
            'std_avg_length': std_avg_length,
            'mean_avg_timecost': mean_avg_timecost,
            'std_avg_timecost': std_avg_timecost
        }
        # Only completed episodes carry per-episode metrics; reset_invalid
        # episodes are tracked separately and must not enter these stats (nor
        # break them when every episode was reset_invalid -> empty lists).
        all_episode_lengths = [record["trajectory_length_mm"] for record in all_episode_records]
        all_episode_steps = [record["steps"] for record in all_episode_records]
        all_final_trans_errors = [record["final_trans_error_cm"] for record in all_episode_records]
        all_final_angle_errors = [record["final_criterion_angle_error_deg"] for record in all_episode_records]

        def _safe_mean(xs):
            return float(np.mean(xs)) if len(xs) else 0.0

        def _safe_std(xs):
            return float(np.std(xs)) if len(xs) else 0.0

        results.update({
            'mean_episode_length_mm_all': _safe_mean(all_episode_lengths),
            'std_episode_length_mm_all': _safe_std(all_episode_lengths),
            'mean_episode_steps_all': _safe_mean(all_episode_steps),
            'std_episode_steps_all': _safe_std(all_episode_steps),
            'mean_final_trans_error_cm': _safe_mean(all_final_trans_errors),
            'std_final_trans_error_cm': _safe_std(all_final_trans_errors),
            'mean_final_angle_error_deg': _safe_mean(all_final_angle_errors),
            'std_final_angle_error_deg': _safe_std(all_final_angle_errors),
            'reset_invalid_total': int(len(all_reset_invalid_records)),
            'completed_episodes_total': int(len(all_episode_records)),
        })
        logger.info(
            "Reset validity: completed=%d reset_invalid=%d (reset_invalid excluded from success denominator)",
            len(all_episode_records),
            len(all_reset_invalid_records),
        )
        logger.info(
            "All episodes: length=%0.2f±%0.2fmm steps=%0.2f±%0.2f "
            "final_trans=%0.3f±%0.3fcm final_angle=%0.2f±%0.2fdeg",
            results['mean_episode_length_mm_all'],
            results['std_episode_length_mm_all'],
            results['mean_episode_steps_all'],
            results['std_episode_steps_all'],
            results['mean_final_trans_error_cm'],
            results['std_final_trans_error_cm'],
            results['mean_final_angle_error_deg'],
            results['std_final_angle_error_deg'],
        )

        save_results(args, results, train_seeds, test_env)

        if args.results_json:
            result_path = Path(args.results_json).expanduser()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps({
                "task": args.task_name,
                "reset_invalid_total": int(len(all_reset_invalid_records)),
                "completed_episodes_total": int(len(all_episode_records)),
                "reset_invalid_records": all_reset_invalid_records,
                "max_consecutive_reset_invalid": int(args.max_consecutive_reset_invalid),
                "algorithm": args.algorithm,
                "controller": args.controller,
                "residual_policy_weight": args.residual_policy_weight,
                "residual_servo_weight": args.residual_servo_weight,
                "residual_direction_guard": not args.disable_residual_direction_guard,
                "checkpoint_compat": bool(args.checkpoint_compat),
                "command_integrated": bool(getattr(env.unwrapped, "command_integrated", False)),
                "command_state_clamp": bool(getattr(env.unwrapped, "command_state_clamp", False)),
                "command_rpy_contract": (
                    "wrapped_to_minus_pi_pi"
                    if getattr(env.unwrapped, "command_state_clamp", False)
                    else "legacy_unbounded"
                ),
                "success_reward_pose_source": (
                    "measured" if getattr(env.unwrapped, "measured_success_reward", False) else "command"
                ),
                "success_jaw_source": (
                    "RigidBodyState.joint_positions[6]"
                    if getattr(env.unwrapped, "measured_success_reward", False)
                    else "command_integrated"
                ),
                "needle_settle_steps": int(getattr(env.unwrapped, "needle_settle_steps", 0)),
                "needle_settle_interval_s": float(getattr(env.unwrapped, "needle_settle_interval_s", 0.0)),
                "deterministic_eval": bool(args.deterministic_eval),
                "synchronous_physics": bool(getattr(env.unwrapped, "synchronous_physics", False)),
                "execution_mode": (
                    "strict_sync"
                    if getattr(env.unwrapped, "synchronous_physics", False)
                    else "asynchronous"
                ),
                "ambf_world_state_frequency_hz": int(os.environ.get("SURGICAI_AMBF_COMM_FREQ", "0")),
                "ambf_physics_frequency_hz": int(os.environ.get("SURGICAI_AMBF_PHYSICS_FREQ", "0")),
                "ambf_sim_speed_factor": float(os.environ.get("SURGICAI_AMBF_SIM_SPEED", "1")),
                "physics_steps_per_action": int(getattr(env.unwrapped, "physics_steps_per_action", 0)),
                "physics_barrier_timeout_s": float(getattr(env.unwrapped, "physics_barrier_timeout_s", 0.0)),
                "command_workspace_low_m": np.asarray(
                    getattr(env.unwrapped, "COMMAND_WORKSPACE_LOW_M", []), dtype=float
                ).tolist(),
                "command_workspace_high_m": np.asarray(
                    getattr(env.unwrapped, "COMMAND_WORKSPACE_HIGH_M", []), dtype=float
                ).tolist(),
                "command_jaw_limits": [
                    float(getattr(env.unwrapped, "COMMAND_JAW_LOW", 0.0)),
                    float(getattr(env.unwrapped, "COMMAND_JAW_HIGH", 1.0)),
                ],
                "randomize_psm_reset": bool(getattr(env.unwrapped, "randomize_psm_reset", False)),
                "randomize_needle_reset": bool(getattr(env.unwrapped, "randomize_needle_reset", False)),
                "fixed_historical_goal": bool(getattr(env.unwrapped, "fixed_historical_goal", False)),
                "raw_rpy_contract": bool(getattr(env.unwrapped, "raw_rpy_contract", False)),
                "wrap_obs_rpy_delta": bool(getattr(env.unwrapped, "wrap_obs_rpy_delta", False)),
                "align_obs_rpy_branch": bool(getattr(env.unwrapped, "align_obs_rpy_branch", False)),
                "require_closed_jaw": bool(getattr(env.unwrapped, "require_closed_jaw", False)),
                "require_grasp_confirmation": bool(getattr(env.unwrapped, "require_grasp_confirmation", False)),
                "grasp_confirmation_steps": int(getattr(env.unwrapped, "grasp_confirmation_steps", 1)),
                "grasp_approach_dwell_steps": args.grasp_approach_dwell_steps,
                "grasp_open_jaw": args.grasp_open_jaw,
                "checkpoint_goal_bank_size": len(getattr(env.unwrapped, "checkpoint_goal_bank", [])),
                "checkpoint_goal_bank_model_path": args.checkpoint_goal_bank_model_path,
                "checkpoint_goal_random": bool(getattr(env.unwrapped, "checkpoint_goal_random", False)),
                "checkpoint_goal_random_size": len(getattr(env.unwrapped, "checkpoint_goal_random_goals", [])),
                "checkpoint_goal_random_seed": args.checkpoint_goal_random_seed,
                "checkpoint_goal_jitter": bool(getattr(env.unwrapped, "checkpoint_goal_jitter", False)),
                "checkpoint_goal_jitter_size": len(getattr(env.unwrapped, "checkpoint_goal_jitter_goals", [])),
                "checkpoint_goal_jitter_seed": args.checkpoint_goal_jitter_seed,
                "checkpoint_goal_jitter_scale": args.checkpoint_goal_jitter_scale,
                "checkpoint_goal_offset_sweep": bool(getattr(env.unwrapped, "checkpoint_goal_offset_sweep", False)),
                "checkpoint_goal_offset_size": len(getattr(env.unwrapped, "checkpoint_goal_offset_goals", [])),
                "checkpoint_goal_offset_dim": args.checkpoint_goal_offset_dim,
                "checkpoint_goal_offset_values": args.checkpoint_goal_offset_values,
                "external_goal_bank": getattr(
                    env.unwrapped, "external_goal_bank_path", None
                ),
                "external_goal_source": getattr(
                    env.unwrapped, "external_goal_source", None
                ),
                "external_goal_bank_size": len(
                    getattr(env.unwrapped, "external_goal_bank_entries", [])
                ),
                "external_pairing_trans_tol_mm": getattr(
                    env.unwrapped, "external_pairing_trans_tol_mm", None
                ),
                "external_pairing_rot_tol_deg": getattr(
                    env.unwrapped, "external_pairing_rot_tol_deg", None
                ),
                "freeze_live_goal": bool(getattr(env.unwrapped, "freeze_live_goal", False)),
                "goal_source_audit": bool(getattr(env.unwrapped, "goal_source_audit", False)),
                "fixed_psm_reset": bool(args.fixed_psm_reset),
                "needle_random_x_mm": args.needle_random_x_mm,
                "needle_random_y_mm": args.needle_random_y_mm,
                "needle_random_rz_deg": args.needle_random_rz_deg,
                "psm_reset_noise_xyz_mm": args.psm_reset_noise_xyz_mm,
                "psm_reset_noise_rpy_deg": args.psm_reset_noise_rpy_deg,
                "psm_reset_noise_jaw": args.psm_reset_noise_jaw,
                "psm2_reset_offset_xyz_mm": args.psm2_reset_offset_xyz_mm,
                "psm2_reset_offset_rpy_deg": args.psm2_reset_offset_rpy_deg,
                "psm2_reset_offset_jaw": args.psm2_reset_offset_jaw,
                "goal_offset_xyz_mm": args.goal_offset_xyz_mm,
                "goal_offset_rpy_deg": args.goal_offset_rpy_deg,
                "grasp_start_deg": args.grasp_start_deg,
                "grasp_end_deg": args.grasp_end_deg,
                "grasp_angle_deg": args.grasp_angle_deg,
                "lift_height_mm": args.lift_height_mm,
                "step_size": np.asarray(env.unwrapped.step_size, dtype=float).tolist(),
                "reward_type": args.reward_type,
                "translation_threshold_cm": args.trans_error,
                "angle_threshold_deg": args.angle_error,
                "eval_seed": args.eval_seed,
                "episodes_per_seed": args.num_episodes,
                "train_seeds": train_seeds,
                "structured_step_trace": bool(args.structured_step_trace),
                "episode_jsonl": args.episode_jsonl,
                "structured_step_trace_schema_version": 2 if args.structured_step_trace else None,
                "structured_step_trace_schema": ({
                    "name": "SurgicAI.Approach.step_trace",
                    "version": 2,
                    "action": "normalized_7d_xyz_rpy_jaw",
                    "pose_state_order": ["x", "y", "z", "roll", "pitch", "yaw", "jaw"],
                    "units": {
                        "xyz": "cm",
                        "rpy": "rad",
                        "rotation_geodesic": "deg",
                        "jaw": "normalized_joint_angle",
                    },
                    "gap_policy": "translation, rotation, and jaw are separate; no mixed-unit L2",
                    "measured_jaw_source": "RigidBodyState.joint_positions[6]",
                    "termination_reasons": [
                        "continuing",
                        "success",
                        "grasp_confirmed",
                        "diverged_abort",
                        "stall_abort",
                        "truncation",
                        "terminated_other",
                    ],
                } if args.structured_step_trace else None),
                "divergence_abort_cm": args.divergence_abort_cm,
                "stall_abort_step": args.stall_abort_step,
                "measured_contract_rates": {
                    "pose_1mm_rot10": float(np.mean([
                        record["measured_contract_pose_1mm_rot10_ever"]
                        for record in all_episode_records
                    ])) if all_episode_records else 0.0,
                    "pose_1cm_rot10": float(np.mean([
                        record["measured_contract_pose_1cm_rot10_ever"]
                        for record in all_episode_records
                    ])) if all_episode_records else 0.0,
                    "paper_1mm_rot10_jaw01": float(np.mean([
                        record["measured_contract_paper_1mm_rot10_jaw01_ever"]
                        for record in all_episode_records
                    ])) if all_episode_records else 0.0,
                    "eval_1cm_rot10_jaw01": float(np.mean([
                        record["measured_contract_eval_1cm_rot10_jaw01_ever"]
                        for record in all_episode_records
                    ])) if all_episode_records else 0.0,
                    "jaw_source": "RigidBodyState.joint_positions[6]",
                },
                "eval_obs_summaries": eval_obs_summaries,
                "episode_records": all_episode_records,
                **{key: float(value) for key, value in results.items()},
            }, indent=2) + "\n")

        if args.min_success_rate is not None and mean_success_rate < args.min_success_rate:
            raise SystemExit(
                f"Success-rate gate failed: {mean_success_rate:.2%} < {args.min_success_rate:.2%}"
            )

        env.close()

if __name__ == "__main__":
    main()
