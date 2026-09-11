import argparse
import hashlib
import json
import pickle
import time
import numpy as np
import gymnasium as gym
import importlib
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from RL.algorithm_configs_online import get_algorithm_config
import gc
import torch
import sys
import threading
from pathlib import Path

from RL.rl_paths import ExperimentKey, checkpoints_dir, ensure_dir, experiment_dir, rl_dir
from RL.needle_reset_ranges import (
    ASSUMED_REAL_RZ_DEG,
    ASSUMED_REAL_X_MM,
    ASSUMED_REAL_Y_MM,
)
from RL.utils.checkpoint_io import load_sb3_checkpoint, resolve_checkpoint_path
from RL.utils.cli_args import add_common_logging_args, add_experiment_variant_arg, add_seed_arg, add_threshold_args
from RL.utils.logging_utils import get_logger, setup_logging
from RL.utils.seed import seed_everything
from RL.utils.utils import resolve_src_env, default_step_size, threshold_from_args, experiment_variant

gc.collect()
torch.cuda.empty_cache()
logger = get_logger(__name__)

def _try_load_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def load_expert_data(task_name: str, expert_data_path: str | None):
    """
    Load expert trajectories if provided/available.

    Search order:
    - `--expert-data PATH` if provided
    - `RL/Expert_traj/<task_name>/all_episodes_merged.pkl` (historical default)
    """
    candidates: list[Path] = []
    if expert_data_path:
        candidates.append(Path(expert_data_path).expanduser())
    candidates.append(rl_dir() / "Expert_traj" / str(task_name) / "all_episodes_merged.pkl")

    for p in candidates:
        data = _try_load_pickle(p)
        if data is not None:
            logger.info("Loaded expert data from %s", p)
            return data

    logger.warning("No expert data found (continuing without expert trajectories).")
    return None


def validate_expert_data(expert_data, step_size):
    """Fail fast on malformed or still-command-integrated demonstrations."""
    if not isinstance(expert_data, list) or not expert_data:
        raise ValueError("Expert data must be a non-empty transition list")
    required = {"obs", "next_obs", "action", "reward", "done"}
    obs_keys = {"observation", "achieved_goal", "desired_goal"}
    residuals = []
    done_count = 0
    step_obs_units = np.asarray(step_size, dtype=np.float64).copy()
    step_obs_units[:3] *= 100.0
    for index, transition in enumerate(expert_data):
        missing = required - set(transition)
        if missing:
            raise ValueError(f"Demo transition {index} is missing {sorted(missing)}")
        for side in ("obs", "next_obs"):
            if set(transition[side]) != obs_keys:
                raise ValueError(
                    f"Demo transition {index} {side} keys are {sorted(transition[side])}"
                )
            for key, expected_shape in (
                ("observation", (21,)),
                ("achieved_goal", (7,)),
                ("desired_goal", (7,)),
            ):
                value = np.asarray(transition[side][key], dtype=np.float64)
                if value.shape != expected_shape or not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"Demo transition {index} {side}.{key} is invalid: {value.shape}"
                    )
        action = np.asarray(transition["action"], dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Demo transition {index} action is invalid")
        if np.any(action < -1.00001) or np.any(action > 1.00001):
            raise ValueError(f"Demo transition {index} action exceeds [-1, 1]")
        residuals.append(
            np.asarray(transition["next_obs"]["achieved_goal"], dtype=np.float64)
            - np.asarray(transition["obs"]["achieved_goal"], dtype=np.float64)
            - action * step_obs_units
        )
        done_count += int(bool(np.asarray(transition["done"]).reshape(-1)[0]))

    if done_count < 10:
        raise ValueError(
            f"Only {done_count} complete demonstrations; BC requires at least 10"
        )
    residual_norm = np.linalg.norm(np.asarray(residuals), axis=1)
    exact_fraction = float(np.mean(residual_norm <= 1.0e-5))
    if exact_fraction > 0.01:
        raise ValueError(
            "Measured demo validation failed: too many transitions still satisfy "
            f"the exact command integrator ({exact_fraction:.2%})"
        )
    return {
        "transitions": len(expert_data),
        "complete_demonstrations": done_count,
        "integrator_exact_fraction_le_1e-5": exact_fraction,
        "integrator_residual_norm_median": float(np.median(residual_norm)),
        "integrator_residual_norm_max": float(np.max(residual_norm)),
    }

def create_model(args, env, expert_data):
    algorithm_config = get_algorithm_config(args.algorithm, env, args.task_name, args.reward_type, args.seed, expert_data)
    model_class = algorithm_config['class']
    model_params = algorithm_config['params']
    return model_class(**model_params)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def create_or_resume_model(args, env, expert_data):
    if not args.resume_model:
        model = create_model(args, env, expert_data)
        return model, 0, None

    algorithm_config = get_algorithm_config(
        args.algorithm,
        env,
        args.task_name,
        args.reward_type,
        args.seed,
        expert_data,
    )
    checkpoint = resolve_checkpoint_path(args.resume_model)
    model = load_sb3_checkpoint(algorithm_config["class"], checkpoint, env=env)
    resume_start = int(model.num_timesteps)
    target = int(args.total_timesteps)
    if resume_start >= target:
        raise ValueError(
            f"Resume checkpoint already has {resume_start} timesteps; "
            f"target is {target}"
        )

    # SB3 checkpoints intentionally exclude the online replay buffer.  Restore
    # the BC dataset explicitly and record the empty online-buffer restart so a
    # resumed run is never mistaken for a bit-identical uninterrupted run.
    params = algorithm_config["params"]
    if hasattr(model, "preprocess_demo_data"):
        model.demo_data = model.preprocess_demo_data(expert_data)
    for name in ("BC_coeff", "demo_ratio", "Q_filter"):
        if name in params:
            setattr(model, name, params[name])
    replay_buffer = getattr(model, "replay_buffer", None)
    if replay_buffer is not None and hasattr(replay_buffer, "preprocess_demo_data"):
        replay_buffer.demo_transitions = expert_data
        replay_buffer.demo_data = replay_buffer.preprocess_demo_data(expert_data)

    logger.warning(
        "RESUME_WITH_FRESH_ONLINE_REPLAY checkpoint=%s start=%d target=%d",
        checkpoint,
        resume_start,
        target,
    )
    return model, resume_start, checkpoint

def setup_environment(args):
    max_episode_steps = int(args.max_episode_steps)
    step_size = default_step_size(trans_step=1.5e-3, angle_step_deg=3.0, jaw_step=0.05)
    threshold = threshold_from_args(args.trans_error, args.angle_error)
    SRC_class = resolve_src_env(args.task_name)
    
    gym.envs.register(id=f"{args.algorithm}_{args.reward_type}", entry_point=SRC_class, max_episode_steps=max_episode_steps)
    env_kwargs = {
        "render_mode": "human",
        "reward_type": args.reward_type,
        "max_episode_step": max_episode_steps,
        "seed": args.seed,
        "step_size": step_size,
        "threshold": threshold,
        "stepDR": args.stepDR,
    }
    if args.task_name.lower() == "approach" and args.measured_contract:
        env_kwargs.update({
            "command_integrated": False,
            "command_state_clamp": True,
            "measured_success_reward": True,
            # Pose-close-only benchmark contract: success on measured pose-close
            # plus scripted actuate attach; drop the jaw-closed gate (the jaws
            # physically close on the phantom, not the needle). Must match the
            # contract the pose-close demos were collected under, else the policy
            # reaches pose with jaw open and a jaw gate would score 0%.
            "require_closed_jaw": not args.pose_close_only,
            "jaw_success_source": args.jaw_success_source,
            "require_grasp_confirmation": False,
            # R3 demonstrations and independent evaluation both freeze one
            # reset-time goal.  Training must use that same single desired_goal;
            # otherwise reward/HER optimize one goal while termination accepts
            # any of 25 live grasp-angle candidates.
            "freeze_live_goal": True,
            "needle_random_range": np.asarray(
                [
                    args.needle_random_x_mm * 1.0e-3,
                    args.needle_random_y_mm * 1.0e-3,
                    np.deg2rad(args.needle_random_rz_deg),
                ],
                dtype=np.float32,
            ),
            "needle_settle_steps": args.needle_settle_steps,
            "needle_settle_interval_s": args.needle_settle_interval_s,
        })
    env = gym.make(f"{args.algorithm}_{args.reward_type}", **env_kwargs)
    if args.measured_contract:
        unwrapped = env.unwrapped
        if bool(getattr(unwrapped, "command_integrated", False)):
            raise RuntimeError("Measured training contract unexpectedly enabled command integration")
        if not bool(getattr(unwrapped, "measured_success_reward", False)):
            raise RuntimeError("Measured training contract did not enable measured reward/success")
        logger.info(
            "MEASURED_TRAIN_PREFLIGHT max_steps=%d trans_cm=%g rot_deg=%g "
            "jaw_success_source=%s command_state_clamp=%s random_range=%s",
            int(getattr(unwrapped, "max_timestep", -1)),
            float(getattr(unwrapped, "threshold_trans", np.nan)),
            float(np.rad2deg(getattr(unwrapped, "threshold_angle", np.nan))),
            getattr(unwrapped, "jaw_success_source", None),
            bool(getattr(unwrapped, "command_state_clamp", False)),
            np.asarray(getattr(unwrapped, "random_range", []), dtype=float).tolist(),
        )
    return env, step_size, threshold, max_episode_steps

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the RL algorithm to use')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='dense', help='Reward type')
    parser.add_argument('--total_timesteps', type=int, default=150000, help='Total timesteps for training')
    parser.add_argument('--save_freq', type=int, default=50000, help='Frequency of saving checkpoints')
    add_seed_arg(parser, name="--seed", default=10)
    add_threshold_args(parser)
    parser.add_argument('--randomization_params', type=str, default='0,0,0,0,0', help='Randomization parameters')
    parser.add_argument('--stepDR', action='store_true', help='Enable state-space domain randomization')
    parser.add_argument('--variant', type=str, default=None, help='Free-form experiment directory variant (for example measured_e3e).')
    parser.add_argument('--gui', action='store_true', help='Enable GUI for domain randomization')
    parser.add_argument('--expert-data', type=str, default=None, help='Optional path to expert trajectories pickle')
    parser.add_argument('--measured-contract', action='store_true', help='Train Approach from measured Cartesian observations and measured pose reward/success.')
    parser.add_argument('--jaw-success-source', choices=['measured', 'command'], default='command', help='Jaw source for Approach termination; measured Cartesian pose is always retained.')
    parser.add_argument('--pose-close-only', action='store_true', help='Approach benchmark contract: success on measured pose-close + scripted actuate attach; drop the jaw-closed gate (jaws close on phantom, not needle). Match the pose-close demo collection.')
    parser.add_argument('--needle-settle-steps', type=int, default=60, help='Maximum reset settling samples before capturing the live needle goal.')
    parser.add_argument('--needle-settle-interval-s', type=float, default=0.1, help='Wall-clock interval between reset settling cycles.')
    parser.add_argument('--needle-random-x-mm', type=float, default=ASSUMED_REAL_X_MM, help='Approach needle reset x half-range in millimeters; default remains the R2 range.')
    parser.add_argument('--needle-random-y-mm', type=float, default=ASSUMED_REAL_Y_MM, help='Approach needle reset y half-range in millimeters; default remains the R2 range.')
    parser.add_argument('--needle-random-rz-deg', type=float, default=ASSUMED_REAL_RZ_DEG, help='Approach needle reset yaw half-range in degrees; default remains the R2 range.')
    parser.add_argument('--max-episode-steps', type=int, default=1000, help='Environment and Gym TimeLimit horizon.')
    parser.add_argument('--preflight-only', action='store_true', help='Validate the expert dataset without constructing AMBF or calling learn().')
    parser.add_argument('--no-progress-bar', action='store_true', help='Disable the SB3 tqdm/rich progress bar.')
    parser.add_argument('--resume-model', type=str, default=None, help='Resume from an SB3 checkpoint up to --total_timesteps; the online replay buffer restarts empty.')
    add_common_logging_args(parser)
    return parser.parse_args()

def run_training(args, env):
      
    
    # Load expert data
    expert_data = load_expert_data(args.task_name, args.expert_data)
    demo_stats = validate_expert_data(expert_data, env.unwrapped.step_size)
    logger.info("MEASURED_DEMO_PREFLIGHT %s", demo_stats)

    randomization_str = experiment_variant(
        variant=args.variant,
        stepDR=bool(args.stepDR),
        randomization_params=str(args.randomization_params),
    )
    run_key = ExperimentKey(
        task_name=args.task_name,
        algorithm=args.algorithm,
        reward_type=args.reward_type,
        seed=args.seed,
        variant=randomization_str,
    )
    out_dir = ensure_dir(experiment_dir(run_key))
    run_metadata_path = out_dir / "training_run.json"
    prior_metadata = {}
    if args.resume_model and run_metadata_path.is_file():
        try:
            with open(run_metadata_path, "r", encoding="utf-8") as stream:
                prior_metadata = json.load(stream)
        except (OSError, ValueError):
            logger.exception("Could not read prior training metadata at %s", run_metadata_path)

    model, resume_start, resume_checkpoint = create_or_resume_model(
        args,
        env,
        expert_data,
    )
    replay_buffer = getattr(model, "replay_buffer", None)
    replay_buffer_size = int(replay_buffer.size()) if replay_buffer is not None else None
    started_unix_s = time.time()
    run_metadata = {
        "status": "running",
        "algorithm": args.algorithm,
        "task": args.task_name,
        "reward_type": args.reward_type,
        "seed": args.seed,
        "variant": randomization_str,
        "total_timesteps_requested": args.total_timesteps,
        "measured_contract": bool(args.measured_contract),
        "goal_semantics": (
            "frozen_single_episode_goal" if args.measured_contract else "task_default"
        ),
        "success_goal_count": 1 if args.measured_contract else None,
        "jaw_success_source": args.jaw_success_source,
        "command_state_clamp": bool(env.unwrapped.command_state_clamp),
        "command_workspace_low_m": np.asarray(
            env.unwrapped.COMMAND_WORKSPACE_LOW_M, dtype=float
        ).tolist(),
        "command_workspace_high_m": np.asarray(
            env.unwrapped.COMMAND_WORKSPACE_HIGH_M, dtype=float
        ).tolist(),
        "command_jaw_bounds": [
            float(env.unwrapped.COMMAND_JAW_LOW),
            float(env.unwrapped.COMMAND_JAW_HIGH),
        ],
        "translation_threshold_cm": args.trans_error,
        "rotation_threshold_deg": args.angle_error,
        "max_episode_steps": args.max_episode_steps,
        "needle_settle_steps": args.needle_settle_steps,
        "needle_settle_interval_s": args.needle_settle_interval_s,
        "random_range": np.asarray(env.unwrapped.random_range, dtype=float).tolist(),
        "expert_data_path": str(Path(args.expert_data).expanduser().resolve()) if args.expert_data else None,
        "expert_data": demo_stats,
        "started_unix_s": started_unix_s,
        "original_started_unix_s": prior_metadata.get("started_unix_s"),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_checkpoint_sha256": sha256_file(resume_checkpoint) if resume_checkpoint else None,
        "resume_start_timesteps": resume_start,
        "remaining_timesteps_requested": int(args.total_timesteps) - resume_start,
        "resume_online_replay_buffer_restored": False if resume_checkpoint else None,
        "resume_replay_buffer_size_at_start": replay_buffer_size,
    }
    with open(run_metadata_path, "w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(ensure_dir(checkpoints_dir(run_key))),
        name_prefix="rl_model"
    )
        
    # callback_list = CallbackList([checkpoint_callback, domain_randomization_callback])
    callback_list = CallbackList([checkpoint_callback])
    
    # Train the model
    model.learn(
        total_timesteps=int(args.total_timesteps) - resume_start,
        progress_bar=not args.no_progress_bar,
        callback=callback_list,
        reset_num_timesteps=False,
    )
    
    # Save the final model
    
    save_path = out_dir / "final_model"
    model.save(str(save_path))
    logger.info("Final model saved to %s", save_path)

    run_metadata.update({
        "status": "complete",
        "finished_unix_s": time.time(),
        "wall_time_s": time.time() - started_unix_s,
        "num_timesteps": int(model.num_timesteps),
        "final_model": str(save_path.with_suffix(".zip")),
    })
    with open(run_metadata_path, "w", encoding="utf-8") as stream:
        json.dump(run_metadata, stream, indent=2)

    env.close()
    ral_instance = getattr(env.unwrapped, "ral_instance", None)
    if ral_instance is not None and hasattr(ral_instance, "shutdown"):
        # RAL owns a ROS executor thread.  Leaving it alive until interpreter
        # teardown can make an otherwise complete run abort in rclcpp.
        ral_instance.shutdown()


if __name__ == "__main__":
    
    args = parse_arguments()
    setup_logging(level=args.log_level, log_file=args.log_file)
    seed_everything(args.seed)

    if args.preflight_only:
        expert_data = load_expert_data(args.task_name, args.expert_data)
        step_size = default_step_size(trans_step=1.5e-3, angle_step_deg=3.0, jaw_step=0.05)
        logger.info("MEASURED_DEMO_PREFLIGHT %s", validate_expert_data(expert_data, step_size))
        raise SystemExit(0)
    
    # Setup the environment
    env, step_size, threshold, max_episode_steps = setup_environment(args)
    domain_randomization_callback = None
    if args.randomization_params != "0,0,0,0,0" or args.gui:
        # Import optional ROS/Qt dependencies only when requested.
        from RL.Domain_randomization.Domain_callback import DomainRandomizationCallback
        domain_randomization_callback = DomainRandomizationCallback(env, args.randomization_params, args.seed)

        if args.gui:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QTimer

            app = QApplication(sys.argv)
            # Delay GUI creation until Qt event loop starts.
            QTimer.singleShot(0, lambda: domain_randomization_callback.start_gui(app))

    # Run synchronously so training exceptions propagate to the shell and the
    # final checkpoint/metadata cannot be reported after a failed worker thread.
    run_training(args, env)

    # if args.gui:
    #     sys.exit(app.exec_())
