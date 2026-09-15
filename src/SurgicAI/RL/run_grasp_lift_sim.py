#!/usr/bin/env python3
"""Run approach + grasp + lift in AMBF, using the deployment state machine.

This is the rehearsal for ``deploy/run_grasp_lift.py``.  Both drive the same
``surgicai_rl_deploy.sequence.GraspLiftSequencer``; the only difference is that
AMBF can tell you whether the needle is really held and a real dVRK cannot.

    source /opt/ros/humble/setup.bash
    source "$HOME/ambf_ros_ws/install/setup.bash"
    python3 RL/run_grasp_lift_sim.py --episodes 10 --trans_error 0.5 --angle_error 30

The per-episode record includes both the ground-truth grasp state and the jaw
evidence the *real* deployment would have had to work from, so you can see
directly how much the real run is flying blind.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from RL.GraspLift_env import SRC_grasp_lift
from RL.Approach_env import NeedleResetValidityError
from RL.utils.cli_args import add_common_logging_args, add_threshold_args
from RL.utils.logging_utils import get_logger, setup_logging
from RL.utils.seed import seed_everything
from RL.utils.utils import default_step_size, threshold_from_args

logger = get_logger(__name__)


def build_controller(args):
    from surgicai_rl_deploy.controllers import (
        D2Controller,
        RLController,
        ResidualController,
    )

    if args.controller == "d2":
        return D2Controller(staged=True)

    from surgicai_rl_deploy.policy import ApproachPolicy

    if not args.model:
        raise SystemExit("--model is required for the rl/residual controllers")
    policy = ApproachPolicy.load(
        args.model, device=args.device, verify=not args.allow_unknown_model
    )
    if args.controller == "rl":
        return RLController(policy)
    return ResidualController(
        policy, policy_weight=args.policy_weight, servo_weight=args.servo_weight
    )


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_logging_args(ap)
    add_threshold_args(ap)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--max-cycles", type=int, default=800)

    ap.add_argument("--controller", choices=["d2", "rl", "residual"], default="d2",
                    help="drives the APPROACH phase only; close and lift are "
                         "always the geometric servo, as on hardware")
    ap.add_argument("--model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-unknown-model", action="store_true")
    ap.add_argument("--policy-weight", type=float, default=0.50)
    ap.add_argument("--servo-weight", type=float, default=0.75)

    ap.add_argument("--lift-distance-cm", type=float, default=1.5)
    ap.add_argument("--lift-axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--lift-sign", type=int, choices=[-1, 1], default=1,
                    help="+1 is away from the pad in the PSM base frame, where "
                         "the reset pose sits at z = -0.08 and the needle at "
                         "z = -0.12")
    ap.add_argument("--lift-source", choices=["frame_axis", "needle_evaluator"],
                    default="frame_axis",
                    help="'frame_axis' is byte-for-byte the real deployment's "
                         "lift; 'needle_evaluator' uses the simulator's own "
                         "needle-frame offset as a cross-check")
    ap.add_argument("--grasp-confirm-timeout-steps", type=int, default=200)
    ap.add_argument("--max-episode-step", type=int, default=400)
    ap.add_argument("--json-out")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)
    seed_everything(args.seed)

    env = SRC_grasp_lift(
        seed=args.seed,
        reward_type="dense",
        threshold=threshold_from_args(args.trans_error, args.angle_error),
        max_episode_step=args.max_episode_step,
        step_size=default_step_size(),
        lift_distance_m=args.lift_distance_cm / 100.0,
        lift_axis=args.lift_axis,
        lift_sign=args.lift_sign,
        lift_source=args.lift_source,
        grasp_confirm_timeout_steps=args.grasp_confirm_timeout_steps,
    )
    controller = build_controller(args)

    results = []
    for episode in range(args.episodes):
        try:
            env.reset(seed=args.seed + episode)
        except NeedleResetValidityError as exc:
            logger.warning("episode %d skipped: %s", episode, exc)
            results.append({"episode": episode, "outcome": "reset_invalid",
                            "detail": str(exc)})
            continue

        summary = env.run_grasp_lift(
            approach_controller=controller,
            max_cycles=args.max_cycles,
            verbose=not args.quiet,
        )
        record = {
            "episode": episode,
            "outcome": summary["reason"],
            "phase": summary["phase"],
            "cycles": summary["steps"],
            "needle_grasped": summary["needle_grasped_ground_truth"],
            "succeeded": env.succeeded(),
            # what the real deployment would have seen instead of ground truth
            "jaw_evidence": summary["jaw_evidence"],
        }
        results.append(record)
        logger.info(
            "episode %d: %s (grasped=%s, %d cycles)",
            episode, record["outcome"], record["needle_grasped"], record["cycles"],
        )

    attempted = [r for r in results if r.get("outcome") != "reset_invalid"]
    succeeded = [r for r in attempted if r.get("succeeded")]
    print()
    print(f"episodes attempted : {len(attempted)}")
    print(f"grasped and lifted : {len(succeeded)}")
    if attempted:
        print(f"success rate       : {100.0 * len(succeeded) / len(attempted):.1f}%")
    print(
        "\nnote: 'grasped' here is the AMBF finger ghost sensor. The real "
        "deployment has no such signal and reports grasp_verified=False."
    )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {
                "args": vars(args),
                "controller": controller.describe(),
                "results": results,
            },
            indent=2, default=str,
        ))
        print(f"wrote {args.json_out}")

    return 0 if attempted and len(succeeded) == len(attempted) else 1


if __name__ == "__main__":
    sys.exit(main())
