from __future__ import annotations

import argparse


def add_common_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path.",
    )


def add_task_view_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task_name", type=str, required=True, help="Task/environment name")
    parser.add_argument("--view_name", type=str, required=True, help="Camera/view name")


def add_threshold_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trans_error", type=float, required=True, help="Translational error threshold (cm)")
    parser.add_argument("--angle_error", type=float, required=True, help="Angular error threshold (deg)")


def add_seed_arg(parser: argparse.ArgumentParser, *, name: str = "--seed", default: int = 10) -> None:
    parser.add_argument(name, type=int, default=default, help="Random seed")


def add_experiment_variant_arg(parser: argparse.ArgumentParser) -> None:
    """
    Standard experiment naming variant.

    If set, this overrides any derived naming from other flags (e.g. stepDR or randomization params).
    """
    parser.add_argument(
        "--variant",
        type=str,
        choices=["base_env", "stepDR", "randomization"],
        default=None,
        help="Experiment variant name (overrides auto-derived variant if provided).",
    )

