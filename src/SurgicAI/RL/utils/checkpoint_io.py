"""Checkpoint path and legacy-module compatibility helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from RL.rl_paths import repo_dir, rl_dir


def register_legacy_checkpoint_modules() -> None:
    """Expose only the historical module names serialized in old SB3 ZIPs."""
    package = importlib.import_module("RL.RL_algo")
    demo_module = importlib.import_module("RL.RL_algo.DemoHerReplayBuffer")
    her_module = importlib.import_module("RL.RL_algo.HerReplayBuffer")
    sys.modules.setdefault("RL_algo", package)
    sys.modules.setdefault("RL_algo.DemoHerReplayBuffer", demo_module)
    sys.modules.setdefault("RL_algo.HerReplayBuffer", her_module)


def _with_optional_zip(path: Path) -> list[Path]:
    candidates = [path]
    if path.suffix.lower() != ".zip":
        candidates.append(path.with_suffix(".zip"))
    return candidates


def resolve_checkpoint_path(model_path: str | Path) -> Path:
    """Resolve a checkpoint independently of whether execution starts at repo/RL/CWD."""
    requested = Path(model_path).expanduser()
    roots = [Path.cwd(), repo_dir(), rl_dir()]
    raw_candidates = [requested] if requested.is_absolute() else [root / requested for root in roots]

    checked: list[Path] = []
    for raw in raw_candidates:
        for candidate in _with_optional_zip(raw):
            resolved = candidate.resolve(strict=False)
            if resolved in checked:
                continue
            checked.append(resolved)
            if resolved.is_file():
                return resolved

    attempted = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(f"Checkpoint not found. Tried:\n  - {attempted}")


def load_sb3_checkpoint(model_class: Any, model_path: str | Path, env: Any = None) -> Any:
    """Load an SB3 checkpoint after resolving its path and legacy pickle modules."""
    resolved = resolve_checkpoint_path(model_path)
    register_legacy_checkpoint_modules()
    return model_class.load(str(resolved), env=env)
