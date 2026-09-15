"""Loading the released SB3 checkpoints without the training repo.

Two things in the released zips would otherwise break a clean deployment host:

1. ``replay_buffer_class`` pickles ``RL_algo.DemoHerReplayBuffer``, which only
   exists inside the training tree.
2. ``demo_data`` holds CUDA tensors, so loading on a CPU-only box raises.

Both are training-time state that the actor never touches at inference, so we
hand SB3 ``custom_objects`` overrides for them.  SB3 substitutes those keys
*before* unpickling, so neither is ever deserialised.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from .contract import CHECKPOINT_CONTRACTS, CONTRACTS, OTHER_UPSTREAM_CHECKPOINTS

# Published SHA256 digests (models/rl/MANIFEST.sha256) plus every upstream
# checkpoint this package has a contract for, so a released file is recognised
# rather than needing --allow-unknown-model.
KNOWN_CHECKPOINTS = {
    "0407987e296d78b8b63ccf49c16e35395b00cf8d4ebc4cfe857b57f3381f2a2f": (
        "m3_measured_r3_100k"
    ),
    "6286a88c21f04abfbc4b0747a87a67bc2c5dcba17f710692c6b5138f7776e525": (
        "r6_unified_single_goal_yaw15_seed1_final"
    ),
    **{d: CONTRACTS[k].name for d, k in CHECKPOINT_CONTRACTS.items()},
    **OTHER_UPSTREAM_CHECKPOINTS,
}

_SAFE_CUSTOM_OBJECTS = {
    "replay_buffer_class": None,
    "replay_buffer_kwargs": {},
    "demo_data": None,
    "action_noise": None,
    "lr_schedule": lambda _progress: 0.0,
    "train_freq": (1, "episode"),
}


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ApproachPolicy:
    """Thin deterministic-inference wrapper around the TD3 actor."""

    def __init__(self, model, checkpoint_path: Path, digest: str, identity: str):
        self.model = model
        self.checkpoint_path = Path(checkpoint_path)
        self.sha256 = digest
        self.identity = identity

    @classmethod
    def load(cls, checkpoint_path, device: str = "cpu", verify: bool = True):
        # Check the path before importing torch: a typo used to surface as a
        # ModuleNotFoundError for stable_baselines3, which sends you off
        # installing two gigabytes to fix a wrong filename.
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            hint = ""
            if not path.is_absolute():
                hint = (
                    f"\n(resolved against {Path.cwd()}; the checkpoint usually "
                    "lives in models/rl/ at the repository root, so from "
                    "deploy/ the path is ../models/rl/...)"
                )
            raise FileNotFoundError(f"checkpoint not found: {path}{hint}")

        try:
            from stable_baselines3 import TD3  # imported late: heavy
        except ImportError as exc:
            raise SystemExit(
                "stable-baselines3 is required for --controller rl/residual but "
                "is not installed in this interpreter "
                f"({sys.executable}).\n"
                "    python3 -m venv --system-site-packages .venv-deploy\n"
                "    source .venv-deploy/bin/activate\n"
                "    pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
                "    pip install 'stable-baselines3>=2.0,<3' 'gymnasium>=0.29'\n"
                "The CPU wheel is enough: the actor is a 3x256 MLP.\n"
                "--controller d2 needs none of this."
            ) from exc

        digest = sha256_of(path)
        identity = KNOWN_CHECKPOINTS.get(digest, "UNKNOWN")
        if verify and identity == "UNKNOWN":
            raise ValueError(
                f"{path.name} has SHA256 {digest}, which is not one of the "
                "released checkpoints. Re-download, or pass verify=False and "
                "record the digest and training contract yourself."
            )

        model = TD3.load(str(path), device=device, custom_objects=dict(_SAFE_CUSTOM_OBJECTS))
        obs_space = model.observation_space
        if sorted(obs_space.spaces) != ["achieved_goal", "desired_goal", "observation"]:
            raise ValueError(f"unexpected observation space: {obs_space}")
        if obs_space["observation"].shape != (21,):
            raise ValueError("expected a 21-dim observation block")
        if model.action_space.shape != (7,):
            raise ValueError("expected a 7-dim action space")
        return cls(model, path, digest, identity)

    def act(self, observation: dict) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(7)

    def describe(self) -> str:
        return f"{self.checkpoint_path.name} [{self.identity}] sha256={self.sha256[:12]}…"
