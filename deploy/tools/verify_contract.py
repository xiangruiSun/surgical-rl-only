#!/usr/bin/env python3
"""Prove the observation builder matches the checkpoint's own stored observations.

The released zip carries ``_last_obs`` / ``_last_original_obs`` (the final
training observation) and the full embedded demonstration set.  This script
rebuilds those 21-dim vectors from ``achieved_goal`` + ``desired_goal`` with
:func:`surgicai_rl_deploy.obs.build_observation` and asserts an exact match, so
a wrong scaling or a wrong concatenation order cannot pass silently.

    python tools/verify_contract.py --model .../r6_...zip
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.contract import GOAL_SCALE
from surgicai_rl_deploy.obs import build_observation


def _load_data_blob(model_path: Path) -> dict:
    with zipfile.ZipFile(model_path) as zf:
        return json.loads(zf.read("data").decode("utf-8"))


def _unpickle(blob: str):
    import torch
    import torch.storage

    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    return pickle.loads(base64.b64decode(blob))


def check_pair(achieved_scaled, desired_scaled, observation_scaled, label, failures):
    # Our builder takes RAW units, so undo the cm scaling first.
    raw_a = np.asarray(achieved_scaled, dtype=np.float64) / GOAL_SCALE
    raw_d = np.asarray(desired_scaled, dtype=np.float64) / GOAL_SCALE
    rebuilt = build_observation(raw_a, raw_d)
    for key, expected in (
        ("achieved_goal", achieved_scaled),
        ("desired_goal", desired_scaled),
        ("observation", observation_scaled),
    ):
        got = rebuilt[key]
        if not np.allclose(got, np.asarray(expected, dtype=np.float32), atol=1e-4):
            failures.append((label, key, np.asarray(expected), got))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--demo-samples", type=int, default=500)
    args = ap.parse_args()

    path = Path(args.model).expanduser()
    data = _load_data_blob(path)
    failures = []
    checked = 0

    for key in ("_last_obs", "_last_original_obs"):
        if key not in data:
            continue
        obs = _unpickle(data[key][":serialized:"])
        check_pair(
            np.asarray(obs["achieved_goal"]).ravel(),
            np.asarray(obs["desired_goal"]).ravel(),
            np.asarray(obs["observation"]).ravel(),
            key,
            failures,
        )
        checked += 1

    if "demo_data" in data:
        demo = _unpickle(data["demo_data"][":serialized:"])
        o = demo["observations"]
        a = np.asarray(o["achieved_goal"])
        d = np.asarray(o["desired_goal"])
        obs = np.asarray(o["observation"])
        n = min(args.demo_samples, len(a))
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        for i in idx:
            check_pair(a[i], d[i], obs[i], f"demo[{i}]", failures)
        checked += n

    print(f"checked {checked} stored observations")
    if failures:
        print(f"FAIL: {len(failures)} mismatches")
        label, key, expected, got = failures[0]
        np.set_printoptions(precision=5, suppress=True)
        print(f"  first mismatch: {label} / {key}")
        print(f"    stored : {expected}")
        print(f"    rebuilt: {got}")
        return 1
    print("PASS: observation contract reproduced exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
