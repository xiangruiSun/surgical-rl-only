#!/usr/bin/env python3
"""Print what a released checkpoint actually contains: identity, spaces, and
the goal region it was trained on.

    python tools/inspect_checkpoint.py --model .../r6_...zip
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
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.policy import KNOWN_CHECKPOINTS, sha256_of


def _unpickle(blob):
    import torch
    import torch.storage

    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    return pickle.loads(base64.b64decode(blob))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    path = Path(args.model).expanduser()

    digest = sha256_of(path)
    print(f"file    : {path.name}")
    print(f"sha256  : {digest}")
    print(f"identity: {KNOWN_CHECKPOINTS.get(digest, 'UNKNOWN - not a released checkpoint')}")

    with zipfile.ZipFile(path) as zf:
        data = json.loads(zf.read("data").decode("utf-8"))
    print(f"algo    : TD3 + HER + BC, policy={data.get('policy_class', {}).get('__module__', '?')}")
    print(f"net_arch: {data.get('policy_kwargs')}")
    print(f"steps   : {data.get('num_timesteps')}  seed={data.get('seed')}  gamma={data.get('gamma')}")
    print(f"BC_coeff: {data.get('BC_coeff')}  demo_ratio={data.get('demo_ratio')}")

    if "demo_data" not in data:
        return 0
    demo = _unpickle(data["demo_data"][":serialized:"])
    o = demo["observations"]
    a = np.asarray(o["achieved_goal"])
    d = np.asarray(o["desired_goal"])
    starts = [0] + [i for i in range(1, len(d)) if not np.allclose(d[i], d[i - 1])]
    S = np.array(starts)
    a0, g0 = a[S], d[S]

    np.set_printoptions(precision=3, suppress=True)
    print(f"\ndemonstrations: {len(a)} transitions over {len(S)} episodes")
    print("units: xyz in cm, rpy in rad (extrinsic xyz), jaw normalised 0..1\n")
    print("desired_goal support")
    print(f"  min  {d.min(0)}")
    print(f"  max  {d.max(0)}")
    print(f"  mean {d.mean(0)}")

    dp = g0[:, :3] - a0[:, :3]
    Rs = Rotation.from_euler("xyz", a0[:, 3:6])
    dp_tool = np.einsum("nij,nj->ni", Rs.as_matrix().transpose(0, 2, 1), dp)
    rel = Rs.inv() * Rotation.from_euler("xyz", g0[:, 3:6])
    print("\nstart -> goal displacement (cm)")
    print(f"  policy frame mean {dp.mean(0)}   |d| p50 {np.median(np.linalg.norm(dp, axis=1)):.2f}")
    print(f"  tool frame   min  {dp_tool.min(0)}")
    print(f"  tool frame   max  {dp_tool.max(0)}")
    print(f"  tool frame   mean {dp_tool.mean(0)}")
    mags = np.degrees(rel.magnitude())
    print(f"\nstart -> goal rotation (deg): min {mags.min():.1f}  median "
          f"{np.median(mags):.1f}  max {mags.max():.1f}")
    print(f"jaw: starts at {a0[:, 6].mean():.2f}, goal is {g0[:, 6].mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
