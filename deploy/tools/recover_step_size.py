#!/usr/bin/env python3
"""Recover the scale the *demonstrations* were collected at.

READ THIS BEFORE USING THE NUMBER
---------------------------------
This tool answers one narrow question exactly: at what step size do the
transitions embedded in a checkpoint integrate?  Solving

    achieved[t+1] = achieved[t] + action[t] * step_size

per channel recovers that to numerical zero.

**That is not necessarily the scale the policy acts at, and for the released
SurgicAI checkpoints it is not.**  SurgicAI carries two:

======================================  =======================  ==============
source                                  Approach                 used for
======================================  =======================  ==============
``RL/Env_info/Approach_noise_env_info``  0.5 mm / 2 deg          the demos
``RL/RL_training_online.py`` (and        1.0 mm / 3 deg,         training AND
``RL/Model_evaluation.py``)              300 max steps           evaluation
======================================  =======================  ==============

Both scripts hard-code ``trans_step = 1.0e-3`` and ``angle_step = deg2rad(3)``.
The published 96% +- 6% was measured at that scale, so that is the scale the
actor's outputs mean.  Replaying the upstream Approach checkpoint from its own
demonstration starts, through the deployment loop, against a perfect arm:

    0.5 mm / 2 deg  (what this tool recovers)     7/20    35%
    1.0 mm / 3 deg  (what training used)         19/20    95%

So: use this tool to confirm the demonstrations parse and to detect a
checkpoint whose demo set was collected under a different regime.  Use
``tools/replay_demos.py --compare`` to decide what to actually drive the policy
with -- it runs the policy instead of curve-fitting its training data.

    python3 tools/recover_step_size.py --model <checkpoint>
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

from surgicai_rl_deploy.contract import GOAL_SCALE, STEP_SIZE_RAW

CHANNELS = ("x", "y", "z", "roll", "pitch", "yaw", "jaw")


def unpickle(blob: str):
    import torch
    import torch.storage

    torch.storage._load_from_bytes = lambda b: torch.load(
        io.BytesIO(b), map_location="cpu", weights_only=False
    )
    return pickle.loads(base64.b64decode(blob))


def load_demos(model_path: Path):
    """Return ``(actions, achieved_raw, desired_raw)`` from the embedded demos."""
    with zipfile.ZipFile(model_path) as zf:
        data = json.loads(zf.read("data").decode("utf-8"))
    if "demo_data" not in data:
        raise SystemExit(
            f"{model_path.name} embeds no demonstration set, so its action scale "
            "cannot be recovered from it. Supply the scale from the training "
            "configuration instead (for SurgicAI, RL/Env_info/<task>_env_info)."
        )
    blob = data["demo_data"]
    payload = unpickle(blob[":serialized:"]) if isinstance(blob, dict) else unpickle(blob)
    actions = np.asarray(payload["actions"], dtype=np.float64)
    observations = payload["observations"]
    if isinstance(observations, np.ndarray) and observations.shape == ():
        observations = observations.item()
    achieved = np.asarray(observations["achieved_goal"], dtype=np.float64) / GOAL_SCALE
    desired = np.asarray(observations["desired_goal"], dtype=np.float64) / GOAL_SCALE
    return actions, achieved, desired


def episode_bounds(desired: np.ndarray) -> np.ndarray:
    """Episodes are delimited by a change in the (frozen) desired goal."""
    change = np.where(np.any(np.abs(np.diff(desired, axis=0)) > 1e-9, axis=1))[0] + 1
    return np.concatenate([[0], change, [len(desired)]])


def recover(actions, achieved, desired, *, min_action=0.05):
    """Per-channel least-squares solve of delta = action * step."""
    bounds = episode_bounds(desired)
    deltas, acts = [], []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo < 2:
            continue
        deltas.append(achieved[lo + 1 : hi] - achieved[lo : hi - 1])
        acts.append(actions[lo : hi - 1])
    if not deltas:
        raise SystemExit("no usable transitions in the demonstration set")
    delta = np.concatenate(deltas)
    action = np.concatenate(acts)

    step = np.zeros(7)
    used = np.zeros(7, dtype=int)
    for c in range(7):
        mask = np.abs(action[:, c]) >= min_action
        used[c] = int(mask.sum())
        if used[c] < 10:
            step[c] = np.nan
            continue
        # least squares through the origin: step = (a . d) / (a . a)
        a, d = action[mask, c], delta[mask, c]
        step[c] = float(np.dot(a, d) / np.dot(a, a))
    return step, action, delta, used, len(bounds) - 1


def residual(step, action, delta):
    predicted = action * step
    trans = np.linalg.norm(predicted[:, :3] - delta[:, :3], axis=1) * 1000.0
    rot = np.degrees(np.linalg.norm(predicted[:, 3:6] - delta[:, 3:6], axis=1))
    jaw = np.abs(predicted[:, 6] - delta[:, 6])
    return trans, rot, jaw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--min-action", type=float, default=0.05,
                    help="ignore transitions whose action on a channel is "
                         "smaller than this, where the ratio is ill-conditioned")
    ap.add_argument("--tolerance-mm", type=float, default=0.01,
                    help="residual below which the recovered scale is called exact")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    path = Path(args.model).expanduser()
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")

    actions, achieved, desired = load_demos(path)
    step, action, delta, used, episodes = recover(
        actions, achieved, desired, min_action=args.min_action
    )

    print(f"checkpoint : {path.name}")
    print(f"demos      : {len(actions)} transitions over {episodes} episodes")
    print()
    print(f"{'channel':<8}{'recovered':>14}{'in package':>14}{'ratio':>9}{'samples':>9}")
    print("-" * 54)
    for c, name in enumerate(CHANNELS):
        pkg = float(STEP_SIZE_RAW[c])
        rec = step[c]
        if c < 3:
            shown, pkg_shown, unit = rec * 1000.0, pkg * 1000.0, "mm"
        elif c < 6:
            shown, pkg_shown, unit = np.degrees(rec), np.degrees(pkg), "deg"
        else:
            shown, pkg_shown, unit = rec, pkg, ""
        ratio = rec / pkg if pkg else float("nan")
        print(f"{name:<8}{shown:>11.4f} {unit:<2}{pkg_shown:>11.4f} {unit:<2}"
              f"{ratio:>9.3f}{used[c]:>9}")

    trans, rot, jaw = residual(step, action, delta)
    pkg_trans, pkg_rot, _ = residual(np.asarray(STEP_SIZE_RAW, dtype=np.float64),
                                     action, delta)
    print()
    print(f"{'':<22}{'median':>12}{'p95':>12}")
    print(f"recovered  trans      {np.median(trans):>10.4f} mm{np.percentile(trans,95):>9.4f} mm")
    print(f"recovered  rot        {np.median(rot):>10.4f} de{np.percentile(rot,95):>9.4f} de")
    print(f"package    trans      {np.median(pkg_trans):>10.4f} mm{np.percentile(pkg_trans,95):>9.4f} mm")
    print(f"package    rot        {np.median(pkg_rot):>10.4f} de{np.percentile(pkg_rot,95):>9.4f} de")
    print()

    exact = np.median(trans) < args.tolerance_mm
    matches_package = np.allclose(step[:6], np.asarray(STEP_SIZE_RAW)[:6], rtol=1e-3)

    if exact:
        print("The demonstrations integrate cleanly at the recovered scale, so")
        print("the demo set parses and is self-consistent.")
        if matches_package:
            print("It also equals the scale this package would act with.")
        else:
            print("It is NOT the scale this package acts with -- which may be")
            print("perfectly correct. For the upstream SurgicAI checkpoints the")
            print("demonstrations were collected at 0.5 mm / 2 deg while the")
            print("policy was trained and evaluated at 1.0 mm / 3 deg")
            print("(RL/RL_training_online.py, RL/Model_evaluation.py).")
        print()
        print("DO NOT pass the recovered numbers as --trans-step-mm on the")
        print("strength of this result alone. Decide the acting scale with:")
        print("    python3 tools/replay_demos.py --model <checkpoint> --compare")
        print("which runs the policy instead of fitting its training data.")
        code = 0 if matches_package else 2
    else:
        print("The recovered scale does not reproduce the demonstrations exactly,")
        print("so the stored transitions are not a clean integration of the stored")
        print("actions. The demo set was probably collected under a different")
        print("regime than it is stored against. Find the training configuration")
        print("and confirm it with tools/replay_demos.py.")
        code = 3

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "checkpoint": str(path),
            "transitions": int(len(actions)),
            "episodes": int(episodes),
            "recovered_step_size": step.tolist(),
            "package_step_size": np.asarray(STEP_SIZE_RAW, dtype=float).tolist(),
            "recovered_residual_trans_mm_median": float(np.median(trans)),
            "package_residual_trans_mm_median": float(np.median(pkg_trans)),
            "exact": bool(exact),
            "matches_package": bool(matches_package),
        }, indent=2))
        print(f"\nwrote {args.json_out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
