#!/usr/bin/env python3
"""Read a checkpoint's training support back out of its own demonstrations.

Everything in :mod:`surgicai_rl_deploy.contract` that describes *where a policy
was trained* -- the goal box, the start-offset box in the tool frame, the wrist
rotation range, the jaw envelope, the episode length -- was produced by this.
Run it on any SurgicAI checkpoint and paste the block it prints.

    python3 tools/profile_checkpoint.py --model <checkpoint>

It also reports the roll branch statistics, which is how the branch defect was
found in the first place: for both released checkpoints, 100% of the stored
goal rolls lie outside scipy's canonical [-pi, pi] and 0% lie outside
SurgicAI's (-2*pi, 0].  A checkpoint that fails the second test was produced by
something other than ``Frame2Vec(bound=True)`` and none of this package's
assumptions apply to it.

What it cannot tell you
-----------------------
The scale the policy *acts* at.  The demonstrations integrate at the scale in
``RL/Env_info``; the policy was trained at the scale in
``RL/RL_training_online.py``, and for the released checkpoints those differ
(0.5 mm / 2 deg against 1.0 mm / 3 deg).  Use ``tools/replay_demos.py`` to
settle that -- it runs the policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recover_step_size import episode_bounds, load_demos  # noqa: E402

from surgicai_rl_deploy.contract import (  # noqa: E402
    OTHER_UPSTREAM_CHECKPOINTS,
    contract_for_digest,
)


def recover_scale(actions, achieved, bounds, min_action=0.05):
    deltas, acts = [], []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo < 2:
            continue
        deltas.append(achieved[lo + 1:hi] - achieved[lo:hi - 1])
        acts.append(actions[lo:hi - 1])
    delta, action = np.concatenate(deltas), np.concatenate(acts)
    step = np.full(7, np.nan)
    for c in range(7):
        mask = np.abs(action[:, c]) >= min_action
        if mask.sum() >= 10:
            a, d = action[mask, c], delta[mask, c]
            step[c] = float(np.dot(a, d) / np.dot(a, a))
    residual = np.linalg.norm(
        (action * np.nan_to_num(step, nan=0.0))[:, :3] - delta[:, :3], axis=1
    ) * 1000.0
    return step, float(np.median(residual))


def profile(path: Path, device: str = "cpu") -> dict:
    from surgicai_rl_deploy.policy import sha256_of

    digest = sha256_of(path)
    actions, achieved, desired = load_demos(path)
    bounds = episode_bounds(desired)
    n = len(bounds) - 1

    step, residual = recover_scale(actions, achieved, bounds)

    starts = np.array([achieved[bounds[i]] for i in range(n)])
    goals = np.array([desired[bounds[i]] for i in range(n)])
    lengths = np.diff(bounds)

    offsets, rots, rotvecs, travel = [], [], [], []
    for s, g in zip(starts, goals):
        Rs = Rotation.from_euler("xyz", s[3:6]).as_matrix()
        Rg = Rotation.from_euler("xyz", g[3:6]).as_matrix()
        dp = (g[:3] - s[:3]) * 100.0
        offsets.append(Rs.T @ dp)
        rel = Rotation.from_matrix(Rs.T @ Rg)
        rots.append(np.degrees(rel.magnitude()))
        rotvecs.append(rel.as_rotvec())
        travel.append(np.linalg.norm(dp))
    offsets = np.array(offsets)
    rots = np.array(rots)

    def outside_bound(col):
        return float(np.mean((col > 0.0) | (col <= -2 * np.pi)) * 100.0)

    return {
        "path": str(path),
        "sha256": digest,
        "known_contract": (
            c.name if (c := contract_for_digest(digest)) else
            OTHER_UPSTREAM_CHECKPOINTS.get(digest)
        ),
        "transitions": int(len(actions)),
        "episodes": int(n),
        "demo_step_size": step.tolist(),
        "demo_step_residual_mm": residual,
        "roll_outside_pi_desired_pct": float(np.mean(np.abs(desired[:, 3]) > np.pi) * 100),
        "roll_outside_pi_achieved_pct": float(np.mean(np.abs(achieved[:, 3]) > np.pi) * 100),
        "roll_outside_surgicai_bound_desired_pct": outside_bound(desired[:, 3]),
        "roll_outside_surgicai_bound_achieved_pct": outside_bound(achieved[:, 3]),
        "trained_goal_vec7": goals.mean(axis=0).tolist(),
        "goal_min_cm": (goals[:, :3].min(axis=0) * 100.0).tolist(),
        "goal_max_cm": (goals[:, :3].max(axis=0) * 100.0).tolist(),
        "start_offset_tool_min": offsets.min(axis=0).tolist(),
        "start_offset_tool_max": offsets.max(axis=0).tolist(),
        "start_offset_tool_mean": offsets.mean(axis=0).tolist(),
        "start_rot_deg_min": float(rots.min()),
        "start_rot_deg_max": float(rots.max()),
        "start_rot_deg_median": float(np.median(rots)),
        "start_to_goal_rotvec_mean": np.array(rotvecs).mean(axis=0).tolist(),
        "demo_start_jaw": float(np.median(starts[:, 6])),
        "demo_goal_jaw": float(np.median(goals[:, 6])),
        "demo_episode_steps": int(np.median(lengths)),
        "demo_travel_cm": float(np.median(travel)),
        "unique_starts": int(len(np.unique(np.round(starts, 6), axis=0))),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, nargs="+")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    out = []
    for name in args.model:
        path = Path(name).expanduser()
        if not path.is_file():
            raise SystemExit(f"checkpoint not found: {path}")
        p = profile(path)
        out.append(p)

        print("=" * 72)
        print(f"{path}")
        print(f"sha256           : {p['sha256'][:32]}")
        print(f"known as         : {p['known_contract'] or 'UNKNOWN to this package'}")
        print(f"demonstrations   : {p['transitions']} transitions, {p['episodes']} "
              f"episodes, {p['unique_starts']} distinct start pose(s)")
        print(f"demo step size   : "
              f"{np.asarray(p['demo_step_size'][:3]) * 1000} mm / "
              f"{np.degrees(p['demo_step_size'][3:6])} deg / "
              f"jaw {p['demo_step_size'][6]}")
        print(f"  residual       : {p['demo_step_residual_mm']:.2e} mm  "
              f"({'clean' if p['demo_step_residual_mm'] < 1e-3 else 'NOT a clean integration'})")
        print(f"roll outside +-pi: desired {p['roll_outside_pi_desired_pct']:.0f}%  "
              f"achieved {p['roll_outside_pi_achieved_pct']:.0f}%")
        print(f"roll outside     : desired "
              f"{p['roll_outside_surgicai_bound_desired_pct']:.0f}%  achieved "
              f"{p['roll_outside_surgicai_bound_achieved_pct']:.0f}%   "
              "(SurgicAI's (-2pi, 0] -- both must be 0)")
        print()
        print("    trained_goal_vec7=np.array(")
        print(f"        {np.round(p['trained_goal_vec7'], 6).tolist()}),")
        print(f"    goal_min_cm=np.array({np.round(p['goal_min_cm'], 3).tolist()}),")
        print(f"    goal_max_cm=np.array({np.round(p['goal_max_cm'], 3).tolist()}),")
        print("    start_offset_tool_min=np.array("
              f"{np.round(p['start_offset_tool_min'], 3).tolist()}),")
        print("    start_offset_tool_max=np.array("
              f"{np.round(p['start_offset_tool_max'], 3).tolist()}),")
        print("    start_offset_tool_mean=np.array("
              f"{np.round(p['start_offset_tool_mean'], 3).tolist()}),")
        print(f"    start_rot_deg_min={p['start_rot_deg_min']:.1f},")
        print(f"    start_rot_deg_max={p['start_rot_deg_max']:.1f},")
        print(f"    start_rot_deg_median={p['start_rot_deg_median']:.1f},")
        print("    start_to_goal_rotvec_mean=np.array("
              f"{np.round(p['start_to_goal_rotvec_mean'], 4).tolist()}),")
        print(f"    demo_start_jaw={p['demo_start_jaw']:.2f},")
        print(f"    demo_goal_jaw={p['demo_goal_jaw']:.2f},")
        print(f"    demo_episode_steps={p['demo_episode_steps']},")
        print(f"    demo_travel_cm={p['demo_travel_cm']:.2f},")
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
