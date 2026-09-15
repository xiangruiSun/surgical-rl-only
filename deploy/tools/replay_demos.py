#!/usr/bin/env python3
"""Replay a checkpoint against its own demonstrations, through the deploy loop.

Why this is the test that matters
---------------------------------
``verify_contract.py`` proves the observation *builder* is byte-exact against
the vectors stored in a checkpoint.  ``recover_step_size.py`` proves the action
*scale* reproduces the stored transitions.  Neither runs the policy.

This does.  For each embedded demonstration episode it takes the first achieved
state as the start, the frozen desired goal as the goal, and drives the real
:class:`~surgicai_rl_deploy.loop.ApproachLoop` -- the same object the robot runs
-- against a perfect kinematic arm, in the training frame, with no bridge.

Under those conditions the loop *is* the training environment: SurgicAI's
``subtask_env.step`` integrates the command and never reads the arm back, so a
perfect arm and an open-loop observation are the same dynamics.  If the loop is
faithful, the policy must reproduce roughly its published success rate.  If it
does not, the gap is in this package, not in the policy or the hardware -- and
each fidelity switch can be turned off one at a time to find out which.

    python3 tools/replay_demos.py --model <checkpoint>
    python3 tools/replay_demos.py --model <checkpoint> --compare

``--compare`` runs the matrix of conventions and prints one line each, which is
the fastest way to see what a given defect costs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recover_step_size import episode_bounds, load_demos  # noqa: E402

from surgicai_rl_deploy.contract import (  # noqa: E402
    CONTRACTS,
    contract_for_digest,
)
from surgicai_rl_deploy.controllers import D2Controller, RLController  # noqa: E402
from surgicai_rl_deploy.frames import Pose, bound_roll  # noqa: E402
from surgicai_rl_deploy.loop import ApproachLoop, LoopConfig, SafetyLimits  # noqa: E402

#: Training applied no safety envelope at all.  Replaying with the deployment
#: clamps switched on measures what the envelope costs; this is the "off".
PERMISSIVE = SafetyLimits(
    workspace_pad_cm=1000.0,
    max_step_translation_mm=1000.0,
    max_step_rotation_deg=360.0,
    max_tracking_error_cm=1000.0,
    max_consecutive_clamps=0,
)


def episodes(actions, achieved, desired):
    bounds = episode_bounds(desired)
    out = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi - lo < 2:
            continue
        out.append({
            "start_vec7": achieved[lo].copy(),
            "goal_vec7": desired[lo].copy(),
            "demo_steps": int(hi - lo),
            "demo_actions": actions[lo:hi].copy(),
        })
    return out


def lagging_arm(measured: Pose, command: Pose, alpha: float) -> Pose:
    """First-order arm: it moves a fraction ``alpha`` of the way each cycle.

    ``alpha = 1`` is the perfect arm the training environment implicitly
    assumed -- in simulation the observation *is* the command, so there is no
    arm at all.  A real PSM follows ``servo_cp`` with a lag that depends on
    rate, cable tension and how far it was asked to go; anything below 1 is a
    step in that direction.  This is the only way an offline replay can tell
    the open-loop observation contract apart from the closed-loop one, because
    with a perfect arm the two are numerically identical.
    """
    from scipy.spatial.transform import Rotation

    p = measured.p + alpha * (command.p - measured.p)
    rotvec = Rotation.from_matrix(measured.R.T @ command.R).as_rotvec()
    R = measured.R @ Rotation.from_rotvec(rotvec * alpha).as_matrix()
    return Pose(p, R, command.jaw)


def run_episode(controller, ep, contract, *, rpy_convention, observation_source,
                step_size, limits, max_steps, rot_metric, jaw_from_demo=True,
                alpha: float = 1.0):
    start = Pose.from_vec7(ep["start_vec7"])
    goal_vec7 = ep["goal_vec7"]
    goal_pose = Pose.from_vec7(goal_vec7)

    cfg = LoopConfig(
        frame_mode="identity",
        goal_orientation="explicit",
        goal_quat_xyzw=tuple(goal_pose.quat_xyzw()),
        goal_jaw=str(float(goal_vec7[6])),
        use_policy_jaw=False,
        max_steps=max_steps,
        success_trans_cm=contract.success_trans_cm,
        success_rot_rad=contract.success_rot_rad,
        step_size=np.asarray(step_size, dtype=np.float64),
        rpy_convention=rpy_convention,
        observation_source=observation_source,
        rot_metric=rot_metric,
        policy_jaw_start=(float(ep["start_vec7"][6]) if jaw_from_demo else None),
        # The demonstrations carry the training branch already, so nothing has
        # to be inferred -- but only the non-canonical conventions may use it.
        goal_rpy_train=tuple(goal_vec7[3:6]),
    )
    loop = ApproachLoop(controller, cfg, limits, contract=contract)
    loop.begin(start, goal_vec7[:3])

    measured = start
    closest = np.inf
    closest_rot = np.inf
    result = None
    for _ in range(max_steps):
        result = loop.step(measured)
        closest = min(closest, result.trans_err_cm)
        closest_rot = min(closest_rot, result.rpy_norm_err_rad)
        if result.done:
            break
        measured = (
            result.command if alpha >= 1.0
            else lagging_arm(measured, result.command, alpha)
        )
    return {
        "success": result.reason == "success",
        "outcome": result.reason,
        "steps": int(result.index),
        "final_trans_cm": float(result.trans_err_cm),
        "final_rpy_norm_deg": float(np.degrees(result.rpy_norm_err_rad)),
        "closest_trans_cm": float(closest),
        "closest_rpy_norm_deg": float(np.degrees(closest_rot)),
        "demo_steps": ep["demo_steps"],
    }


def summarise(rows):
    ok = [r for r in rows if r["success"]]
    steps = [r["steps"] for r in ok]
    return {
        "episodes": len(rows),
        "success": len(ok),
        "rate": len(ok) / len(rows) if rows else 0.0,
        "median_steps": float(np.median(steps)) if steps else float("nan"),
        "median_closest_cm": float(np.median([r["closest_trans_cm"] for r in rows])),
        "median_final_rot_deg": float(np.median([r["final_rpy_norm_deg"] for r in rows])),
    }


def check_branch_recovery(eps) -> dict:
    """Does bound_roll() reproduce the stored branch without being told it?"""
    goal_ok = achieved_ok = 0
    for ep in eps:
        for key, hits in (("goal_vec7", "goal"), ("start_vec7", "start")):
            stored = ep[key][3:6]
            recovered = bound_roll(Pose.from_vec7(ep[key]).to_vec7()[3:6])
            if np.allclose(recovered, stored, atol=1e-5):
                if hits == "goal":
                    goal_ok += 1
                else:
                    achieved_ok += 1
    return {"goals": goal_ok, "starts": achieved_ok, "of": len(eps)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--contract", choices=sorted(CONTRACTS),
                    help="override the contract normally chosen by SHA256")
    ap.add_argument("--controller", choices=["rl", "d2"], default="rl")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = the contract's")
    ap.add_argument("--rot-metric", choices=["rpy_norm", "geodesic"],
                    default="rpy_norm",
                    help="rpy_norm is SurgicAI's own success test")
    ap.add_argument("--rpy-convention",
                    choices=["surgicai_bound", "unwrap", "canonical"],
                    default="surgicai_bound")
    ap.add_argument("--observation-source", choices=["command", "measured"],
                    default="command")
    ap.add_argument("--trans-step-mm", type=float)
    ap.add_argument("--angle-step-deg", type=float)
    ap.add_argument("--arm-alpha", type=float, default=1.0,
                    help="first-order arm tracking fraction per cycle; 1.0 is "
                         "the perfect arm the training env implies, 0.3 is a "
                         "real servo that lags")
    ap.add_argument("--with-safety", action="store_true",
                    help="apply the deployment safety envelope during replay")
    ap.add_argument("--compare", action="store_true",
                    help="sweep the fidelity switches and print one row each")
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    path = Path(args.model).expanduser()
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")

    from surgicai_rl_deploy.policy import ApproachPolicy, sha256_of

    digest = sha256_of(path)
    contract = (
        CONTRACTS[args.contract] if args.contract else contract_for_digest(digest)
    )
    if contract is None:
        raise SystemExit(
            f"{path.name} (sha256 {digest[:12]}) has no registered contract.\n"
            "Pass --contract to pick one, after checking it really applies:\n"
            "  " + ", ".join(sorted(CONTRACTS))
        )

    actions, achieved, desired = load_demos(path)
    eps = episodes(actions, achieved, desired)
    if args.episodes:
        eps = eps[: args.episodes]

    step_size = np.asarray(contract.step_size, dtype=np.float64).copy()
    if args.trans_step_mm is not None:
        step_size[:3] = args.trans_step_mm * 1e-3
    if args.angle_step_deg is not None:
        step_size[3:6] = np.deg2rad(args.angle_step_deg)
    max_steps = args.max_steps or contract.max_steps
    limits = SafetyLimits() if args.with_safety else PERMISSIVE

    print(f"checkpoint : {path.name}")
    print(f"sha256     : {digest[:16]}")
    print(f"contract   : {contract.describe()}")
    print(f"episodes   : {len(eps)}   max steps {max_steps}   "
          f"safety envelope {'ON' if args.with_safety else 'off'}   "
          f"arm alpha {args.arm_alpha:g}")

    branch = check_branch_recovery(eps)
    print(f"branch     : bound_roll() reproduces {branch['goals']}/{branch['of']} "
          f"stored goal rolls and {branch['starts']}/{branch['of']} stored start "
          "rolls without being told the branch")
    print()

    if args.controller == "rl":
        policy = ApproachPolicy.load(str(path), device=args.device, verify=False)
        controller = RLController(policy)
    else:
        controller = D2Controller(staged=True, step_size=step_size)

    def sweep(rpy_convention, observation_source, label):
        rows = [
            run_episode(controller, ep, contract,
                        rpy_convention=rpy_convention,
                        observation_source=observation_source,
                        step_size=step_size, limits=limits,
                        max_steps=max_steps, rot_metric=args.rot_metric,
                        alpha=args.arm_alpha)
            for ep in eps
        ]
        s = summarise(rows)
        print(f"{label:<44}{s['success']:>3}/{s['episodes']:<4}"
              f"{100*s['rate']:>7.1f}%{s['median_steps']:>9.0f}"
              f"{s['median_closest_cm']:>10.3f}{s['median_final_rot_deg']:>10.2f}")
        return s, rows

    header = (f"{'configuration':<44}{'reproduced':>8}{'rate':>9}"
              f"{'steps':>9}{'closest':>10}{'rot deg':>10}")
    print(header)
    print("-" * len(header))

    results = {}
    if args.compare:
        for conv in ("surgicai_bound", "unwrap", "canonical"):
            for src in ("command", "measured"):
                s, _ = sweep(conv, src, f"roll={conv:<15} obs={src}")
                results[f"{conv}|{src}"] = s
    else:
        s, rows = sweep(args.rpy_convention, args.observation_source,
                        f"roll={args.rpy_convention:<15} obs={args.observation_source}")
        results["selected"] = s
        results["rows"] = rows

    print()
    best = max(
        (v for k, v in results.items() if isinstance(v, dict) and "rate" in v),
        key=lambda v: v["rate"],
    )
    if best["rate"] >= 0.80:
        print("The loop reproduces the checkpoint's own demonstrations. Whatever")
        print("goes wrong on the robot is downstream of the contract.")
    elif best["rate"] >= 0.30:
        print("Partial reproduction. The contract is closer than it was but")
        print("something still differs from training -- compare the rows above.")
    else:
        print("The loop does not reproduce the checkpoint's own demonstrations")
        print("even in the training frame against a perfect arm. Nothing measured")
        print("on the robot through this path means anything yet.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "checkpoint": str(path), "sha256": digest,
            "contract": contract.name,
            "step_size": step_size.tolist(),
            "results": results,
        }, indent=2, default=float))
        print(f"\nwrote {args.json_out}")
    return 0 if best["rate"] >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
