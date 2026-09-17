#!/usr/bin/env python3
"""Rehearse the whole calibration protocol against a synthetic robot.

Four experiments, each answering a question that is expensive to answer on
hardware and free to answer here.

``power``
    How many placements before the correction is worth having, and before the
    model selector reliably picks the right rung?  Sweeps the number of
    placements against a known truth and reports the achieved error and what
    was selected.

``controls``
    The negative and positive control, run side by side.  A world whose residual
    is exactly affine -- which is what the physics predicts, see
    ``tools/residual_structure.py`` -- and a world with a genuinely curved
    perception bias.  A selector that adopts a curved model in the first case is
    broken; one that misses it in the second is useless.  Both have to hold.

``needle-point``
    What using the FoundationPose mesh origin instead of a point on the arc
    costs, measured rather than argued.  This is section 7 of the protocol with
    a number attached.

``repeats``
    What repeated hand-taught grasps buy.  The protocol asks for them; this says
    how much they are worth, and which of the two noise sources dominates.

    python3 tools/rehearse_grasp_calibration.py controls
    python3 tools/rehearse_grasp_calibration.py power --trials 8
    python3 tools/rehearse_grasp_calibration.py needle-point
    python3 tools/rehearse_grasp_calibration.py repeats

None of this is evidence about the real robot.  It is evidence about whether the
*procedure* can tell the difference between two hypotheses, which is a different
and prior question -- and the one worth settling before a session, because a
procedure that cannot say "no" will always say "yes".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.calib import validate as V  # noqa: E402
from surgicai_rl_deploy.calib.bernstein import BernsteinBasis  # noqa: E402
from surgicai_rl_deploy.calib.perception import GraspPointSpec  # noqa: E402
from surgicai_rl_deploy.calib.simulate import (  # noqa: E402
    SyntheticWorld,
    make_dataset,
    saddle_bias,
)

FAST_SMOOTHING = np.array([0.0, 1e-5, 1e-3, 1e-1, 10.0, np.inf])


def run_one(world, n, seed, n_grasps=3, max_degree=3, kind="total", grasp_point=None):
    ds = make_dataset(world, n, seed=seed, n_grasps=n_grasps,
                      grasp_point=grasp_point).usable()
    floor = ds.noise_floor().get("residual_sd_3d_mm") or 0.4
    sel = V.select_model(
        ds, kind=kind, max_degree=max_degree, smoothings=FAST_SMOOTHING,
        seed=seed, robust="huber", margin_mm=floor / np.sqrt(max(len(ds), 1)),
    )
    return ds, sel, floor


def degree_of(label: str) -> int:
    if "no correction" in label:
        return -1
    if "constant" in label:
        return 0
    if "affine" in label:
        return 1
    return int(label.split("degree")[1].split(",")[0])


# ---------------------------------------------------------------------------
def cmd_power(args) -> int:
    print("=" * 78)
    print("HOW MANY PLACEMENTS?")
    print("=" * 78)
    print("  truth: hand-eye 1 deg + a saddle-shaped perception bias of 1 mm.")
    print("  The right answer is degree 2.  Each cell is over "
          f"{args.trials} seeds.")
    print()
    print("     N  | selected degree (mode)  | CV RMSE of selection | CV RMSE at deg 2")
    world = SyntheticWorld(fp_extra_bias=saddle_bias())
    for n in args.placements:
        picks, rmses, fixed = [], [], []
        for t in range(args.trials):
            ds, sel, floor = run_one(world, n, seed=1000 + t, n_grasps=args.grasps)
            picks.append(degree_of(sel.best.label))
            rmses.append(sel.best.rmse_mm)
            b = BernsteinBasis(2, "total")
            if b.n_params < len(ds):
                cv = V.cross_validate(ds, b, 1e-3, seed=t, robust="huber")
                fixed.append(cv.rmse_mm)
        vals, counts = np.unique(picks, return_counts=True)
        mode = int(vals[np.argmax(counts)])
        share = counts.max() / len(picks)
        print(f"   {n:4d}  |  {mode}  ({share*100:3.0f}% of seeds)        "
              f"|  {np.mean(rmses):6.3f} mm          |  "
              + (f"{np.mean(fixed):6.3f} mm" if fixed else "  n/a"))
    print()
    print("  Read the first column as the procedure's reliability, not its")
    print("  accuracy: below the point where it settles on one degree, the model")
    print("  chosen depends on which needles happened to be placed.")
    return 0


def cmd_controls(args) -> int:
    print("=" * 78)
    print("NEGATIVE AND POSITIVE CONTROL")
    print("=" * 78)
    worlds = {
        "affine truth (what the physics predicts)": SyntheticWorld(),
        "curved truth (1 mm saddle perception bias)": SyntheticWorld(
            fp_extra_bias=saddle_bias()
        ),
    }
    for name, world in worlds.items():
        print()
        print(f"  {name}")
        print("    seed | selected                                  | CV RMSE | floor")
        adopted = []
        for t in range(args.trials):
            _, sel, floor = run_one(world, args.n, seed=200 + t, n_grasps=args.grasps)
            adopted.append(degree_of(sel.best.label))
            print(f"    {200+t:4d} | {sel.best.label:<41} | {sel.best.rmse_mm:6.3f}  "
                  f"| {floor:.3f}")
        curved = sum(1 for d in adopted if d >= 2)
        print(f"    -> a curved model was adopted in {curved}/{len(adopted)} runs")
    print()
    print("  The procedure is only usable if the first block is mostly 0 or 1 and")
    print("  the second block is mostly 2.  If the first block adopts curvature,")
    print("  the adoption margin is too loose for this many placements.")
    return 0


def cmd_needle_point(args) -> int:
    print("=" * 78)
    print("WHAT THE NEEDLE-POINT CONVENTION COSTS  (protocol section 7)")
    print("=" * 78)
    print("  The arm is always taught to grasp at 30 deg on the arc.  The only")
    print("  thing that changes is which point the pipeline calls 'the needle'.")
    print()
    print("   needle yaw | needle point used  | best CV RMSE | selected")
    for yaw in args.yaw:
        world = SyntheticWorld(needle_yaw_envelope_deg=yaw)
        for mode, spec in [
            ("arc, 30 deg", GraspPointSpec(mode="arc", angle_deg=30.0)),
            ("mesh origin", GraspPointSpec(mode="mesh_origin")),
        ]:
            best, label = [], []
            for t in range(args.trials):
                _, sel, _ = run_one(world, args.n, seed=300 + t,
                                    n_grasps=args.grasps, grasp_point=spec)
                best.append(sel.best.rmse_mm)
                label.append(sel.best.label)
            print(f"    +-{yaw:3.0f} deg  | {mode:<18} | {np.mean(best):7.3f} mm  "
                  f"| {label[0]}")
        print()
    print("  The mesh-origin rows do not get better with a higher degree, because")
    print("  what they are missing is not a function of position at all: it is the")
    print("  needle's own orientation carrying a 10.18 mm lever arm.  No amount of")
    print("  polynomial fixes a convention error.")
    return 0


def cmd_repeats(args) -> int:
    print("=" * 78)
    print("WHAT REPEATED MEASUREMENTS BUY")
    print("=" * 78)
    print("  Frames average down the perception noise; repeated grasps average")
    print("  down the teaching noise AND make the floor measurable at all.")
    print()
    world = SyntheticWorld()
    print("   frames | grasps | measured floor | CV RMSE | floor is complete")
    for n_frames in args.frames:
        for n_grasps in args.grasps_list:
            rmses, floors, complete = [], [], []
            for t in range(args.trials):
                ds = make_dataset(world, args.n, seed=400 + t,
                                  n_frames=n_frames, n_grasps=n_grasps).usable()
                nf = ds.noise_floor()
                floor = nf.get("residual_sd_3d_mm") or float("nan")
                floors.append(floor)
                complete.append(nf.get("grasp_sd_mm") is not None)
                cv = V.cross_validate(ds, BernsteinBasis(1, "total"), 0.0,
                                      seed=t, robust="huber")
                rmses.append(cv.rmse_mm)
            print(f"    {n_frames:5d}  | {n_grasps:5d}  | {np.mean(floors):9.3f} mm  "
                  f"| {np.mean(rmses):6.3f} mm | {'yes' if complete[0] else 'NO'}")
    print()
    print("  With one grasp per placement the floor is reported from perception")
    print("  alone and reads far below the error actually achieved -- which makes")
    print("  every adoption decision downstream too generous.  The grasp repeats")
    print("  are not optional bookkeeping; they are what makes the comparison mean")
    print("  anything.")
    return 0


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Rehearse the calibration protocol on a synthetic robot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("power", help="how many placements are needed")
    p.add_argument("--placements", type=int, nargs="+",
                   default=[20, 30, 45, 60, 90, 120])
    p.add_argument("--trials", type=int, default=8,
                   help="seeds per cell. Below about six the reported detection "
                        "rate is itself noisy: at twenty placements the "
                        "procedure finds a 1 mm curved bias roughly half the "
                        "time, and four lucky seeds will report 100%%.")
    p.add_argument("--grasps", type=int, default=3)
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("controls", help="can the selector say no, and say yes")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--grasps", type=int, default=3)
    p.set_defaults(func=cmd_controls)

    p = sub.add_parser("needle-point", help="what the section 7 convention costs")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--grasps", type=int, default=3)
    p.add_argument("--yaw", type=float, nargs="+", default=[5.0, 30.0])
    p.set_defaults(func=cmd_needle_point)

    p = sub.add_parser("repeats", help="what repeated measurements buy")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--frames", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--grasps-list", type=int, nargs="+", default=[1, 3])
    p.set_defaults(func=cmd_repeats)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
