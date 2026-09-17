#!/usr/bin/env python3
"""Fit the residual correction, and report honestly on whether it earned its place.

    python3 tools/fit_grasp_calibration.py --dataset calib.json --out grasp_cal.json

What it does, in order:

1. Drops the placements that should not be fitted on -- unverified grasps,
   flipped pose estimates, repeats that disagree -- and says which and why.
2. Measures the noise floor from the repeats (section 6).  Everything else is
   read against this number.
3. Walks the nested ladder -- no correction, constant offset, affine, Bernstein
   degree 2, degree 3 -- under leave-one-**placement**-out cross-validation,
   choosing the smoothing weight for each degree by the same loop, and adopting
   a rung only when it beats the one below it by more than one standard error
   of the floor (sections 11 and 12).
4. Re-fits the adopted model on everything, records the cross-validated error in
   the model's own metadata, and writes it out.
5. Runs the runtime precheck against the file it just wrote, so the last thing
   printed is what :mod:`..surgicai_rl_deploy.calib.resolve` will say about it
   on the robot.

``--force-degree`` overrides the selection, for when you want to see what a
particular model does rather than what the data supports.  It is recorded in the
metadata so the override cannot be forgotten later.

``--rehearse`` runs the whole thing against a synthetic world instead of a file,
which is how to see what the output looks like before a session exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.calib.bernstein import BernsteinBasis  # noqa: E402
from surgicai_rl_deploy.calib.dataset import CalibrationDataset  # noqa: E402
from surgicai_rl_deploy.calib.fit import fit_dataset  # noqa: E402
from surgicai_rl_deploy.calib.resolve import GraspResolver  # noqa: E402
from surgicai_rl_deploy.calib import validate as V  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Fit and validate the task-specific grasp correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="calibration dataset JSON")
    src.add_argument("--rehearse", action="store_true",
                    help="run against a synthetic world instead of real data")
    ap.add_argument("--out", default=None, help="where to write the fitted model")
    ap.add_argument("--kind", choices=["total", "tensor"], default="total",
                    help="'total' keeps the model frame-invariant and cheap; "
                         "'tensor' is what Xiao et al. used")
    ap.add_argument("--max-degree", type=int, default=3)
    ap.add_argument("--force-degree", type=int, default=None)
    ap.add_argument("--force-smoothing", type=float, default=None)
    ap.add_argument("--robust", choices=["none", "huber"], default="huber")
    ap.add_argument("--folds", type=int, default=0,
                    help="0 means leave one placement out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rehearse-n", type=int, default=60)
    ap.add_argument("--rehearse-curved", action="store_true",
                    help="give the synthetic world a genuinely curved bias, so "
                         "the selector has something to find")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if args.rehearse:
        from surgicai_rl_deploy.calib.simulate import (
            SyntheticWorld, make_dataset, saddle_bias,
        )
        world = SyntheticWorld(
            fp_extra_bias=saddle_bias() if args.rehearse_curved else None,
            fp_flip_rate=0.02,
        )
        ds = make_dataset(world, args.rehearse_n, seed=args.seed, n_grasps=3)
        print(f"rehearsing against a synthetic world "
              f"({'curved' if args.rehearse_curved else 'affine'} truth, "
              f"{args.rehearse_n} placements)\n")
    else:
        ds = CalibrationDataset.load(args.dataset)

    dropped = ds.dropped()
    if dropped:
        print("DROPPED BEFORE FITTING")
        for pid, reasons in dropped:
            print(f"  {pid}: " + "; ".join(reasons))
        print()
    ds = ds.usable()
    if len(ds) == 0:
        print("nothing left to fit on.")
        return 1

    robust = None if args.robust == "none" else args.robust
    print(V.report(ds, kind=args.kind, max_degree=args.max_degree,
                   k=args.folds, seed=args.seed, robust=robust))
    print()

    # -- pick the model ----------------------------------------------------
    nf = ds.noise_floor()
    floor = nf.get("residual_sd_3d_mm") or 0.4
    margin = float(floor / np.sqrt(max(len(ds), 1)))
    sel = V.select_model(ds, kind=args.kind, max_degree=args.max_degree,
                         k=args.folds, seed=args.seed, robust=robust,
                         margin_mm=margin)
    chosen, smoothing, forced = sel.best.basis, sel.best.smoothing, False
    if args.force_degree is not None:
        chosen = BernsteinBasis(args.force_degree, args.kind)
        smoothing = 0.0 if args.force_smoothing is None else args.force_smoothing
        forced = True
        cv = V.cross_validate(ds, chosen, smoothing, k=args.folds,
                              seed=args.seed, robust=robust)
        print("=" * 78)
        print(f"FORCED: {chosen.describe()}, lambda={smoothing:g}")
        print("  " + cv.summary.describe("cross-validated"))
        print(f"  the data supports {sel.best.label}")
        print()
    else:
        cv = sel.best

    # -- refit on everything, and record what validation said --------------
    model = fit_dataset(
        ds, chosen, smoothing, robust=robust,
        metadata={
            "validated": True,
            "cv_rmse_mm": cv.summary.rmse_3d_mm,
            "cv_median_mm": cv.summary.median_3d_mm,
            "cv_p90_mm": cv.summary.p90_3d_mm,
            "cv_max_mm": cv.summary.max_3d_mm,
            "cv_baseline_mm": sel.baseline.summary.rmse_3d_mm,
            "cv_scheme": ("leave-one-placement-out" if not args.folds
                          else f"{args.folds}-fold, grouped by placement"),
            "noise_floor_mm": floor,
            "noise_floor_complete": bool(nf.get("complete")),
            "adoption_margin_mm": margin,
            "selection": [
                {k: s[k] for k in ("to", "adopted", "rmse_delta_mm", "ci95_mm")}
                for s in sel.steps
            ],
            "forced": forced,
            "dropped": [pid for pid, _ in dropped],
        },
    )

    print("=" * 78)
    print("FITTED MODEL")
    print("=" * 78)
    print("  " + model.describe())
    print(f"  cross-validated {cv.summary.rmse_3d_mm:.3f} mm RMSE against "
          f"{sel.baseline.summary.rmse_3d_mm:.3f} mm uncorrected, "
          f"floor {floor:.3f} mm")
    if not model.metadata["noise_floor_complete"]:
        missing = []
        if nf.get("perception_sd_mm") is None:
            missing.append("repeated FoundationPose frames")
        if nf.get("grasp_sd_mm") is None:
            missing.append("repeated taught grasps")
        print(f"  ! no {' and no '.join(missing)} were recorded, so the floor above")
        print("    is a LOWER bound and the adoption margin is too generous.  On the")
        print("    synthetic rehearsal the teaching term is the larger of the two.")
    print()

    out = Path(args.out or ("grasp_calibration.json" if not args.rehearse
                            else "grasp_calibration_rehearsal.json"))
    model.save(out)
    print(f"wrote {out}")
    print()

    print("=" * 78)
    print("WHAT THE RUNTIME WILL SAY ABOUT THIS FILE")
    print("=" * 78)
    print(GraspResolver.from_files(out).precheck().render())

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "model": model.as_dict(),
            "noise_floor": nf,
            "coverage": ds.coverage(),
            "ladder": sel.steps,
            "results": [
                {"label": r.label, **r.summary.as_dict(),
                 "effective_dof": r.effective_dof, "condition": r.condition}
                for r in sel.results
            ],
        }, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
