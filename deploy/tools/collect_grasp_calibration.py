#!/usr/bin/env python3
"""Plan, ingest and audit a grasp-calibration session.

Three modes, in the order a session uses them.

``plan``
    Before touching the robot: propose where to put the needle.  Section 5 asks
    for coverage in x, y **and** z and warns against clustering, and "put it
    somewhere different each time" reliably produces a cloud with a hole in the
    middle and nothing at the corners.  This emits a maximin-spread design over
    the box you intend to grasp in, with the corners included, in the order to
    collect them -- so a session cut short still leaves a spread-out dataset
    rather than one edge of the box.

``ingest``
    Turn the two logs a session produces into a dataset:

      * a FoundationPose log -- one record per frame, with a placement id
      * a grasp log -- one record per verified successful grasp, same ids

    Both are JSON lists (or JSON-lines).  Poses are given either as a 4x4
    ``T`` or as ``position`` + ``quaternion`` (xyzw).  The placement id is the
    grouping key for everything downstream, so it is required, and two
    placements that share an id are refused rather than merged.

``audit``
    Read a dataset back and say whether it is fit to fit on: repeatability,
    coverage, the orientation freeze, flipped pose estimates, and what degree
    of model this many placements can afford.  Run it *during* the session,
    not after -- a thin axis or a drifting wrist is cheap to fix while the
    needle is still on the pad.

    python3 tools/collect_grasp_calibration.py plan \\
        --centre-cm 2.5 1.0 9.0 --box-cm 4 4 3 --n 40 --out placements.json

    python3 tools/collect_grasp_calibration.py ingest \\
        --pose-log fp_log.json --grasp-log grasps.json --out calib.json

    python3 tools/collect_grasp_calibration.py audit --dataset calib.json

Record formats
--------------
FoundationPose log entry::

    {"placement_id": "p007", "T": [[...4x4...]]}
    {"placement_id": "p007", "position": [x, y, z], "quaternion": [qx,qy,qz,qw]}

Grasp log entry (``measured_cp`` at the verified successful grasp, ECM frame)::

    {"placement_id": "p007", "position": [x, y, z], "quaternion": [qx,qy,qz,qw],
     "verified": true, "note": "closed at 30 deg on the arc, lifted cleanly"}

``verified`` defaults to true, and a ``false`` keeps the record in the file --
so the session is auditable -- while excluding it from every fit.  Positions are
metres.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.calib.dataset import CalibrationDataset, Placement  # noqa: E402
from surgicai_rl_deploy.calib.perception import (  # noqa: E402
    DEFAULT_GRASP_ANGLE_DEG,
    DEFAULT_T_EC,
    GraspPointSpec,
    HandEye,
)
from surgicai_rl_deploy.calib.validate import (  # noqa: E402
    format_variance_budget,
    placements_needed,
)


# ---------------------------------------------------------------------------
def load_records(path) -> list:
    """A JSON list, or JSON-lines.  Both happen, so both are accepted."""
    text = Path(path).read_text().strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def record_to_T(rec: dict) -> np.ndarray:
    if "T" in rec:
        return np.asarray(rec["T"], dtype=np.float64).reshape(4, 4)
    if "position" in rec and "quaternion" in rec:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(np.asarray(rec["quaternion"], dtype=np.float64)).as_matrix()
        T[:3, 3] = np.asarray(rec["position"], dtype=np.float64).reshape(3)
        return T
    raise ValueError(
        "every record needs either a 4x4 'T' or a 'position' and 'quaternion'; "
        f"got keys {sorted(rec)}"
    )


# ---------------------------------------------------------------------------
def maximin_design(centre_m, half_m, n: int, seed: int = 0) -> np.ndarray:
    """A spread-out set of ``n`` points in the box, corners first.

    Greedy maximin over a dense candidate set: each new point is the candidate
    furthest from everything chosen so far.  Two properties matter.  The corners
    come out first, because they are furthest from an empty set -- which is what
    you want, since the corners are where a polynomial correction is least
    constrained and where the spatial hold-out of section 12 will be evaluated.
    And the prefix of any length is itself well spread, so a session that stops
    at placement nineteen still covers the box.
    """
    rng = np.random.default_rng(seed)
    centre = np.asarray(centre_m, dtype=np.float64).reshape(3)
    half = np.asarray(half_m, dtype=np.float64).reshape(3)
    cand = centre + rng.uniform(-1.0, 1.0, size=(4000, 3)) * half
    corners = centre + np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    ) * half
    cand = np.vstack([corners, cand])

    chosen = [0]  # start at a corner
    d = np.linalg.norm(cand - cand[0], axis=1)
    while len(chosen) < int(n) and len(chosen) < len(cand):
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(cand - cand[k], axis=1))
    return cand[chosen]


def cmd_plan(args) -> int:
    centre = np.asarray(args.centre_cm, dtype=np.float64) / 100.0
    half = np.asarray(args.box_cm, dtype=np.float64) / 200.0
    pts = maximin_design(centre, half, args.n, args.seed)

    print(f"{args.n} needle placements over a "
          f"{args.box_cm[0]:.1f} x {args.box_cm[1]:.1f} x {args.box_cm[2]:.1f} cm box")
    print("centred at "
          f"({args.centre_cm[0]:+.2f}, {args.centre_cm[1]:+.2f}, {args.centre_cm[2]:+.2f}) cm, "
          "ECM frame.")
    print()
    print("Collect them IN THIS ORDER.  The first sixteen are the corners and the")
    print("far faces; stopping early then still leaves a dataset that spans the box.")
    print()
    print("   #   id        x        y        z   (cm, ECM)")
    for i, p in enumerate(pts):
        print(f"  {i:3d}   p{i:03d}  {p[0]*100:+7.2f}  {p[1]*100:+7.2f}  {p[2]*100:+7.2f}")
    print()

    nearest = min(
        np.linalg.norm(pts[i] - pts[j])
        for i in range(len(pts)) for j in range(i + 1, len(pts))
    ) * 1000.0
    print(f"closest pair: {nearest:.1f} mm apart")
    print()
    print("Protocol reminders, from section 5 and section 6:")
    print("  * Record at least 5 FoundationPose frames per placement, with the")
    print("    needle stationary.  Averaging divides the random part by sqrt(M)")
    print("    and leaves the systematic part -- which is what is being learned.")
    print(f"  * Teach the grasp {args.grasp_repeats} times on at least "
          f"{max(5, args.n // 5)} of the placements.")
    print("    Without repeated grasps the noise floor is unmeasured, and every")
    print("    model comparison downstream is judged against a threshold that is")
    print("    too low.  On the synthetic rehearsal the hand-taught grasp is the")
    print("    LARGER of the two noise sources, not FoundationPose.")
    print("  * Keep the wrist frozen (section 8) and approach every grasp from the")
    print("    same direction: cable hysteresis is not a function of position and")
    print("    no correction field can represent it.")
    print("  * Only a grasp you watched take the needle is a calibration sample.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {
                "frame": "ECM",
                "centre_cm": list(args.centre_cm),
                "box_cm": list(args.box_cm),
                "placements": [
                    {"placement_id": f"p{i:03d}", "target_position_m": p.tolist()}
                    for i, p in enumerate(pts)
                ],
            },
            indent=2,
        ))
        print(f"\nwrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
def cmd_ingest(args) -> int:
    poses, grasps = load_records(args.pose_log), load_records(args.grasp_log)

    by_id = {}
    for rec in poses:
        pid = str(rec["placement_id"])
        by_id.setdefault(pid, {"poses": [], "grasps": [], "verified": True, "note": ""})
        by_id[pid]["poses"].append(record_to_T(rec))
    for rec in grasps:
        pid = str(rec["placement_id"])
        if pid not in by_id:
            print(f"  ! grasp for placement {pid!r} has no pose estimates; skipped")
            continue
        by_id[pid]["grasps"].append(record_to_T(rec))
        if not bool(rec.get("verified", True)):
            by_id[pid]["verified"] = False
        if rec.get("note"):
            by_id[pid]["note"] = str(rec["note"])

    hand_eye = HandEye(
        np.asarray(json.loads(Path(args.hand_eye).read_text()), dtype=np.float64)
        if args.hand_eye else DEFAULT_T_EC,
        source=args.hand_eye or "package default (supplied with the task description)",
    )
    ds = CalibrationDataset(
        hand_eye=hand_eye,
        grasp_point=GraspPointSpec(mode=args.needle_point, angle_deg=args.grasp_angle),
        provenance={
            "pose_log": str(args.pose_log),
            "grasp_log": str(args.grasp_log),
            "arm": args.arm,
            "note": args.note,
        },
    )
    skipped = []
    for pid in sorted(by_id):
        rec = by_id[pid]
        if not rec["grasps"]:
            skipped.append(pid)
            continue
        ds.add(Placement(pid, rec["poses"], rec["grasps"],
                         verified=rec["verified"], note=rec["note"]))

    print(f"ingested {len(ds)} placements "
          f"({sum(p.n_pose_frames for p in ds.placements)} pose frames, "
          f"{sum(p.n_grasps for p in ds.placements)} taught grasps)")
    if skipped:
        print(f"  {len(skipped)} placement(s) had pose estimates but no grasp: "
              + ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else ""))
    ds.save(args.out)
    print(f"wrote {args.out}")
    print()
    return _audit(ds, args)


# ---------------------------------------------------------------------------
def _audit(ds: CalibrationDataset, args) -> int:
    print("=" * 78)
    print("SESSION AUDIT")
    print("=" * 78)
    cov = ds.coverage()
    print(f"  placements                 {len(ds)}"
          + (f"  ({cov['n_dropped']} unusable, listed below)"
             if cov.get("n_dropped") else ""))
    print(f"  span                       {cov['span_cm'][0]:.2f} x "
          f"{cov['span_cm'][1]:.2f} x {cov['span_cm'][2]:.2f} cm")
    print(f"  uniformity (1.0 = uniform) "
          f"{cov['uniformity'][0]:.2f}, {cov['uniformity'][1]:.2f}, "
          f"{cov['uniformity'][2]:.2f}")
    print(f"  needle point               {ds.grasp_point.describe()}")
    sens = ds.grasp_point.orientation_sensitivity_mm()
    print(f"    orientation sensitivity  {sens['rms_mm']:.2f} mm RMS over +-30 deg yaw")
    print(f"  wrist spread               {ds.orientation_spread_deg():.2f} deg "
          f"-> {ds.jaw_offset_leakage_mm():.2f} mm of unmodellable residual")

    nf = ds.usable().noise_floor() if len(ds.usable()) else ds.noise_floor()
    print()
    print("  repeatability (section 6):")
    for key, name in [
        ("perception_sd_mm", "FoundationPose, per frame"),
        ("grasp_sd_mm", "taught grasp, per repeat"),
        ("residual_sd_mm", "residual, after averaging"),
    ]:
        v = nf.get(key)
        print(f"    {name:<28} "
              + ("NOT MEASURED" if v is None
                 else f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}) mm sd"))
    floor = nf.get("residual_sd_3d_mm")
    if not nf.get("complete"):
        missing = []
        if nf.get("perception_sd_mm") is None:
            missing.append("repeated FoundationPose frames")
        if nf.get("grasp_sd_mm") is None:
            missing.append("repeated taught grasps")
        print()
        print(f"    ! No placement has {' or '.join(missing)}, so the floor")
        print("      below is a LOWER BOUND and every adoption decision made")
        print("      against it will be too generous.  In the synthetic rehearsal")
        print("      the taught grasp is the LARGER of the two terms; collect a")
        print("      handful of repeats before fitting.")
    if floor is not None:
        print(f"    3-D noise floor              {floor:.3f} mm")

    r = ds.usable().residuals() if len(ds.usable()) else np.zeros((0, 3))
    if len(r):
        n3 = np.linalg.norm(r, axis=1) * 1000.0
        print()
        print(f"  residual so far            {n3.mean():.2f} mm mean, "
              f"{n3.min():.2f} to {n3.max():.2f} mm")
        print(f"    mean offset              ({r[:,0].mean()*1000:+.2f}, "
              f"{r[:,1].mean()*1000:+.2f}, {r[:,2].mean()*1000:+.2f}) mm")

    problems = ds.problems()
    print()
    if problems:
        print("  PROBLEMS")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  no problems found")

    usable = len(ds.usable())
    print()
    print(format_variance_budget(usable, floor or 0.4))
    print()
    print("  Read that column against the floor above, not against zero: a model")
    print("  whose fitting noise is a third of the floor is not the thing limiting")
    print("  the result.  To push it below a fixed budget instead:")
    for budget in (args.budget_mm, (floor or 0.4) / 3.0):
        target = placements_needed(2, "total", floor or 0.4, budget)
        print(f"    degree-2 total under {budget:.2f} mm: {target} placements "
              f"({max(0, target - usable)} more)")
    return 0


def cmd_audit(args) -> int:
    return _audit(CalibrationDataset.load(args.dataset), args)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Plan, ingest and audit a grasp-calibration session",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--budget-mm", type=float, default=0.1,
                    help="acceptable fitting noise, for the sample-size advice")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="propose where to put the needle")
    p.add_argument("--centre-cm", type=float, nargs=3, required=True)
    p.add_argument("--box-cm", type=float, nargs=3, default=[4.0, 4.0, 3.0])
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--grasp-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("ingest", help="build a dataset from the session logs")
    p.add_argument("--pose-log", required=True)
    p.add_argument("--grasp-log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hand-eye", default=None,
                   help="JSON file holding the 4x4 ^E T_C; default is the "
                        "transform this package ships")
    p.add_argument("--needle-point", choices=["arc", "mesh_origin"], default="arc")
    p.add_argument("--grasp-angle", type=float, default=DEFAULT_GRASP_ANGLE_DEG)
    p.add_argument("--arm", default="PSM1")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("audit", help="is this dataset fit to fit on?")
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=cmd_audit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
