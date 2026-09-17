#!/usr/bin/env python3
"""What shape is the residual field, before anyone collects a single placement?

Run this first.  It answers, from the dVRK's own DH parameters and the
endoscope's working distance, the question the protocol document leaves open:
which of the error terms in section 4 actually produce a *curved* function of
position, and how curved, in millimetres.

That matters because the manual grasp demonstrations a calibration is built from
are expensive -- tens of needles per session, not thousands of poses under a
laser tracker -- and the polynomial degree that can be afforded is decided by
that number, not by which degree fits best in sample.

What it computes
----------------
1. **Hand-eye error is exactly affine.**  Not approximately.  For any hand-eye
   error, ``r = (R_true R_nom^T - I) p_nom + b`` identically, so the affine
   model of section 11 is not a simplification of that term, it *is* that term.
   Checked numerically against a deliberately large error.

2. **A PSM joint offset barely curves at all.**  The first two joints rotate the
   entire distal chain about axes through the base, so their offsets act on the
   tool position as ``(R(delta) - I) p`` -- again exactly linear.  Insertion is
   prismatic and nearly so.  Over a four-centimetre grasping box, a one-degree
   offset produces 2.5 mm of error of which an affine fit leaves 0.014 mm.

3. **Perception-side terms are affine too, at this scale.**  Depth scale, a
   depth bias growing with range, residual lens distortion: all under 0.04 mm
   after an affine fit over a six-centimetre envelope.

4. **What that many placements can afford.**  Coefficient counts against the
   out-of-sample variance each one costs.

The honest conclusion, which the tool prints
--------------------------------------------
No error source identified in section 4 produces curvature above the manual
grasp repeatability over this workspace.  That does not make the Bernstein term
pointless -- FoundationPose is a *learned* estimator and its bias is under no
obligation to be low-order, which is exactly what makes it worth measuring --
but it does mean a positive result at degree two or higher is a claim about the
perception model rather than about the robot, and should be reported as one.

    python3 tools/residual_structure.py
    python3 tools/residual_structure.py --box-cm 6 --placements 60 --noise-mm 0.4
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.calib.perception import (  # noqa: E402
    DEFAULT_T_EC,
    GraspPointSpec,
    NEEDLE_ANGLES_DEG,
    NEEDLE_RADIUS_M,
)
from surgicai_rl_deploy.calib.validate import format_variance_budget  # noqa: E402

PI_2 = np.pi / 2

#: dVRK PSM with a Large Needle Driver (tool 400006), modified DH, copied from
#: ``surgical_robotics_challenge/kinematics/config/kinematic/psm_400006.json``
#: and ``.../tool/LARGE_NEEDLE_DRIVER_400006.json``.
#: (alpha, a, D, offset, prismatic)
PSM_DH = [
    (PI_2, 0.0, 0.0, PI_2, False),      # outer yaw
    (-PI_2, 0.0, 0.0, -PI_2, False),    # outer pitch
    (PI_2, 0.0, 0.0, -0.4389, True),    # outer insertion
    (0.0, 0.0, 0.416, 0.0, False),      # outer roll
    (-PI_2, 0.0, 0.0, -PI_2, False),    # wrist pitch
    (-PI_2, 0.009, 0.0, -PI_2, False),  # wrist yaw
    (-PI_2, 0.0, 0.0, PI_2, False),     # fixed tip frame
]
JOINT_NAMES = [
    "outer yaw", "outer pitch", "insertion", "roll", "wrist pitch", "wrist yaw"
]


def modified_dh(alpha, a, theta, d) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def psm_fk(q) -> np.ndarray:
    """Tool frame in the PSM base frame, for six joint values."""
    T = np.eye(4)
    for (alpha, a, D, offset, prismatic), qi in zip(PSM_DH, list(q) + [0.0]):
        if prismatic:
            T = T @ modified_dh(alpha, a, 0.0, D + offset + qi)
        else:
            T = T @ modified_dh(alpha, a, qi + offset, D)
    return T


# ---------------------------------------------------------------------------
def poly_design(p, total_degree: int) -> np.ndarray:
    """Total-degree monomial design matrix.  Used only for decomposition here;
    the fitting itself uses the Bernstein basis in :mod:`..calib.bernstein`."""
    x, y, z = np.asarray(p).T
    cols = [np.ones(len(x))]
    if total_degree >= 1:
        cols += [x, y, z]
    if total_degree >= 2:
        cols += [x * x, y * y, z * z, x * y, x * z, y * z]
    if total_degree >= 3:
        for a in range(4):
            for b in range(4 - a):
                c = 3 - a - b
                if a + b + c == 3:
                    cols.append(x ** a * y ** b * z ** c)
    return np.column_stack(cols)


def left_after(p, e, degree: int) -> float:
    """3-D RMS of ``e`` left over after fitting a polynomial of that degree, mm."""
    X = poly_design(p, degree)
    coef, *_ = np.linalg.lstsq(X, e, rcond=None)
    res = e - X @ coef
    return float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))) * 1000.0)


def raw_rms(e) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(e) ** 2, axis=1))) * 1000.0)


# ---------------------------------------------------------------------------
def section_handeye(args) -> None:
    print("=" * 78)
    print("1.  HAND-EYE CALIBRATION ERROR IS EXACTLY AFFINE IN p_nom")
    print("=" * 78)
    rng = np.random.default_rng(args.seed)

    dR = Rotation.from_rotvec(
        np.deg2rad(args.handeye_deg) * np.array([0.3, -0.7, 0.65]) / np.linalg.norm([0.3, -0.7, 0.65])
    ).as_matrix()
    dt = np.array([0.003, -0.002, 0.0015])
    T_nom = DEFAULT_T_EC
    T_true = np.eye(4)
    T_true[:3, :3] = dR @ T_nom[:3, :3]
    T_true[:3, 3] = dR @ T_nom[:3, 3] + dt

    half = args.box_cm / 200.0
    pC = np.column_stack([
        rng.uniform(-half, half, args.samples),
        rng.uniform(-half, half, args.samples),
        args.range_cm / 100.0 + rng.uniform(-half, half, args.samples),
    ])
    to = lambda T, p: p @ T[:3, :3].T + T[:3, 3]  # noqa: E731
    p_nom = to(T_nom, pC)
    r = to(T_true, pC) - p_nom

    print(f"  a {args.handeye_deg:.2f} deg + 3.9 mm hand-eye error, over a "
          f"{args.box_cm:.0f} cm box at {args.range_cm:.0f} cm range:")
    print(f"    residual                     {raw_rms(r):8.3f} mm RMS")
    print(f"    left after a CONSTANT fit    {left_after(p_nom, r, 0):8.3f} mm RMS")
    print(f"    left after an AFFINE fit     {left_after(p_nom, r, 1):8.2e} mm RMS")
    A_closed = T_true[:3, :3] @ T_nom[:3, :3].T - np.eye(3)
    X = poly_design(p_nom, 1)
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    print(f"    fitted slope vs closed form  {np.abs(coef[1:].T - A_closed).max():8.2e}")
    print()
    print("  Section 11's affine baseline is not a cheaper alternative to the")
    print("  polynomial.  For this term it is the exact model, and a Bernstein")
    print("  polynomial of any degree can only reproduce it.")
    print()
    print("  what a rotation error costs, in mm, at endoscope working distances:")
    print("      range |   0.25 deg    0.5 deg    1.0 deg    2.0 deg")
    for rng_m in (0.05, 0.08, 0.10, 0.15):
        cells = "".join(f"{rng_m*np.deg2rad(d)*1000:10.2f} " for d in (0.25, 0.5, 1.0, 2.0))
        print(f"     {rng_m*100:4.0f} cm |{cells}")
    print()
    print("  jhu-dvrk/dvrk_camera_registration puts its own accuracy at a "
          "'5mm to 10mm cube',")
    print("  which at 8 cm is under two degrees.  That is the size of the term "
          "being corrected.")
    print()


def section_kinematics(args) -> None:
    print("=" * 78)
    print("2.  PSM JOINT-OFFSET ERROR: HOW MUCH CURVATURE OVER A GRASPING BOX?")
    print("=" * 78)
    q0 = np.array([0.0, 0.0, 0.16, 0.0, np.deg2rad(20.0), np.deg2rad(-10.0)])
    n = 9
    half_rad = args.box_cm / 200.0 / 0.17   # roughly, joint angle for that span
    sweeps = [
        np.linspace(-half_rad, half_rad, n),
        np.linspace(-half_rad, half_rad, n),
        np.linspace(q0[2] - args.box_cm / 200.0, q0[2] + args.box_cm / 200.0, n),
    ]
    qs, pts = [], []
    for a in sweeps[0]:
        for b in sweeps[1]:
            for c in sweeps[2]:
                q = q0.copy()
                q[0], q[1], q[2] = a, b, c
                qs.append(q)
                pts.append(psm_fk(q)[:3, 3])
    qs, pts = np.array(qs), np.array(pts)
    span = (pts.max(axis=0) - pts.min(axis=0)) * 100.0
    print(f"  swept {len(pts)} configurations over "
          f"{span[0]:.1f} x {span[1]:.1f} x {span[2]:.1f} cm of tool position")
    print()
    print("  a 1.0 deg offset (1.0 mm for insertion) on one joint, error in mm RMS:")
    print("    joint           raw    after const   after affine   after quadratic")
    for j in range(6):
        delta = np.zeros(6)
        delta[j] = 1.0e-3 if j == 2 else np.deg2rad(1.0)
        e = np.array([psm_fk(q + delta)[:3, 3] - psm_fk(q)[:3, 3] for q in qs])
        print(f"    {JOINT_NAMES[j]:<12} {raw_rms(e):7.3f}  {left_after(pts, e, 0):11.3f}  "
              f"{left_after(pts, e, 1):12.4f}  {left_after(pts, e, 2):13.5f}")
    print()
    print("  The first two joints rotate the whole distal chain about axes through")
    print("  the base, so their offsets act on the tool position as (R(d) - I) p:")
    print("  exactly linear, whatever the size of the offset.  This is why the")
    print("  'after affine' column is essentially zero and not merely small.")
    print()
    print("  Context: the dVRK caveats paper (arXiv:2210.13598) reports that a 1 deg")
    print("  error in the second joint's potentiometer offset costs 3.4 mm RMSE of")
    print("  end-effector error, and that no accurate calibration exists for it.")
    print("  Large -- and, per the table above, affine.")
    print()


def section_perception(args) -> None:
    print("=" * 78)
    print("3.  PERCEPTION-SIDE ERROR: WHAT SURVIVES AN AFFINE FIT")
    print("=" * 78)
    rng = np.random.default_rng(args.seed + 1)
    f = args.focal_px
    print("  envelope   source                            raw    after affine   after quad")
    for box_cm in (0.6, 2.0, args.box_cm, 6.0):
        half = box_cm / 200.0
        z0 = args.range_cm / 100.0
        pC = np.column_stack([
            rng.uniform(-half, half, args.samples),
            rng.uniform(-half, half, args.samples),
            z0 + rng.uniform(-half, half, args.samples),
        ])
        x, y, z = pC.T
        u, v = f * x / z, f * y / z
        r2 = (u ** 2 + v ** 2) / f ** 2
        zero = np.zeros_like(z)
        cases = {
            "depth scale 1%": np.column_stack([x, y, z]) * 0.01,
            "depth bias ~ range^2, 0.5 mm": np.column_stack(
                [zero, zero, 0.0005 * (z / z0) ** 2]),
            "residual lens distortion": np.column_stack(
                [15.0 * z / f * u / f * r2, 15.0 * z / f * v / f * r2, zero]),
            "needle mesh scale 1%": np.tile([NEEDLE_RADIUS_M * 0.01, 0, 0], (len(z), 1)),
        }
        for name, e in cases.items():
            print(f"   {box_cm:4.1f} cm   {name:<30} {raw_rms(e):6.3f}  "
                  f"{left_after(pC, e, 1):12.4f}  {left_after(pC, e, 2):11.4f}")
        print()
    print("  Every geometric perception term is affine to well under a tenth of a")
    print("  millimetre over this envelope.  Curvature, if the data shows any, is")
    print("  coming from the learned part of FoundationPose -- which is a real")
    print("  possibility and the one thing here worth measuring, but it is a claim")
    print("  about the network, not about the optics or the arm.")
    print()


def section_needle(args) -> None:
    print("=" * 78)
    print("4.  THE NEEDLE POINT: MESH ORIGIN vs THE ARC  (protocol section 7)")
    print("=" * 78)
    print(f"  SurgicAI needle arc radius   {NEEDLE_RADIUS_M*1000:.2f} mm")
    print("  the mesh origin is the CENTRE of the arc -- a point in empty space")
    print()
    for name, ang in NEEDLE_ANGLES_DEG.items():
        q = GraspPointSpec(angle_deg=ang).point_N()
        marker = "  <- SurgicAI's own grasp target" if name == "bm" else ""
        print(f"    {name:<6} theta = {ang:5.0f} deg   "
              f"({q[0]*1000:+7.2f}, {q[1]*1000:+7.2f}, {q[2]*1000:+6.2f}) mm{marker}")
    print()
    print("  If the raw FoundationPose translation is used as 'the needle position'")
    print("  while the arm is taught to grasp on the arc, the residual carries an")
    print("  offset that ROTATES with the needle.  The part of it no position-only")
    print("  model can represent, against the needle-yaw envelope:")
    print()
    print("      yaw envelope |   RMS      max")
    spec = GraspPointSpec(angle_deg=args.grasp_angle)
    for yaw in (5, 10, 20, 30, 45):
        s = spec.orientation_sensitivity_mm(yaw)
        print(f"        +- {yaw:3d} deg  | {s['rms_mm']:6.2f}   {s['max_mm']:6.2f}  mm")
    print()
    print("  This repository's own reset caps needle yaw at +-30 deg, where the term")
    print("  is 2.7 mm RMS -- the same size as the entire residual being modelled,")
    print("  and not a function of position.  Fixing the needle-point convention is")
    print("  worth more than any polynomial degree.")
    print()


def section_budget(args) -> None:
    print("=" * 78)
    print("5.  WHAT A SESSION'S WORTH OF PLACEMENTS CAN AFFORD")
    print("=" * 78)
    print(format_variance_budget(args.placements, args.noise_mm))
    print()
    from surgicai_rl_deploy.calib.validate import placements_needed

    print(f"  placements needed to hold the fitting noise under "
          f"{args.budget_mm:.2f} mm, at a {args.noise_mm:.2f} mm floor:")
    for kind in ("total", "tensor"):
        for degree in (1, 2, 3):
            n = placements_needed(degree, kind, args.noise_mm, args.budget_mm)
            print(f"    Bernstein {kind:<7} degree {degree}: "
                  f"{comb(degree+3,3) if kind=='total' else (degree+1)**3:3d} coeff -> "
                  f"{n:5d} placements")
    print()
    print("  Xiao et al. had 588 and 393 poses from a laser tracker and still")
    print("  restricted their Bernstein model to two variables and dropped the")
    print("  third to avoid overfitting.  A hand-taught grasp session yields tens.")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Derive the structure of the grasp residual before collecting it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--box-cm", type=float, default=4.0,
                    help="size of the grasping workspace to analyse")
    ap.add_argument("--range-cm", type=float, default=8.0,
                    help="camera-to-needle working distance")
    ap.add_argument("--handeye-deg", type=float, default=1.0,
                    help="assumed hand-eye rotation error")
    ap.add_argument("--placements", type=int, default=60,
                    help="number of physical needle placements in the budget table")
    ap.add_argument("--noise-mm", type=float, default=0.4,
                    help="per-axis residual noise floor")
    ap.add_argument("--budget-mm", type=float, default=0.1,
                    help="acceptable fitting noise, for the sample-size table")
    ap.add_argument("--grasp-angle", type=float, default=30.0,
                    help="needle arc angle of the grasp point, degrees")
    ap.add_argument("--focal-px", type=float, default=900.0)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["handeye", "kinematics", "perception",
                                       "needle", "budget"], default=None)
    args = ap.parse_args(argv)

    sections = {
        "handeye": section_handeye,
        "kinematics": section_kinematics,
        "perception": section_perception,
        "needle": section_needle,
        "budget": section_budget,
    }
    for name, fn in sections.items():
        if args.only in (None, name):
            fn(args)

    if args.only is None:
        print("=" * 78)
        print("WHAT TO DO WITH THIS")
        print("=" * 78)
        print("  * Fix the needle-point convention first (section 4 above).  It is")
        print("    worth millimetres; the polynomial degree is worth tenths.")
        print("  * Collect the repeats section 6 of the protocol asks for -- BOTH")
        print("    perception frames and repeated hand-taught grasps.  Without the")
        print("    grasp repeats the noise floor is under-reported and every model")
        print("    comparison downstream is judged against the wrong threshold.")
        print("  * Approach every taught grasp from the same direction.  Cable")
        print("    hysteresis on a dVRK is not a function of position at all, so no")
        print("    f(x, y, z) can represent it; keeping the approach consistent is")
        print("    the only way to keep it out of the residual.  This is why the")
        print("    Berkeley calibration work needed an LSTM over a temporal window")
        print("    rather than a memoryless map.")
        print("  * Expect the answer to be affine.  Report it if it is.  A")
        print("    calibration that turns out to be a rigid correction of the")
        print("    hand-eye transform is a useful result, not a failed experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
