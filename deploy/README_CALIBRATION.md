# Task-specific empirical calibration for needle grasping

A second-stage correction between where the perception pipeline says the needle
is and where `PSM1/measured_cp` has to read for the jaw to actually take it.

```
FoundationPose  ->  needle point  ->  ^E T_C  ->  p_nom  ->  + f(p_nom)  ->  target
\_________________________________________________/        \____________/
          coarse geometric localisation                  task-specific correction
```

The geometric stage is not replaced. `f` is a Bernstein polynomial fitted to
hand-taught successful grasps, following Xiao et al., *Delta Robot Kinematic
Calibration for Precise Robot-Assisted Retinal Surgery* (ISMR 2022), which fits
the residual left after a geometric calibration and gets a delta robot under
20 µm.

Everything here is numpy and scipy. No ROS, no torch, no checkpoint, no robot.

---

## Read this before collecting anything

```bash
python3 tools/residual_structure.py
```

It derives, from the dVRK's own DH parameters and the endoscope's working
distance, what each error term contributes **as a function of position**. The
answer changes what is worth doing, so here it is up front.

### 1. Hand-eye calibration error is exactly affine

For any hand-eye error, however large,

```
r = (R_true R_nom^T - I) p_nom + b
```

identically. Not approximately — the tool checks it numerically and gets 9e-15
mm left after an affine fit. So the "simpler baseline" of the protocol's §11 is
not a simplification of that term; it **is** that term, and a polynomial of any
degree can only reproduce it.

A one-degree hand-eye rotation error is 1.4 mm at an 8 cm working distance.
`jhu-dvrk/dvrk_camera_registration`, which is how most groups produce this
transform, states its own accuracy as "closer to 5mm to 10mm cube" — and the
residual it prints while doing so is the standard deviation of the Frobenius
norm of a composed homogeneous matrix, a number with no unit anyone can act on.

### 2. A PSM joint offset barely curves at all

Over a 4 cm grasping box, a 1° offset on one joint:

| joint | raw | after constant | after affine | after quadratic |
|---|---|---|---|---|
| outer yaw | 2.543 | 0.296 | **0.0000** | 0.00000 |
| outer pitch | 2.551 | 0.297 | **0.0160** | 0.00146 |
| insertion (1 mm) | 1.000 | 0.107 | **0.0101** | 0.00103 |
| wrist pitch | 0.157 | 0.013 | 0.0015 | 0.00020 |

mm RMS. The first two joints rotate the whole distal chain about axes through
the base, so their offsets act on tool position as `(R(δ) - I) p` — exactly
linear, at any size. This matters because the dVRK caveats paper
([arXiv:2210.13598](https://arxiv.org/pdf/2210.13598)) reports that a 1° error
in the *second joint's* potentiometer offset costs 3.4 mm of end-effector RMSE
and that no accurate calibration for it exists. Large, and affine.

### 3. Perception-side geometry is affine too, at this scale

Depth scale, a depth bias growing as range squared, residual lens distortion, a
mesh-scale error: all leave **under 0.04 mm** after an affine fit over a 6 cm
envelope.

### 4. So what is the Bernstein term for?

Two honest answers.

FoundationPose is a *learned* estimator. Its bias as a function of pose is under
no obligation to be low-order, and that is the one thing in this system that
could genuinely curve. Measuring it is the experiment. But a positive result at
degree 2 or above is then a claim about the **network**, not about the robot,
and should be reported that way.

And: a calibration that turns out to be a rigid correction of the hand-eye
transform is a useful result, not a failed experiment. The framework is built so
that "no, affine was enough" is a conclusion it can reach and defend.

### 5. Two things worth more than the polynomial degree

**The needle point.** The SurgicAI needle mesh has its origin at the **centre of
the arc** — a point in empty space 10.18 mm from the wire
(`RL/utils/needle_kinematics_new.py`: `Radius = 0.1018`, /10 to metres).
SurgicAI's own grasp target is `get_bm_pose()`, 30° along the arc. Using the raw
FoundationPose translation while the arm is taught to grasp on the arc puts an
offset in the residual that **rotates with the needle**: over this project's own
±30° needle-yaw cap that is 2.7 mm RMS and 5.2 mm worst case, and no function of
position can represent it. Measured end to end in the rehearsal:

| needle yaw | needle point | best cross-validated RMSE |
|---|---|---|
| ±5° | arc, 30° | 0.466 mm |
| ±5° | mesh origin | 0.659 mm |
| ±30° | arc, 30° | **0.464 mm** |
| ±30° | mesh origin | **3.072 mm** |

**Approach direction.** Cable hysteresis on a dVRK is not a function of position
at all, so no `f(x, y, z)` can represent it. Berkeley's calibration work needed
an LSTM over a *temporal window* for exactly this reason
([RGBD fiducial sensing + RNN](https://awesomepapers.io/robotics/papers/2003.08520):
2.96 mm → 0.65 mm, 1800 samples). Keeping the approach direction consistent is
the only way to keep hysteresis out of the residual, and the pipeline's staging
and via-point structure already gives you that for free — use it.

---

## The model family, and why the baselines are inside it

A trivariate polynomial of total degree `n` on the unit cube is a member of the
tensor-product Bernstein space of degree `n` per axis, through the exact
monomial lift `u^a = Σ_i [C(i,a)/C(n,a)] B_i^n(u)`. So the whole ladder lives in
one basis:

| degree | coefficients/axis | what it is |
|---|---|---|
| 0 | 1 | **a constant offset** — §11's first non-trivial baseline |
| 1 | 4 | **`r = A p + b`** — §11's affine baseline |
| 2 | 10 | |
| 3 | 20 | |

Tensor-product degree `n` costs `(n+1)³` instead: 8, 27, 64. With tens of
hand-taught placements that difference decides whether a model is estimable at
all. Total degree is also **frame-invariant** — an affine change of variables
maps total degree `n` to total degree `n`, so the model is the same whether the
polynomial is written in camera or ECM coordinates. A tensor-product space is
not: a rotation mixes the axes.

### The estimator

```
minimise   || A c - r ||²_W  +  λ c' Ω c
```

Three deliberate choices.

**It is linear, so it is solved linearly.** The reference paper fits the
equivalent objective with Levenberg–Marquardt. The model is linear in the
coefficients, so LM is a nonlinear solver on a linear problem: it needs a
starting guess and an iteration budget and can stop early. Here it is one
augmented QR least-squares solve.

**The penalty is on curvature, not coefficient size.** `Ω` is the exact
thin-plate energy `∫ f_uu² + f_vv² + f_ww² + 2f_uv² + 2f_uw² + 2f_vw²`, computed
in closed form from the Bernstein derivative operator and Gram matrix — no
quadrature, no monomial round-trip. **Its null space is exactly the affine
functions**, so it shrinks only what is bent and leaves the physically
meaningful hand-eye term alone. The consequence is that the regularisation path
runs between the two models §11 asks you to compare:

```
λ -> ∞    the affine model
λ -> 0    the unpenalised Bernstein fit
```

Choosing `λ` by cross-validation **is** deciding whether the polynomial earned
its place. There is no separate experiment.

**It can disbelieve a sample.** `--robust huber` runs IRLS. A learned 6-D pose
estimator on a thin, near-symmetric object fails by landing on the wrong branch,
and this project has already recorded near-180° failures from its own pose audit
(`RL/needle_reset_ranges.py`). One such placement inside ordinary least squares
drags the whole field.

### What a model can command

By Bernstein's convex-hull property, the correction anywhere inside the box is
bounded by the largest control coefficient. `ResidualModel.max_correction_mm()`
reports that bound — it is a bound, not a sample estimate — and the runtime
refuses a model whose bound exceeds a stated ceiling. Outside the box nothing
bounds it, which is why `outside_policy` defaults to `refuse`.

---

## How many placements

A model with `k` coefficients per axis fitted to `N` samples at noise `σ` pays
an out-of-sample penalty of about `σ√(3k/N)` whether or not the parameters were
needed:

| model | k | N=30 | N=60 | N=100 | N=200 |
|---|---|---|---|---|---|
| constant offset | 1 | 0.13 | 0.09 | 0.07 | 0.05 |
| affine | 4 | 0.25 | 0.18 | 0.14 | 0.10 |
| total degree 2 | 10 | 0.40 | 0.28 | 0.22 | 0.15 |
| total degree 3 | 20 | 0.57 | 0.40 | 0.31 | 0.22 |
| tensor degree 2 | 27 | 0.66 | 0.46 | 0.36 | 0.25 |
| tensor degree 3 | 64 | — | — | 0.55 | 0.39 |

mm added RMS, at σ = 0.4 mm. Xiao et al. had 588 and 393 poses from a laser
tracker and *still* restricted their Bernstein model to two variables and
dropped the third to avoid overfitting. A hand-taught grasp session yields tens.

**Recommendation: total degree, ≤ 2, at 60 or more placements.** Let the
cross-validation pick between 0, 1 and 2, and expect it to say 1.

---

## The session

### 1. Plan where the needle goes

```bash
python3 tools/collect_grasp_calibration.py plan \
    --centre-cm 2.5 1.0 9.0 --box-cm 4 4 3 --n 60 --out placements.json
```

Maximin spread, corners first, so a session cut short still spans the box.

### 2. Collect

Per §5 of the protocol, for each placement:

1. Needle stationary. Record **≥ 5 FoundationPose frames** — averaging divides
   the random part by √M and leaves the systematic part, which is what is being
   learned.
2. Jog PSM1 until the jaw is correctly placed. Record `measured_cp`.
3. **Close and check it actually took the needle.** Only a verified grasp is a
   sample.
4. On at least a fifth of the placements, **re-teach the grasp 3 times**.

That last one is not bookkeeping. From the rehearsal:

| frames | grasps | measured floor | achieved CV RMSE | floor is honest |
|---|---|---|---|---|
| 1 | 1 | — | 0.911 mm | no |
| 5 | 1 | 0.313 mm | 0.640 mm | **no** |
| 5 | 3 | 0.437 mm | 0.446 mm | yes |
| 20 | 3 | 0.336 mm | 0.353 mm | yes |

With one grasp per placement the floor is computed from perception alone and
reads far below the error actually achieved, which makes every adoption decision
downstream too generous. With both, the floor and the achieved error agree — and
the teaching term is the **larger** of the two.

Keep the wrist frozen (§8) and approach every grasp the same way.

### 3. Ingest and audit — during the session, not after

```bash
python3 tools/collect_grasp_calibration.py ingest \
    --pose-log fp_log.json --grasp-log grasps.json --out calib.json
python3 tools/collect_grasp_calibration.py audit --dataset calib.json
```

A thin axis or a drifting wrist is cheap to fix while the needle is still on the
pad.

### 4. Fit

```bash
python3 tools/fit_grasp_calibration.py --dataset calib.json --out grasp_cal.json
```

Drops what should not be fitted on and says why; measures the floor; walks the
ladder under leave-one-**placement**-out cross-validation with a paired
bootstrap at each rung; refits the adopted model on everything; and finishes by
printing what the runtime will say about the file it just wrote.

### 5. Run

```bash
python3 tools/offline_grasp_lift.py \
    --start-pos <x y z> --start-quat <qx qy qz qw> \
    --needle-pose fp_estimate.json --grasp-calibration grasp_cal.json \
    --lift-sign -1 --grasp-gate always

python3 run_pipeline.py \
    --needle-pose fp_estimate.json --grasp-calibration grasp_cal.json \
    --suture-pos <x y z> --suture-quat <qx qy qz qw> --suture-confirmed \
    --controller d2 --interface move_cp --rate 2 --lift-sign -1 --execute
```

`--grasp-pos` still works and still wins if both are given. The needle
observation is resolved into exactly the same three numbers before anything
downstream sees it, so the plan, the feasibility precheck and the trace know
nothing about perception.

---

## What the runtime refuses

Rendered in the same block as the existing precheck:

| check | fails when |
|---|---|
| `calibration.bound` | the model can command more correction inside its own box than the ceiling allows |
| `calibration.conventions` | the hand-eye transform or needle point differs from the one it was fitted under (§15) |
| `calibration.validated` | the model carries no cross-validated error |
| `calibration.frames` | the repeated estimates disagree enough to be a pose flip |
| `calibration.region` | the needle is outside the calibrated box (§15) |
| `calibration.orientation` | *warns* when the wrist is far from the taught angle |

---

## Rehearsing the procedure before trusting it

```bash
python3 tools/rehearse_grasp_calibration.py controls
python3 tools/rehearse_grasp_calibration.py power
python3 tools/rehearse_grasp_calibration.py needle-point
python3 tools/rehearse_grasp_calibration.py repeats
```

`simulate.py` builds a synthetic robot and camera with known faults and lets the
residual *emerge* — a needle is placed, a true transform carries it, a jaw offset
and an arm error move the grasp, a wrong transform and a biased estimator make
the nominal point. Nothing imposes a polynomial, so a fit recovering one proves
something.

The negative control (affine truth, which is what the physics predicts) must not
adopt curvature; the positive control (a 1 mm saddle-shaped perception bias)
must. Both are in the test suite, three seeds each.

---

## Files

| | |
|---|---|
| `calib/bernstein.py` | the basis, the lift, the exact curvature penalty |
| `calib/models.py` | the fitted field, its box, the convex-hull bound, the refusal |
| `calib/perception.py` | needle geometry, `^E T_C`, flip detection, frame averaging |
| `calib/dataset.py` | placements, repeats, the noise floor, the session audits |
| `calib/fit.py` | the penalised, optionally robust, solve |
| `calib/validate.py` | grouped CV, the ladder, the §13 metrics, the report |
| `calib/simulate.py` | the synthetic world |
| `calib/resolve.py` | the runtime path and its prechecks |
| `calib/cli.py` | `--needle-pose` becoming `--grasp-pos` |
| `tools/residual_structure.py` | what the residual looks like, before any data |
| `tools/collect_grasp_calibration.py` | plan, ingest, audit |
| `tools/fit_grasp_calibration.py` | fit, validate, write |
| `tools/rehearse_grasp_calibration.py` | power analysis and controls |

Tests: `tests/test_calib_*.py`.

---

## What none of this measures

Coordinate accuracy is an intermediate metric. The performance figure is the
**success rate of real grasps at needle placements never used for calibration**.
Nothing above substitutes for running them.

## References

- Xiao, Alamdar, Song, Ebrahimi, Gehlbach, Taylor, Iordachita, *Delta Robot
  Kinematic Calibration for Precise Robot-Assisted Retinal Surgery*, ISMR 2022
  — the two-stage method this transcribes.
- Farouki, *The Bernstein polynomial basis: a centennial retrospective*, CAGD 29
  (2012) — why the basis, and where its stability does and does not hold.
- [Caveats on the first-generation dVRK](https://arxiv.org/pdf/2210.13598) — the
  1° / 3.4 mm potentiometer result, and the absence of a calibration for it.
- [Efficiently Calibrating Cable-Driven Surgical Robots with RGBD Fiducial
  Sensing and Recurrent Neural Networks](https://awesomepapers.io/robotics/papers/2003.08520)
  — 2.96 → 0.65 mm with 1800 samples and an LSTM; the hysteresis argument.
- [Markerless Suture Needle 6D Pose Tracking](https://arxiv.org/pdf/2109.12722)
  — 0.6–1.2 mm needle pose error, which is the perception noise floor to expect.
- `jhu-dvrk/dvrk_camera_registration` — the "5mm to 10mm cube" figure.
