"""Cross-validation, model selection and the metrics of section 13.

Everything in this module splits by **physical needle placement**.  There is no
function here that takes a frame, because there is no honest split that
separates two frames of one stationary needle: section 12.

What is being decided
---------------------
Not "what is the best polynomial".  The decision is a single ordered question:

    1. Does *any* correction beat the uncorrected geometric estimate?
    2. Does a constant offset explain it?
    3. Does an affine field beat the constant offset?
    4. Does curvature -- a Bernstein term of degree two or more -- beat affine?

Each rung has to clear the one below it **out of sample**, and by more than the
noise floor.  :func:`select_model` walks exactly that ladder and reports each
step with a paired bootstrap interval, so "the polynomial helped" is a statement
with a confidence attached rather than a difference of two RMS numbers.

A warning about the order of magnitude
--------------------------------------
Before collecting anything, run ``tools/residual_structure.py``.  It computes,
from the dVRK's own DH parameters and the endoscope's working distance, what
each error source in section 4 contributes as a function of position.  The
answer over a four-centimetre box is that hand-eye error is *exactly* affine,
that a one-degree joint offset leaves 0.014 mm after an affine fit, and that
depth-scale and lens-distortion errors leave under 0.01 mm.  Every physically
identified term in section 4 is affine to well under the manual grasp
repeatability.

That does not make step 4 pointless -- FoundationPose is a learned estimator and
its bias is under no obligation to be low-order, which is precisely what makes
it worth measuring.  It does mean that a *positive* result at step 4 is a claim
about the perception model rather than about the robot, and should be reported
that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import List, Optional, Sequence

import numpy as np

from .bernstein import MAX_DEGREE, BernsteinBasis, n_coefficients
from .dataset import CalibrationDataset
from .fit import fit, smoothing_grid, solve
from .models import ResidualModel, Workspace, zero_model


# ---------------------------------------------------------------------------
# what you can afford before you collect anything
# ---------------------------------------------------------------------------
def variance_budget(n_placements: int, noise_sd_mm: float = 0.4) -> list:
    """What each model costs out of sample, purely for estimating parameters.

    A model with ``k`` coefficients per axis, fitted by unpenalised least
    squares to ``N`` samples whose residual noise has standard deviation
    ``sigma`` per axis, carries an expected out-of-sample squared error of
    ``sigma^2 (1 + k/N)`` per axis against ``sigma^2 (1 - k/N)`` in sample.  The
    extra ``sigma sqrt(3k/N)`` in the 3-D norm is paid whether or not the
    parameters were needed.

    So this table is the answer to "what degree should I use" **before** any
    data exists, and it is usually a shorter answer than people expect: at
    sixty placements and a 0.4 mm floor, a tensor-product degree-3 model is not
    merely unwise, it has more coefficients than there are placements.
    """
    N = int(n_placements)
    sigma = float(noise_sd_mm)
    rows = []
    for kind in ("total", "tensor"):
        for degree in range(0, 4):
            k = n_coefficients(degree, kind)
            estimable = k < N
            rows.append(
                {
                    "model": f"Bernstein {kind} degree {degree}",
                    "alias": {0: "constant offset", 1: "affine"}.get(degree, ""),
                    "kind": kind,
                    "degree": degree,
                    "k_per_axis": k,
                    "estimable": estimable,
                    "variance_cost_mm": (
                        float(sigma * np.sqrt(3.0 * k / N)) if estimable else float("nan")
                    ),
                }
            )
    # the two families agree at degree 0 and 1 by construction; keep both rows
    # so the table reads as one ladder per family rather than a merged one
    return rows


def format_variance_budget(n_placements: int, noise_sd_mm: float = 0.4) -> str:
    rows = variance_budget(n_placements, noise_sd_mm)
    out = [
        f"  parameter cost at N = {n_placements} placements, "
        f"noise sd = {noise_sd_mm:.2f} mm per axis",
        "    model                          alias              k    added RMS (mm)",
    ]
    for r in rows:
        cost = "  not estimable" if not r["estimable"] else f"{r['variance_cost_mm']:10.2f}"
        out.append(f"    {r['model']:<30} {r['alias']:<16} {r['k_per_axis']:4d} {cost}")
    return "\n".join(out)


def placements_needed(
    degree: int, kind: str = "total", noise_sd_mm: float = 0.4,
    budget_mm: float = 0.1,
) -> int:
    """Placements required before a model's variance cost drops below a budget.

    Inverting the expression above: ``N >= 3 k sigma^2 / budget^2``.  Useful in
    the other direction -- "we want the fitting noise under a tenth of a
    millimetre, how many needles is that" -- which is the question that actually
    gets asked when someone is standing at the robot.
    """
    k = n_coefficients(int(degree), kind)
    return int(np.ceil(3.0 * k * (float(noise_sd_mm) ** 2) / (float(budget_mm) ** 2)))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@dataclass
class ErrorSummary:
    """The table section 13 asks for, in millimetres."""

    n: int
    mean_abs_mm: np.ndarray       # per axis
    bias_mm: np.ndarray           # per axis, signed
    rmse_axis_mm: np.ndarray      # per axis
    rmse_3d_mm: float
    median_3d_mm: float
    p90_3d_mm: float
    p95_3d_mm: float
    max_3d_mm: float
    errors_3d_mm: np.ndarray = field(repr=False, default=None)

    def describe(self, label: str = "") -> str:
        a = self.rmse_axis_mm
        b = self.bias_mm
        return (
            f"{label:<28} RMSE {self.rmse_3d_mm:6.3f}  median {self.median_3d_mm:6.3f}  "
            f"p90 {self.p90_3d_mm:6.3f}  max {self.max_3d_mm:6.3f}   "
            f"per-axis RMSE ({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f})  "
            f"bias ({b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f})"
        )

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_abs_mm": self.mean_abs_mm.tolist(),
            "bias_mm": self.bias_mm.tolist(),
            "rmse_axis_mm": self.rmse_axis_mm.tolist(),
            "rmse_3d_mm": self.rmse_3d_mm,
            "median_3d_mm": self.median_3d_mm,
            "p90_3d_mm": self.p90_3d_mm,
            "p95_3d_mm": self.p95_3d_mm,
            "max_3d_mm": self.max_3d_mm,
        }


def summarise(errors_m) -> ErrorSummary:
    """``errors_m`` is ``(N, 3)`` of predicted minus measured, in metres."""
    e = np.asarray(errors_m, dtype=np.float64).reshape(-1, 3) * 1000.0
    n3 = np.linalg.norm(e, axis=1)
    return ErrorSummary(
        n=len(e),
        mean_abs_mm=np.abs(e).mean(axis=0),
        bias_mm=e.mean(axis=0),
        rmse_axis_mm=np.sqrt((e ** 2).mean(axis=0)),
        rmse_3d_mm=float(np.sqrt(np.mean(n3 ** 2))),
        median_3d_mm=float(np.median(n3)),
        p90_3d_mm=float(np.percentile(n3, 90)),
        p95_3d_mm=float(np.percentile(n3, 95)),
        max_3d_mm=float(n3.max()),
        errors_3d_mm=n3,
    )


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def grouped_folds(n: int, k: int, seed: int = 0) -> List[np.ndarray]:
    """``k`` index folds over ``n`` placements.  ``k >= n`` means leave-one-out."""
    idx = np.arange(n)
    if k >= n:
        return [np.array([i]) for i in idx]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    return [f for f in np.array_split(idx, int(k)) if len(f)]


def spatial_holdout(p_nom_m, held_fraction: float = 0.25) -> List[np.ndarray]:
    """One fold: the placements furthest from the centroid.

    Section 12 asks for "spatially held-out regions" as well as ordinary
    held-out configurations, and the two ask different questions.  A random
    fold measures interpolation between placements that surround it.  A shell
    fold measures what happens at the edge of the calibrated box -- which is
    where a polynomial correction is least constrained and most confident, and
    where, on real hardware, the needle will eventually land.
    """
    p = np.asarray(p_nom_m, dtype=np.float64).reshape(-1, 3)
    d = np.linalg.norm(p - p.mean(axis=0), axis=1)
    n_hold = max(1, int(round(len(p) * float(held_fraction))))
    order = np.argsort(d)
    return [order[-n_hold:]]


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    label: str
    basis: BernsteinBasis
    smoothing: float
    summary: ErrorSummary
    effective_dof: float
    condition: float
    #: per-placement out-of-fold 3-D error, one entry per *scored* placement
    per_placement_mm: np.ndarray = field(repr=False, default=None)
    #: dataset indices the entries above correspond to.  Equal to
    #: ``arange(len(dataset))`` for a full cross-validation, and a subset when
    #: explicit folds were given -- a spatial hold-out scores only the shell.
    scored_index: np.ndarray = field(repr=False, default=None)

    @property
    def rmse_mm(self) -> float:
        return self.summary.rmse_3d_mm


def cross_validate(
    dataset: CalibrationDataset,
    basis: BernsteinBasis,
    smoothing: float = 0.0,
    folds: Optional[Sequence[np.ndarray]] = None,
    k: int = 0,
    seed: int = 0,
    robust: Optional[str] = None,
    label: str = "",
) -> CVResult:
    """Out-of-fold error of one (basis, smoothing) pair.

    The workspace box is taken once from the **whole** dataset and reused in
    every fold.  That is deliberate and it is not leakage in any way that
    matters: for an unpenalised fit the normalisation is an affine change of
    variables, and the space of polynomials of a given total degree is closed
    under those, so the fitted function is bit-for-bit the same whichever box is
    used.  The box only affects the units the curvature penalty is measured in,
    which must be identical across folds or the folds are not comparing the same
    estimator.
    """
    p_nom = dataset.p_nom()
    r = dataset.residuals()
    ws = Workspace.from_points(p_nom)
    uvw = ws.normalise(p_nom)
    n = len(p_nom)
    # Built once: a leave-one-out sweep over a smoothing grid otherwise rebuilds
    # the same matrix a few thousand times.
    A_full = basis.design(uvw)

    if folds is None:
        folds = grouped_folds(n, k or n, seed)

    pred = np.full_like(r, np.nan)
    edfs, conds = [], []
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        if len(train) <= basis.n_params and np.isfinite(smoothing) and smoothing == 0.0:
            # Underdetermined fold: lstsq would return a minimum-norm solution
            # and quietly report a number.  Refuse instead.
            raise ValueError(
                f"{basis.describe()} has {basis.n_params} coefficients and this "
                f"fold trains on {len(train)} placements: the fit is "
                "underdetermined and any error it reports is an artefact. "
                "Use a lower degree, or a positive smoothing weight."
            )
        res = fit(
            basis, uvw[train], r[train], smoothing, robust, design=A_full[train]
        )
        pred[test] = A_full[test] @ res.coefficients
        edfs.append(res.effective_dof)
        conds.append(res.condition)

    scored = np.unique(np.concatenate([np.asarray(f, dtype=int) for f in folds]))
    summary = summarise(pred[scored] - r[scored])
    return CVResult(
        label=label or f"{basis.describe()} lambda={smoothing:g}",
        basis=basis,
        smoothing=smoothing,
        summary=summary,
        effective_dof=float(np.mean(edfs)),
        condition=float(np.max(conds)),
        per_placement_mm=summary.errors_3d_mm,
        scored_index=scored,
    )


def uncorrected(dataset: CalibrationDataset) -> CVResult:
    """Section 11's first baseline: ``p_grasp_hat = p_nom``, no fitting at all.

    Needs no folds -- there is nothing fitted to leak -- but is returned in the
    same shape as every other result so the comparison table is uniform.
    """
    r = dataset.residuals()
    summary = summarise(-r)
    basis = BernsteinBasis(0, "total")
    return CVResult(
        label="no correction (p_nom as-is)",
        basis=basis,
        smoothing=np.inf,
        summary=summary,
        effective_dof=0.0,
        condition=1.0,
        per_placement_mm=summary.errors_3d_mm,
        scored_index=np.arange(len(r)),
    )


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------
def paired_improvement(
    a: CVResult, b: CVResult, n_boot: int = 4000, seed: int = 0
) -> dict:
    """Is ``b`` better than ``a``, and by how much, with an interval?

    The placements are paired -- both models were scored on the same needles --
    so the comparison is a paired bootstrap over placements rather than two
    independent error bars.  ``rmse_delta_mm`` is positive when ``b`` is better.

    ``p_better`` is the bootstrap fraction of resamples in which ``b`` wins.  It
    is not a p-value and should not be reported as one; it is the share of the
    resampling distribution on the useful side of zero.
    """
    ea = np.asarray(a.per_placement_mm, dtype=np.float64)
    eb = np.asarray(b.per_placement_mm, dtype=np.float64)
    if len(ea) != len(eb):
        raise ValueError("paired comparison needs the same placements in both results")
    if (
        a.scored_index is not None
        and b.scored_index is not None
        and not np.array_equal(a.scored_index, b.scored_index)
    ):
        raise ValueError(
            "paired comparison needs both results scored on the same placements; "
            "a full cross-validation and a spatial hold-out are not comparable"
        )
    rng = np.random.default_rng(seed)
    n = len(ea)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    rmse_a = np.sqrt((ea[idx] ** 2).mean(axis=1))
    rmse_b = np.sqrt((eb[idx] ** 2).mean(axis=1))
    delta = rmse_a - rmse_b
    return {
        "rmse_a_mm": float(np.sqrt((ea ** 2).mean())),
        "rmse_b_mm": float(np.sqrt((eb ** 2).mean())),
        "rmse_delta_mm": float(np.sqrt((ea ** 2).mean()) - np.sqrt((eb ** 2).mean())),
        "ci95_mm": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        "p_better": float((delta > 0).mean()),
        "median_paired_delta_mm": float(np.median(ea - eb)),
        "n": n,
    }


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
@dataclass
class Selection:
    results: List[CVResult]
    best: CVResult
    baseline: CVResult
    steps: List[dict]
    noise_floor: dict
    notes: List[str]


def select_model(
    dataset: CalibrationDataset,
    kind: str = "total",
    max_degree: int = 3,
    smoothings: Optional[Sequence[float]] = None,
    k: int = 0,
    seed: int = 0,
    robust: Optional[str] = None,
    margin_mm: float = 0.0,
) -> Selection:
    """Walk the ladder and report each rung honestly.

    ``margin_mm`` is how much a rung must beat the one below it by before it is
    adopted.  Zero means "any improvement at all", which overfits the model
    *selection* even when each individual fit is cross-validated.  Setting it to
    the noise floor's own standard error is the defensible choice, and
    :func:`report` does that automatically.
    """
    if smoothings is None:
        smoothings = smoothing_grid()

    n = len(dataset)
    base = uncorrected(dataset)
    results = [base]
    notes = []

    best_by_degree = {}
    for degree in range(0, int(max_degree) + 1):
        basis = BernsteinBasis(degree, kind)
        if basis.n_params >= n:
            notes.append(
                f"{basis.describe()} skipped: {basis.n_params} coefficients "
                f"against {n} placements"
            )
            continue
        # Degree 0 and 1 have no curvature to penalise, so one fit each.
        lams = [0.0] if degree < 2 else list(smoothings)
        rung = []
        for lam in lams:
            try:
                res = cross_validate(
                    dataset, basis, lam, k=k, seed=seed, robust=robust
                )
            except ValueError as exc:
                notes.append(f"{basis.describe()} lambda={lam:g}: {exc}")
                continue
            rung.append(res)
        if not rung:
            continue
        pick = min(rung, key=lambda r: r.rmse_mm)
        alias = {0: "constant offset", 1: "affine"}.get(degree)
        pick.label = (
            f"{alias} (Bernstein {kind} degree 0)" if degree == 0 else
            f"{alias} (Bernstein {kind} degree 1)" if degree == 1 else
            f"Bernstein {kind} degree {degree}, lambda={pick.smoothing:g}"
        )
        best_by_degree[degree] = pick
        results.append(pick)

    # The ladder: each rung must beat the previous one by the margin.
    steps = []
    current = base
    for degree in sorted(best_by_degree):
        cand = best_by_degree[degree]
        cmp = paired_improvement(current, cand, seed=seed)
        adopted = cmp["rmse_delta_mm"] > float(margin_mm)
        steps.append({"from": current.label, "to": cand.label, "adopted": adopted, **cmp})
        if adopted:
            current = cand

    return Selection(
        results=results,
        best=current,
        baseline=base,
        steps=steps,
        noise_floor=dataset.noise_floor(),
        notes=notes,
    )


# ---------------------------------------------------------------------------
def report(
    dataset: CalibrationDataset,
    kind: str = "total",
    max_degree: int = 3,
    k: int = 0,
    seed: int = 0,
    robust: Optional[str] = "huber",
) -> str:
    """The whole of sections 6, 11, 12 and 13, as text."""
    lines = []
    W = 78
    lines.append("=" * W)
    lines.append("TASK-SPECIFIC GRASP CALIBRATION -- CROSS-VALIDATED REPORT")
    lines.append("=" * W)

    # -- the data
    cov = dataset.coverage()
    lines.append("")
    lines.append(f"  placements                  {len(dataset)}")
    lines.append(
        f"  nominal span                {cov['span_cm'][0]:.2f} x "
        f"{cov['span_cm'][1]:.2f} x {cov['span_cm'][2]:.2f} cm"
    )
    lines.append(f"  hand-eye                    {dataset.hand_eye.describe()}")
    lines.append(f"  needle point                {dataset.grasp_point.describe()}")
    sens = dataset.grasp_point.orientation_sensitivity_mm()
    lines.append(
        f"  its orientation sensitivity {sens['rms_mm']:.2f} mm RMS, "
        f"{sens['max_mm']:.2f} mm max over +-30 deg of needle yaw"
    )
    lines.append(
        f"  gripper orientation spread  {dataset.orientation_spread_deg():.1f} deg "
        f"-> about {dataset.jaw_offset_leakage_mm():.2f} mm of unmodellable residual"
    )

    problems = dataset.problems()
    if problems:
        lines.append("")
        lines.append("  PROBLEMS WITH THIS DATASET")
        for p in problems:
            lines.append(f"    - {p}")

    # -- section 6
    nf = dataset.noise_floor()
    lines.append("")
    lines.append("-" * W)
    lines.append("SECTION 6 -- REPEATABILITY, THE FLOOR EVERY OTHER NUMBER SITS ON")
    lines.append("-" * W)
    for key, name in [
        ("perception_sd_mm", "FoundationPose, per frame"),
        ("grasp_sd_mm", "hand-taught grasp, per repeat"),
        ("residual_sd_mm", "residual, after averaging"),
    ]:
        v = nf.get(key)
        if v is None:
            lines.append(f"  {name:<32} not measured (no repeats recorded)")
        else:
            lines.append(
                f"  {name:<32} ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}) mm sd per axis"
            )
    floor = nf.get("residual_sd_3d_mm")
    if floor is not None:
        lines.append(f"  3-D noise floor                  {floor:.3f} mm"
                     + ("" if nf.get("complete") else "   [LOWER BOUND ONLY]"))
        lines.append(
            "  No deterministic correction field can do better than this, and a"
        )
        lines.append(
            "  model whose cross-validated error is already near it is finished."
        )

    # -- the budget
    lines.append("")
    lines.append("-" * W)
    lines.append("WHAT THIS MANY PLACEMENTS CAN AFFORD")
    lines.append("-" * W)
    lines.append(format_variance_budget(len(dataset), floor or 0.4))

    # -- the ladder
    margin = 0.0
    if floor is not None and len(dataset) > 1:
        margin = float(floor / np.sqrt(len(dataset)))  # one standard error
    sel = select_model(
        dataset, kind=kind, max_degree=max_degree, k=k, seed=seed,
        robust=robust, margin_mm=margin,
    )

    lines.append("")
    lines.append("-" * W)
    lines.append(
        f"SECTIONS 11-13 -- LEAVE-ONE-PLACEMENT-OUT, ADOPTION MARGIN {margin:.3f} mm"
    )
    lines.append("-" * W)
    for res in sel.results:
        lines.append("  " + res.summary.describe(res.label))
    lines.append("")
    lines.append("  ladder:")
    for s in sel.steps:
        verdict = "ADOPTED" if s["adopted"] else "rejected"
        lines.append(
            f"    {s['to']:<44} {verdict}   "
            f"{s['rmse_delta_mm']:+.3f} mm "
            f"[{s['ci95_mm'][0]:+.3f}, {s['ci95_mm'][1]:+.3f}]  "
            f"wins {s['p_better']*100:.0f}% of resamples"
        )
    lines.append("")
    lines.append(f"  SELECTED: {sel.best.label}")
    improvement = paired_improvement(sel.baseline, sel.best, seed=seed)
    lines.append(
        f"  against no correction at all: "
        f"{improvement['rmse_a_mm']:.3f} -> {improvement['rmse_b_mm']:.3f} mm RMSE "
        f"({improvement['rmse_delta_mm']:+.3f} mm, 95% CI "
        f"[{improvement['ci95_mm'][0]:+.3f}, {improvement['ci95_mm'][1]:+.3f}])"
    )

    # -- spatial hold-out
    lines.append("")
    lines.append("-" * W)
    lines.append("SECTION 12 -- THE SPATIAL HOLD-OUT (the outer shell of the box)")
    lines.append("-" * W)
    try:
        shell = spatial_holdout(dataset.p_nom())
        shell_res = cross_validate(
            dataset, sel.best.basis, sel.best.smoothing, folds=shell, robust=robust
        )
        lines.append(
            "  " + summarise(-dataset.residuals()[shell[0]]).describe(
                "no correction, shell only"
            )
        )
        lines.append("  " + shell_res.summary.describe("selected model, shell only"))
        lines.append(
            "  Interpolating between placements is easy; holding up at the edge of"
        )
        lines.append(
            "  the calibrated box is the part that predicts a real grasp."
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        lines.append(f"  could not run the shell hold-out: {exc}")

    if sel.notes:
        lines.append("")
        lines.append("  notes:")
        for n_ in sel.notes:
            lines.append(f"    - {n_}")

    lines.append("")
    lines.append("-" * W)
    lines.append("WHAT THIS REPORT DOES NOT MEASURE")
    lines.append("-" * W)
    lines.append(
        "  Section 13: coordinate accuracy is an intermediate metric.  The"
    )
    lines.append(
        "  performance figure is the success rate of real grasps at needle"
    )
    lines.append(
        "  placements that were never used for calibration.  Nothing above is a"
    )
    lines.append("  substitute for running them.")
    return "\n".join(lines)
