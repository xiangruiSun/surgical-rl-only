"""Fitting the correction field.

The estimator, in one line::

    minimise  || A c - r ||_W^2  +  lambda * c' Omega c

``A`` is the Bernstein design matrix, ``Omega`` the curvature penalty from
:meth:`~.bernstein.BernsteinBasis.curvature_penalty`, and ``W`` optional
per-placement weights.  Three things about that line are deliberate.

**It is linear, so it is solved linearly.**  Xiao et al. fit the equivalent
objective with Levenberg-Marquardt.  Their model is linear in ``C`` as well, so
LM is a nonlinear solver on a linear problem: it needs a starting guess and an
iteration budget, and it can stop early.  Here the problem is stacked into one
augmented least-squares system and handed to a QR solve.  There is no starting
guess, no iteration count, and the answer is the minimiser rather than a point
near it.

**The penalty is on curvature, not on coefficient size.**  A plain ridge would
shrink the affine part of the correction -- which section 2 of the protocol
document shows *is* the hand-eye calibration error, exactly, and is the part
most worth keeping.  ``Omega`` has the affine functions in its null space, so it
shrinks only what is bent.  The consequence is that the regularisation path runs
between the two models section 11 asks to compare:

    lambda -> infinity   the affine model
    lambda -> 0          the unpenalised Bernstein fit

Choosing ``lambda`` by cross-validation is therefore the same act as deciding
whether the polynomial earned its place.  There is no separate experiment.

**It can disbelieve a sample.**  ``robust="huber"`` runs the solve as iteratively
reweighted least squares.  This is not statistical decoration: a learned 6-D pose
estimator on a thin, near-symmetric object fails by landing on the wrong branch,
and this project has already recorded near-180-degree failures from its own pose
audit.  One such placement inside an ordinary least-squares fit drags the whole
field, because least squares has no way to disbelieve a point.  The flip detector
in :mod:`.perception` catches the ones with repeats to compare; Huber catches the
ones without.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from .bernstein import BernsteinBasis
from .dataset import CalibrationDataset
from .models import ResidualModel, Workspace

#: Huber cut-off, in multiples of the robust scale estimate.  1.345 is the
#: standard choice: 95% efficiency against a genuinely Gaussian residual, while
#: still bounding the influence of an outlier.
HUBER_K = 1.345

#: Floor on the robust scale estimate, metres.
#:
#: Huber reweights by ``residual / (k * sigma)`` with ``sigma`` from the median
#: absolute deviation.  When a fit happens to be nearly exact -- which is the
#: normal case on synthetic data, and can happen on a small real dataset with a
#: well-determined model -- the MAD collapses toward zero and *every* placement
#: with any residual at all gets downweighted, including the good ones.  Ten
#: micrometres is two orders of magnitude below the dVRK's own repeatability, so
#: below it there is nothing left to disbelieve and the reweighting is noise
#: amplification rather than robustness.
MIN_ROBUST_SCALE_M = 1.0e-5

#: Below this eigenvalue a curvature-penalty direction is treated as null.
#: ``Omega``'s null space is the affine functions, exactly, and the numerically
#: computed eigenvalues of that null space land around 1e-14 relative to the
#: largest -- see the module test.
PENALTY_NULL_TOL = 1e-9


@dataclass
class FitResult:
    """A fit, and everything needed to judge it."""

    coefficients: np.ndarray          # (n_params, 3), metres
    basis: BernsteinBasis
    smoothing: float
    #: trace of the hat matrix: how many parameters the fit actually spent
    effective_dof: float
    #: condition number of the (penalised) design
    condition: float
    #: per-placement weights the final iteration used (1.0 unless robust)
    weights: np.ndarray
    #: in-sample 3-D RMS, millimetres.  NOT a measure of anything; see validate.
    train_rmse_mm: float
    n_samples: int

    def describe(self) -> str:
        lam = "inf (affine)" if not np.isfinite(self.smoothing) else f"{self.smoothing:.3g}"
        return (
            f"{self.basis.describe()}  lambda={lam}  "
            f"edf={self.effective_dof:.1f}/{self.basis.n_params}  "
            f"cond={self.condition:.3g}  train RMSE {self.train_rmse_mm:.3f} mm "
            f"on {self.n_samples} placements"
        )


# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _penalty_root_cached(degree: int, kind: str) -> np.ndarray:
    return _penalty_root(BernsteinBasis(degree, kind))


@lru_cache(maxsize=64)
def _null_space_cached(degree: int, kind: str) -> np.ndarray:
    return _null_space(BernsteinBasis(degree, kind))


def _penalty_root(basis: BernsteinBasis) -> np.ndarray:
    """``S`` with ``S' S = Omega``, keeping only the non-null directions."""
    Omega = basis.curvature_penalty()
    if Omega.size == 0 or not np.any(np.abs(Omega) > 0):
        return np.zeros((0, basis.n_params))
    vals, vecs = np.linalg.eigh(Omega)
    scale = max(float(vals.max()), 1e-300)
    keep = vals > PENALTY_NULL_TOL * scale
    if not np.any(keep):
        return np.zeros((0, basis.n_params))
    return (np.sqrt(vals[keep])[:, None] * vecs[:, keep].T)


def _null_space(basis: BernsteinBasis) -> np.ndarray:
    """Basis of the curvature penalty's null space -- the affine functions.

    Shape ``(n_params, k)`` with ``k = min(4, n_params)``.  Used for
    ``lambda = inf``, where the fit is restricted to exactly that subspace, so
    the affine baseline of section 11 comes out of the same function as every
    other model rather than being special-cased somewhere else.
    """
    Omega = basis.curvature_penalty()
    if Omega.size == 0 or not np.any(np.abs(Omega) > 0):
        return np.eye(basis.n_params)
    vals, vecs = np.linalg.eigh(Omega)
    scale = max(float(vals.max()), 1e-300)
    return vecs[:, vals <= PENALTY_NULL_TOL * scale]


def solve(
    basis: BernsteinBasis,
    uvw,
    residuals_m,
    smoothing: float = 0.0,
    weights=None,
    design=None,
) -> tuple:
    """One penalised least-squares solve.  Returns ``(coefficients, edf, cond)``.

    ``design`` lets a caller hand in a design matrix it already built.  A
    leave-one-placement-out sweep over a smoothing grid evaluates the same basis
    at the same points thousands of times, and rebuilding it each time dominates
    the runtime.
    """
    A = basis.design(uvw) if design is None else np.asarray(design, dtype=np.float64)
    r = np.asarray(residuals_m, dtype=np.float64).reshape(len(A), 3)
    w = np.ones(len(A)) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.sqrt(np.maximum(w, 0.0))[:, None]

    if not np.isfinite(smoothing):
        # lambda = infinity: the fit lives in the penalty's null space.
        N = _null_space_cached(basis.degree, basis.kind)
        An = (A @ N) * sw
        beta, *_ = np.linalg.lstsq(An, r * sw, rcond=None)
        c = N @ beta
        s = np.linalg.svd(An, compute_uv=False)
        cond = float(np.inf if s[-1] <= 0 else s[0] / s[-1])
        return c, float(N.shape[1]), cond

    Aw = A * sw
    rw = r * sw
    if smoothing > 0.0:
        S = _penalty_root_cached(basis.degree, basis.kind) * np.sqrt(float(smoothing))
        stacked = np.vstack([Aw, S])
        rhs = np.vstack([rw, np.zeros((len(S), 3))])
    else:
        stacked, rhs = Aw, rw

    c, *_ = np.linalg.lstsq(stacked, rhs, rcond=None)

    # Effective degrees of freedom: trace of A (A'WA + lambda Omega)^-1 A'W.
    G = stacked.T @ stacked
    try:
        edf = float(np.trace(Aw @ np.linalg.solve(G, Aw.T)))
    except np.linalg.LinAlgError:
        edf = float(np.linalg.matrix_rank(stacked))
    s = np.linalg.svd(stacked, compute_uv=False)
    cond = float(np.inf if s[-1] <= 0 else s[0] / s[-1])
    return c, edf, cond


def fit(
    basis: BernsteinBasis,
    uvw,
    residuals_m,
    smoothing: float = 0.0,
    robust: Optional[str] = None,
    max_iter: int = 8,
    design=None,
) -> FitResult:
    """Fit ``basis`` to the residuals, optionally with a robust loss."""
    uvw = np.asarray(uvw, dtype=np.float64).reshape(-1, 3)
    r = np.asarray(residuals_m, dtype=np.float64).reshape(len(uvw), 3)
    if len(uvw) == 0:
        raise ValueError("cannot fit to no samples")
    A = basis.design(uvw) if design is None else np.asarray(design, dtype=np.float64)

    w = np.ones(len(uvw))
    c, edf, cond = solve(basis, uvw, r, smoothing, w, design=A)

    if robust == "huber":
        for _ in range(int(max_iter)):
            err = np.linalg.norm(r - A @ c, axis=1)
            # Median absolute deviation, scaled to a Gaussian sigma.
            scale = 1.4826 * np.median(np.abs(err - np.median(err)))
            scale = max(float(scale), MIN_ROBUST_SCALE_M)
            t = err / (HUBER_K * scale)
            w_new = np.where(t <= 1.0, 1.0, 1.0 / np.maximum(t, 1e-12))
            converged = np.max(np.abs(w_new - w)) < 1e-4
            w = w_new
            c, edf, cond = solve(basis, uvw, r, smoothing, w, design=A)
            if converged:
                break
    elif robust not in (None, "none"):
        raise ValueError(f"unknown robust loss {robust!r}; use None or 'huber'")

    pred = A @ c
    rmse = float(np.sqrt(np.mean(np.sum((r - pred) ** 2, axis=1))) * 1000.0)
    return FitResult(
        coefficients=c,
        basis=basis,
        smoothing=float(smoothing),
        effective_dof=edf,
        condition=cond,
        weights=w,
        train_rmse_mm=rmse,
        n_samples=len(uvw),
    )


# ---------------------------------------------------------------------------
def fit_dataset(
    dataset: CalibrationDataset,
    basis: BernsteinBasis,
    smoothing: float = 0.0,
    workspace: Optional[Workspace] = None,
    robust: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> ResidualModel:
    """Fit a :class:`~.models.ResidualModel` to a whole dataset.

    The workspace is taken from the dataset's own nominal points unless one is
    supplied.  Passing one explicitly matters during cross-validation: every
    fold must normalise against the *same* box, or the folds are not fitting
    the same function and their errors are not comparable.
    """
    if len(dataset) == 0:
        raise ValueError("cannot fit an empty dataset")
    p_nom = dataset.p_nom()
    ws = workspace or Workspace.from_points(p_nom)
    result = fit(basis, ws.normalise(p_nom), dataset.residuals(), smoothing, robust)

    meta = {
        "n_placements": len(dataset),
        "convention_digest": dataset.convention_digest(),
        "hand_eye": dataset.hand_eye.describe(),
        "hand_eye_T": dataset.hand_eye.T.tolist(),
        "grasp_point": dataset.grasp_point.describe(),
        "grasp_point_spec": dataset.grasp_point.as_dict(),
        "taught_orientation_quat_xyzw": (
            _mean_grasp_quat(dataset) if len(dataset) else None
        ),
        "effective_dof": result.effective_dof,
        "condition": result.condition,
        "train_rmse_mm": result.train_rmse_mm,
        "robust": robust or "none",
        "validated": False,
        **(metadata or {}),
    }
    if robust == "huber":
        downweighted = [
            (pid, float(wi))
            for pid, wi in zip(dataset.ids, result.weights)
            if wi < 0.999
        ]
        meta["downweighted"] = downweighted
    return ResidualModel(
        basis=basis,
        coefficients=result.coefficients,
        workspace=ws,
        smoothing=float(smoothing),
        metadata=meta,
    )


def _mean_grasp_quat(dataset: CalibrationDataset):
    """The wrist orientation the calibration was taught at (section 8).

    Recorded with the model so :mod:`.resolve` can warn when a later grasp is
    commanded at a different wrist angle -- the one change that moves the jaw
    offset without moving the position, and therefore the one change a
    position-only correction cannot see.
    """
    from scipy.spatial.transform import Rotation

    mats = np.array([p.grasp_rotation().as_matrix() for p in dataset.placements])
    return Rotation.from_matrix(mats).mean().as_quat().tolist()


def smoothing_grid(n_points: int = 9, low: float = -7.0, high: float = 3.0) -> np.ndarray:
    """Log-spaced smoothing weights, with the two endpoints of section 11.

    ``0.0`` is the unpenalised Bernstein fit and ``inf`` is the affine model, so
    the grid a cross-validation walks already contains both baselines.
    """
    return np.concatenate([[0.0], np.logspace(low, high, int(n_points)), [np.inf]])
