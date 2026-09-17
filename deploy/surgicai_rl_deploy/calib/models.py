"""The fitted correction field, its calibrated box, and its refusal to leave it.

A :class:`ResidualModel` is the object that section 3 of the protocol calls
``f``::

    p_grasp_hat = p_nom + f(p_nom)

It carries four things that have to travel together, because a coefficient
vector on its own is meaningless: the Bernstein basis it was fitted in, the
workspace box the inputs were normalised against, the smoothing weight, and
enough provenance to tell whether it still applies to the robot in front of you.

The bound that makes this safe to put in a control loop
-------------------------------------------------------
A Bernstein polynomial on ``[0, 1]^3`` lies inside the convex hull of its
control coefficients.  So for a model fitted in the tensor basis, and for the
total-degree models once lifted into it, the correction is bounded by the
largest coefficient -- *everywhere inside the box*, not just at the samples.
:meth:`ResidualModel.max_correction_mm` reports that bound, and
:mod:`.resolve` refuses to load a model whose bound exceeds what the operator
declared plausible.  A polynomial correction that can silently ask for a
four-centimetre move is the failure mode worth designing out, and the convex
hull property means it can be designed out by inspection rather than by testing.

Outside the box there is no such bound and a Bernstein polynomial diverges
quickly, which is why ``outside_policy`` defaults to ``"refuse"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from .bernstein import BernsteinBasis

#: Model file format version.  Bumped when the on-disk meaning changes, so a
#: stale file is refused rather than silently reinterpreted.
FORMAT_VERSION = 1

#: Default ceiling on how large a correction a model is allowed to command,
#: in millimetres.  Chosen as roughly the documented accuracy of a dVRK
#: hand-eye registration (the jhu-dvrk/dvrk_camera_registration README puts it
#: at "closer to 5mm to 10mm cube"), doubled.  A fitted correction larger than
#: this is not a refinement of that calibration, it is a disagreement with it,
#: and it should be looked at by a human before it drives an arm.
DEFAULT_MAX_CORRECTION_MM = 20.0


# ---------------------------------------------------------------------------
# the calibrated region
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Workspace:
    """The axis-aligned box the calibration covers, in the ECM frame, metres.

    Section 15 of the protocol: the mapping is local, and outside this box
    nothing has been measured.  The box is recorded with the model so the
    question "is this new needle inside the calibrated region" has one answer
    rather than one per script.
    """

    low_m: np.ndarray
    high_m: np.ndarray

    def __post_init__(self):
        low = np.asarray(self.low_m, dtype=np.float64).reshape(3)
        high = np.asarray(self.high_m, dtype=np.float64).reshape(3)
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError("workspace bounds must be finite")
        if not np.all(high > low):
            raise ValueError(
                "workspace must have positive extent on every axis; got "
                f"low={low.tolist()} high={high.tolist()}"
            )
        object.__setattr__(self, "low_m", low)
        object.__setattr__(self, "high_m", high)

    @classmethod
    def from_points(cls, points_m, pad_frac: float = 0.05, min_span_m: float = 2e-3):
        """Bounding box of the calibration samples, padded.

        ``pad_frac`` widens each axis by that fraction of its span, because the
        raw min and max are two samples, not a boundary: a new needle a tenth of
        a millimetre outside them is not meaningfully extrapolated and should
        not be refused.  ``min_span_m`` guards the degenerate case where every
        placement happened to share a coordinate -- a two-millimetre floor keeps
        the normalisation finite and, more usefully, makes the resulting model
        obviously local when it is printed.
        """
        pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            raise ValueError("cannot build a workspace from no points")
        low, high = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(high - low, min_span_m)
        pad = span * float(pad_frac)
        centre = 0.5 * (low + high)
        half = 0.5 * span + pad
        return cls(centre - half, centre + half)

    # -- use ---------------------------------------------------------------
    @property
    def span_m(self) -> np.ndarray:
        return self.high_m - self.low_m

    def normalise(self, points_m) -> np.ndarray:
        pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        return (pts - self.low_m) / self.span_m

    def denormalise(self, uvw) -> np.ndarray:
        uvw = np.asarray(uvw, dtype=np.float64).reshape(-1, 3)
        return uvw * self.span_m + self.low_m

    def outside(self, point_m, tol_m: float = 0.0) -> list:
        """Per-axis reasons this point sits outside the calibrated box."""
        p = np.asarray(point_m, dtype=np.float64).reshape(3)
        axes = "xyz"
        out = []
        for i in range(3):
            if p[i] < self.low_m[i] - tol_m or p[i] > self.high_m[i] + tol_m:
                out.append(
                    f"{axes[i]} {p[i]*100:+.2f} cm outside calibrated "
                    f"[{self.low_m[i]*100:+.2f}, {self.high_m[i]*100:+.2f}] cm"
                )
        return out

    def contains(self, point_m, tol_m: float = 0.0) -> bool:
        return not self.outside(point_m, tol_m)

    def describe(self) -> str:
        s = self.span_m * 100.0
        c = 0.5 * (self.low_m + self.high_m) * 100.0
        return (
            f"{s[0]:.2f} x {s[1]:.2f} x {s[2]:.2f} cm box centred at "
            f"({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}) cm"
        )

    def as_dict(self) -> dict:
        return {"low_m": self.low_m.tolist(), "high_m": self.high_m.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Workspace":
        return cls(np.asarray(d["low_m"]), np.asarray(d["high_m"]))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResidualModel:
    """A fitted task-space correction ``f: p_nom -> r``.

    ``coefficients`` has shape ``(basis.n_params, 3)`` and is in **metres**, the
    same units as the poses everything else in this package passes around.
    """

    basis: BernsteinBasis
    coefficients: np.ndarray
    workspace: Workspace
    #: curvature smoothing weight the fit used; ``inf`` means "affine, exactly"
    smoothing: float = 0.0
    #: what to do when asked to correct a point outside the box
    outside_policy: str = "refuse"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        c = np.asarray(self.coefficients, dtype=np.float64)
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        if c.shape != (self.basis.n_params, 3):
            raise ValueError(
                f"coefficients must be ({self.basis.n_params}, 3) for "
                f"{self.basis.describe()}; got {c.shape}"
            )
        if not np.isfinite(c).all():
            raise ValueError("model coefficients must be finite")
        if self.outside_policy not in ("refuse", "clamp", "allow"):
            raise ValueError(
                f"outside_policy must be refuse, clamp or allow; "
                f"got {self.outside_policy!r}"
            )
        object.__setattr__(self, "coefficients", c)

    # -- prediction --------------------------------------------------------
    def residual(self, points_m) -> np.ndarray:
        """The correction ``f(p)`` at each point, metres, shape ``(N, 3)``.

        Honours ``outside_policy``.  ``"refuse"`` raises, which is the default
        because section 15 says the mapping is local and a Bernstein polynomial
        evaluated outside its box does not fail gracefully -- it grows like the
        degree.
        """
        pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        uvw = self.workspace.normalise(pts)
        if self.outside_policy == "refuse":
            for p in pts:
                reasons = self.workspace.outside(p)
                if reasons:
                    raise OutsideCalibratedRegion(p, reasons, self.workspace)
        elif self.outside_policy == "clamp":
            uvw = np.clip(uvw, 0.0, 1.0)
        return self.basis.evaluate(uvw, self.coefficients)

    def correct(self, points_m) -> np.ndarray:
        """``p_nom + f(p_nom)`` -- the predicted successful grasp position."""
        pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 3)
        return pts + self.residual(pts)

    def __call__(self, point_m) -> np.ndarray:
        return self.correct(point_m)[0]

    # -- safety ------------------------------------------------------------
    def max_correction_mm(self) -> float:
        """Hard upper bound on ``||f(p)||`` anywhere inside the box, in mm.

        By the convex hull property of the Bernstein basis, a value of the
        polynomial at any point of ``[0, 1]^3`` is a convex combination of its
        tensor-Bernstein control coefficients, so its norm is at most the
        largest coefficient norm.  This is a bound, not an estimate: no sampling
        is involved and no point inside the box can exceed it.
        """
        tensor_coeff = self.basis.lift() @ self.coefficients
        return float(np.linalg.norm(tensor_coeff, axis=1).max() * 1000.0)

    def check_sane(self, max_correction_mm: float = DEFAULT_MAX_CORRECTION_MM) -> list:
        """Reasons this model should not be trusted to drive an arm."""
        problems = []
        bound = self.max_correction_mm()
        if bound > float(max_correction_mm):
            problems.append(
                f"the model can command corrections up to {bound:.1f} mm inside "
                f"its own box, over the {max_correction_mm:.1f} mm ceiling -- "
                "that is a disagreement with the hand-eye calibration, not a "
                "refinement of it"
            )
        span = self.workspace.span_m * 100.0
        if span.min() < 0.5:
            problems.append(
                f"the calibrated box is only {span.min():.2f} cm across on one "
                "axis: the fit has no leverage in that direction and the model "
                "is effectively constant along it"
            )
        if not self.metadata.get("validated", False):
            problems.append(
                "no cross-validated error was recorded with this model "
                "(fit it through tools/fit_grasp_calibration.py)"
            )
        return problems

    # -- housekeeping ------------------------------------------------------
    def with_policy(self, policy: str) -> "ResidualModel":
        return replace(self, outside_policy=policy)

    def describe(self) -> str:
        lam = "inf (affine)" if not np.isfinite(self.smoothing) else f"{self.smoothing:.3g}"
        return (
            f"{self.basis.describe()}, smoothing {lam}, "
            f"box {self.workspace.describe()}, "
            f"bounded by {self.max_correction_mm():.2f} mm"
        )

    def as_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "basis": {"degree": self.basis.degree, "kind": self.basis.kind},
            "coefficients_m": self.coefficients.tolist(),
            "workspace": self.workspace.as_dict(),
            "smoothing": (None if not np.isfinite(self.smoothing) else float(self.smoothing)),
            "outside_policy": self.outside_policy,
            "max_correction_mm": self.max_correction_mm(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualModel":
        version = int(d.get("format_version", 0))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"calibration file format version {version}, this package "
                f"writes and reads {FORMAT_VERSION}; refit rather than guess"
            )
        smoothing = d.get("smoothing")
        return cls(
            basis=BernsteinBasis(int(d["basis"]["degree"]), str(d["basis"]["kind"])),
            coefficients=np.asarray(d["coefficients_m"], dtype=np.float64),
            workspace=Workspace.from_dict(d["workspace"]),
            smoothing=(np.inf if smoothing is None else float(smoothing)),
            outside_policy=str(d.get("outside_policy", "refuse")),
            metadata=dict(d.get("metadata", {})),
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path) -> "ResidualModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


class OutsideCalibratedRegion(RuntimeError):
    """Raised when a model is asked to correct a point it never saw.

    Carries the point and the box so a caller can report the miss in the same
    words the precheck uses rather than re-deriving them.
    """

    def __init__(self, point_m, reasons, workspace: Workspace):
        self.point_m = np.asarray(point_m, dtype=np.float64).reshape(3)
        self.reasons = list(reasons)
        self.workspace = workspace
        super().__init__(
            "needle is outside the calibrated region: " + "; ".join(self.reasons)
        )


# ---------------------------------------------------------------------------
def zero_model(workspace: Workspace, **kw) -> ResidualModel:
    """The "no correction" model of section 11's first baseline.

    A real object rather than a ``None`` special case, so that every comparison
    in :mod:`.validate` runs the same code path and the uncorrected geometric
    estimate is scored by exactly the machinery that scores everything else.
    """
    basis = BernsteinBasis(0, "total")
    return ResidualModel(
        basis=basis,
        coefficients=np.zeros((basis.n_params, 3)),
        workspace=workspace,
        smoothing=np.inf,
        outside_policy=kw.pop("outside_policy", "allow"),
        metadata={"model": "no correction", "validated": True, **kw},
    )
