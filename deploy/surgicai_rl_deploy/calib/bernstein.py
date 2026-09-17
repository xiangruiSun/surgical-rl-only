"""Trivariate Bernstein polynomials on the unit cube, and the one algebraic
fact that makes the whole calibration a single nested family.

Why Bernstein at all
--------------------
Xiao et al., *Delta Robot Kinematic Calibration for Precise Robot-Assisted
Retinal Surgery* (ISMR 2022) fit the error residual left over after a geometric
calibration with a Bernstein polynomial, and reduced a delta robot's residual
to under 20 micrometres.  That is the method this module transcribes.  Two
things are done differently here, and both are deliberate:

1. **The fit is linear, so it is solved linearly.**  The reference solves (13)
   with Levenberg-Marquardt.  ``BP(x, y, C, n)`` is a *linear* function of the
   coefficient matrix ``C`` -- every basis function is fixed once the degree is
   chosen -- so the normal equations have a closed form and LM is a nonlinear
   solver being run on a linear problem.  It converges to the same answer when
   it converges, and it can stall or land on a local step limit when it does
   not.  :mod:`.fit` uses a QR least-squares solve instead: one shot, no
   iteration count, no initial guess.

2. **The degree ladder starts below linear.**  The reference chose between
   degrees 2 and 6 by validation RMS.  Here the family starts at degree 0.
   That matters because of the algebraic fact below.

The one algebraic fact
----------------------
A trivariate polynomial of total degree ``n`` restricted to the unit cube is a
member of the tensor-product Bernstein space of degree ``n`` per axis: the
monomial lift

    u^a  =  sum_i  [ C(i, a) / C(n, a) ]  B_i^n(u)                       (exact)

carries every monomial of degree ``a <= n`` into Bernstein coefficients, all of
them in ``[0, 1]``.  So the total-degree-``n`` models for ``n = 0, 1, 2, 3``
form a *nested ladder inside one basis*:

    n = 0   ->  1 coefficient   ->  a constant offset       (section 11 baseline)
    n = 1   ->  4 coefficients  ->  r = A p + b             (section 11 baseline)
    n = 2   ->  10 coefficients
    n = 3   ->  20 coefficients

The two baselines the protocol asks to compare against are not separate models
to be bolted on beside the Bernstein one.  They are its bottom two rungs.  Model
selection is therefore a single one-dimensional choice of degree, and "does the
polynomial beat a constant offset" is answered by the same cross-validation
loop that chooses the degree, not by a second experiment.

Tensor-product degree ``n`` is also offered, and it is what the reference used.
It costs ``(n + 1)^3`` coefficients against total degree's ``C(n + 3, 3)``:

    degree      1       2       3       4
    total       4      10      20      35
    tensor      8      27      64     125

With the number of *physical needle placements* a human can hand-teach in a
session -- tens, not thousands -- that difference decides whether the model is
estimable at all.  See :func:`~.validate.variance_budget`.

Frame invariance, which is not a detail
---------------------------------------
The residual can be modelled as a function of the nominal ECM position or of the
camera-frame position; the two differ by a fixed rigid transform.  A *total
degree* space is closed under that transform -- an affine change of variables
maps polynomials of total degree ``n`` to polynomials of total degree ``n`` --
so the total-degree model is the same model either way.  A *tensor-product*
space is not: a rotation mixes the axes and pushes per-axis degree up.  Choosing
tensor product means the answer depends on which frame the polynomial was
written in, which is a property no one wants and nobody checks for.

Conventions
-----------
All inputs to this module are already normalised into ``[0, 1]^3``.  The
normalisation itself, and the refusal to extrapolate outside it, live in
:mod:`.models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Iterator, Tuple

import numpy as np

#: Highest degree this module will build.  Not a numerical limit -- the
#: Bernstein basis evaluates happily far above it -- but a statistical one: a
#: degree-4 total model is 35 coefficients per axis, and nothing in section 2 of
#: the protocol document suggests a residual field with that much structure over
#: a four-centimetre box.  Raise it deliberately, with data that justifies it.
MAX_DEGREE = 6


# ---------------------------------------------------------------------------
# univariate basis
# ---------------------------------------------------------------------------
def basis_1d(u, n: int) -> np.ndarray:
    """``B_i^n(u) = C(n, i) u^i (1 - u)^(n - i)`` for ``i = 0..n``.

    Returns shape ``(len(u), n + 1)``.  Evaluated directly from the definition
    rather than by de Casteljau: at the degrees this package uses the two agree
    to the last bit, and the direct form vectorises.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if n < 0:
        raise ValueError("degree must be non-negative")
    i = np.arange(n + 1)
    coeff = np.array([comb(n, int(k)) for k in i], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = coeff * np.power.outer(u, i) * np.power.outer(1.0 - u, n - i)
    # 0^0 is 1 here, which np.power already gives, but guard the endpoints so a
    # point sitting exactly on a face of the cube cannot produce a NaN.
    return np.nan_to_num(out, nan=0.0)


def gram_1d(n: int, m: int) -> np.ndarray:
    """Exact ``G[i, j] = integral_0^1 B_i^n(u) B_j^m(u) du``.

    Closed form, no quadrature::

        integral B_i^n B_j^m = C(n,i) C(m,j) / [ C(n+m, i+j) (n + m + 1) ]

    which follows from ``B_i^n B_j^m = [C(n,i)C(m,j)/C(n+m,i+j)] B_{i+j}^{n+m}``
    and ``integral B_k^N = 1/(N+1)``.
    """
    G = np.empty((n + 1, m + 1), dtype=np.float64)
    for i in range(n + 1):
        for j in range(m + 1):
            G[i, j] = (
                comb(n, i) * comb(m, j) / (comb(n + m, i + j) * (n + m + 1))
            )
    return G


def derivative_1d(n: int) -> np.ndarray:
    """Matrix ``D`` with ``d/du [c . B^n] = (D c) . B^(n-1)``.

    ``D[i, i] = -n``, ``D[i, i+1] = +n``; shape ``(n, n + 1)``.  Exact, and the
    reason the curvature penalty below needs no quadrature and no monomials.
    """
    if n < 1:
        return np.zeros((0, 1), dtype=np.float64)
    D = np.zeros((n, n + 1), dtype=np.float64)
    for i in range(n):
        D[i, i] = -float(n)
        D[i, i + 1] = float(n)
    return D


def elevate_1d(n: int) -> np.ndarray:
    """Degree elevation ``E`` with ``c . B^n = (E c) . B^(n+1)``.

    ``c'_i = (i/(n+1)) c_{i-1} + (1 - i/(n+1)) c_i``.  Used to express a fitted
    low-degree model in a higher-degree basis, which is how the nesting claim in
    this module's docstring is *tested* rather than asserted.
    """
    E = np.zeros((n + 2, n + 1), dtype=np.float64)
    for i in range(n + 2):
        a = i / (n + 1.0)
        if i - 1 >= 0:
            E[i, i - 1] += a
        if i <= n:
            E[i, i] += 1.0 - a
    return E


def monomial_lift_1d(a: int, n: int) -> np.ndarray:
    """Bernstein coefficients of ``u^a`` in the degree-``n`` basis.

    ``u^a = sum_i [C(i,a)/C(n,a)] B_i^n(u)``, exact, every coefficient in
    ``[0, 1]``.  This is the map that puts the total-degree family inside the
    tensor-product one.
    """
    if a > n:
        raise ValueError(f"cannot lift u^{a} into a degree-{n} basis")
    i = np.arange(n + 1)
    return np.array(
        [comb(int(k), a) / comb(n, a) for k in i], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# trivariate index sets
# ---------------------------------------------------------------------------
def tensor_indices(n: int) -> list:
    return [(i, j, k) for i in range(n + 1) for j in range(n + 1) for k in range(n + 1)]


def total_indices(n: int) -> list:
    """Multi-indices ``(a, b, c)`` with ``a + b + c <= n``, ordered by degree."""
    out = []
    for total in range(n + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                out.append((a, b, total - a - b))
    return out


def n_coefficients(degree: int, kind: str) -> int:
    if kind == "tensor":
        return (degree + 1) ** 3
    if kind == "total":
        return comb(degree + 3, 3)
    raise ValueError(f"unknown basis kind {kind!r}")


# ---------------------------------------------------------------------------
# the basis object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BernsteinBasis:
    """A trivariate Bernstein space on ``[0, 1]^3``.

    ``kind="tensor"``
        Degree ``n`` in each variable, ``(n+1)^3`` free coefficients.  What
        Xiao et al. use (in two variables).
    ``kind="total"``
        Total degree ``n``, ``C(n+3, 3)`` free coefficients, represented inside
        the tensor-product Bernstein basis through the exact monomial lift.
        Evaluation is therefore still Bernstein -- the numerically well-behaved
        part -- while the parameterisation is the parsimonious one.

    Either way :meth:`design` returns the matrix a linear least-squares solve
    consumes, and :meth:`curvature_penalty` returns the matrix that measures how
    bent the resulting surface is.
    """

    degree: int
    kind: str = "total"

    def __post_init__(self):
        if not (0 <= int(self.degree) <= MAX_DEGREE):
            raise ValueError(
                f"degree must be in [0, {MAX_DEGREE}]; got {self.degree!r}"
            )
        if self.kind not in ("tensor", "total"):
            raise ValueError(f"basis kind must be 'tensor' or 'total'; got {self.kind!r}")

    # -- size --------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return n_coefficients(self.degree, self.kind)

    @property
    def tensor_degree(self) -> int:
        """Per-axis degree of the tensor space this basis lives inside."""
        return self.degree

    def __len__(self) -> int:
        return self.n_params

    def describe(self) -> str:
        return f"Bernstein {self.kind} degree {self.degree} ({self.n_params} coeff/axis)"

    # -- the lift ----------------------------------------------------------
    def lift(self) -> np.ndarray:
        """``L`` mapping free parameters to tensor-Bernstein coefficients.

        Shape ``((n+1)^3, n_params)``.  Identity for ``kind="tensor"``.
        """
        return _lift(self.degree, self.kind)

    # -- evaluation --------------------------------------------------------
    def tensor_design(self, uvw) -> np.ndarray:
        """``(N, (n+1)^3)`` matrix of tensor-Bernstein basis values."""
        uvw = np.asarray(uvw, dtype=np.float64).reshape(-1, 3)
        n = self.degree
        Bu = basis_1d(uvw[:, 0], n)
        Bv = basis_1d(uvw[:, 1], n)
        Bw = basis_1d(uvw[:, 2], n)
        return np.einsum("ni,nj,nk->nijk", Bu, Bv, Bw).reshape(len(uvw), -1)

    def design(self, uvw) -> np.ndarray:
        """``(N, n_params)`` design matrix for a least-squares fit."""
        A = self.tensor_design(uvw)
        return A if self.kind == "tensor" else A @ self.lift()

    def evaluate(self, uvw, coefficients) -> np.ndarray:
        """``(N, m)`` values, for coefficients of shape ``(n_params, m)``."""
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(-1, 1)
        return self.design(uvw) @ coefficients

    # -- smoothness --------------------------------------------------------
    def curvature_penalty(self) -> np.ndarray:
        """``Omega`` with ``c' Omega c = integral over the cube of the squared
        second-derivative (thin-plate) energy::

            f_uu^2 + f_vv^2 + f_ww^2 + 2 f_uv^2 + 2 f_uw^2 + 2 f_vw^2

        Exact: every second derivative of a Bernstein polynomial is a Bernstein
        polynomial two degrees down with differenced coefficients, and the L2
        inner products of Bernstein basis functions have the closed form in
        :func:`gram_1d`.  No quadrature, no monomial round-trip.

        **Omega annihilates exactly the affine functions.**  That is the whole
        point of penalising curvature rather than coefficient size: a ridge
        penalty on the coefficients would shrink the hand-eye term, which
        section 2 of the protocol document shows is the physically meaningful,
        exactly-affine part of the residual.  A curvature penalty shrinks only
        what is bent.  So the regularisation path runs

            lambda -> infinity   the affine model   (section 11's second baseline)
            lambda -> 0          the unconstrained Bernstein fit

        and choosing ``lambda`` by cross-validation *is* choosing between them.
        """
        return _curvature_penalty(self.degree, self.kind)

    # -- diagnostics -------------------------------------------------------
    def conditioning(self, uvw) -> float:
        """Condition number of the design matrix at these sample points.

        Worth printing before trusting a fit.  The Bernstein basis is optimally
        stable for *evaluation*; its Gram matrix is not well conditioned, and
        at a high degree with few scattered samples the least-squares problem
        can be badly posed long before the evaluation is.
        """
        A = self.design(uvw)
        s = np.linalg.svd(A, compute_uv=False)
        return float(np.inf if s[-1] <= 0 else s[0] / s[-1])


# ---------------------------------------------------------------------------
# cached construction
# ---------------------------------------------------------------------------
# A basis is defined entirely by (degree, kind) and both matrices below are
# expensive relative to a single least-squares solve -- and a cross-validation
# builds them once per fold per smoothing weight, which is thousands of times.
# They are pure functions of two small integers, so they are cached.
@lru_cache(maxsize=64)
def _lift(degree: int, kind: str) -> np.ndarray:
    n = int(degree)
    if kind == "tensor":
        return np.eye((n + 1) ** 3, dtype=np.float64)
    cols = []
    for a, b, c in total_indices(n):
        block = np.einsum(
            "i,j,k->ijk",
            monomial_lift_1d(a, n),
            monomial_lift_1d(b, n),
            monomial_lift_1d(c, n),
        )
        cols.append(block.reshape(-1))
    out = np.column_stack(cols)
    out.flags.writeable = False
    return out


@lru_cache(maxsize=64)
def _curvature_penalty(degree: int, kind: str) -> np.ndarray:
    n = int(degree)
    m = n + 1
    size = m ** 3
    n_params = n_coefficients(n, kind)
    if n < 2:
        out = np.zeros((n_params, n_params), dtype=np.float64)
        out.flags.writeable = False
        return out

    D1 = derivative_1d(n)
    D2 = derivative_1d(n - 1) @ D1
    G = {d: gram_1d(d, d) for d in range(n - 2, n + 1)}

    def axis_op(order: int):
        if order == 0:
            return np.eye(m), n
        if order == 1:
            return D1, n - 1
        return D2, n - 2

    Omega = np.zeros((size, size), dtype=np.float64)
    terms = [
        ((2, 0, 0), 1.0), ((0, 2, 0), 1.0), ((0, 0, 2), 1.0),
        ((1, 1, 0), 2.0), ((1, 0, 1), 2.0), ((0, 1, 1), 2.0),
    ]
    for orders, weight in terms:
        ops, degs = zip(*(axis_op(o) for o in orders))
        if any(op.shape[0] == 0 for op in ops):
            continue
        blocks = [ops[ax].T @ G[degs[ax]] @ ops[ax] for ax in range(3)]
        Omega += weight * np.einsum(
            "ad,be,cf->abcdef", blocks[0], blocks[1], blocks[2]
        ).reshape(size, size)

    if kind == "total":
        L = _lift(n, kind)
        Omega = L.T @ Omega @ L
    out = 0.5 * (Omega + Omega.T)
    out.flags.writeable = False
    return out


def ladder(max_degree: int = 3, kind: str = "total") -> Iterator[Tuple[str, BernsteinBasis]]:
    """The nested family, cheapest first.

    ``("constant", deg 0)``, ``("affine", deg 1)``, then one rung per degree.
    Naming the bottom two rungs is not cosmetic: they are exactly the two
    baselines section 11 of the protocol asks for, so a report that walks this
    ladder has already run that comparison.
    """
    names = {0: "constant", 1: "affine"}
    for d in range(max_degree + 1):
        yield names.get(d, f"{kind} degree {d}"), BernsteinBasis(degree=d, kind=kind)
