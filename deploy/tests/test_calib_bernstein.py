"""The Bernstein basis, checked against closed forms rather than itself.

Every identity here has an answer that is known independently -- a quadrature, a
finite difference, an exact polynomial -- so a bug in the basis cannot hide
behind a bug in the test.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from surgicai_rl_deploy.calib.bernstein import (
    MAX_DEGREE,
    BernsteinBasis,
    basis_1d,
    derivative_1d,
    elevate_1d,
    gram_1d,
    ladder,
    monomial_lift_1d,
    n_coefficients,
    total_indices,
)

RNG = np.random.default_rng(20260917)


# ---------------------------------------------------------------------------
# univariate identities
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", range(0, 6))
def test_partition_of_unity(n):
    u = np.concatenate([[0.0, 1.0], RNG.random(200)])
    assert np.abs(basis_1d(u, n).sum(axis=1) - 1.0).max() < 1e-13


@pytest.mark.parametrize("n", range(0, 5))
def test_basis_is_non_negative_on_the_unit_interval(n):
    """The property the whole numerical-stability argument rests on."""
    assert basis_1d(RNG.random(500), n).min() >= 0.0


@pytest.mark.parametrize("n,m", [(2, 2), (3, 1), (4, 4), (0, 3)])
def test_gram_matches_quadrature(n, m):
    G = gram_1d(n, m)
    for i in range(n + 1):
        for j in range(m + 1):
            numeric = quad(
                lambda t: basis_1d([t], n)[0, i] * basis_1d([t], m)[0, j], 0.0, 1.0
            )[0]
            assert abs(numeric - G[i, j]) < 1e-10


@pytest.mark.parametrize("n", range(1, 6))
def test_derivative_matches_finite_difference(n):
    c = RNG.normal(size=n + 1)
    D = derivative_1d(n)
    h, t = 1e-6, 0.37
    f = lambda x: basis_1d([x], n)[0] @ c  # noqa: E731
    numeric = (f(t + h) - f(t - h)) / (2 * h)
    exact = basis_1d([t], n - 1)[0] @ (D @ c)
    assert abs(numeric - exact) < 1e-7


@pytest.mark.parametrize("n", range(0, 5))
def test_degree_elevation_is_exact(n):
    c = RNG.normal(size=n + 1)
    u = RNG.random(100)
    assert np.abs(basis_1d(u, n) @ c - basis_1d(u, n + 1) @ (elevate_1d(n) @ c)).max() < 1e-12


@pytest.mark.parametrize("n", range(0, 5))
def test_monomial_lift_is_exact_and_bounded(n):
    u = RNG.random(200)
    for a in range(n + 1):
        c = monomial_lift_1d(a, n)
        assert np.abs(basis_1d(u, n) @ c - u ** a).max() < 1e-12
        # every coefficient in [0, 1] is what makes this lift well conditioned
        assert c.min() >= -1e-15 and c.max() <= 1.0 + 1e-15


def test_monomial_lift_refuses_too_high_a_power():
    with pytest.raises(ValueError):
        monomial_lift_1d(3, 2)


# ---------------------------------------------------------------------------
# the trivariate basis
# ---------------------------------------------------------------------------
def test_coefficient_counts():
    assert [n_coefficients(d, "total") for d in range(4)] == [1, 4, 10, 20]
    assert [n_coefficients(d, "tensor") for d in range(4)] == [1, 8, 27, 64]


def test_total_indices_are_distinct_and_correctly_sized():
    for n in range(5):
        idx = total_indices(n)
        assert len(set(idx)) == len(idx) == n_coefficients(n, "total")
        assert all(sum(t) <= n for t in idx)


@pytest.mark.parametrize("kind", ["total", "tensor"])
def test_reproduces_an_exact_polynomial(kind):
    """A degree-2 model must fit a quadratic with zero residual."""
    p = RNG.random((400, 3))
    x, y, z = p.T
    f = 1.2 - 0.4 * x + 2.1 * y * z - 0.9 * z ** 2 + 0.3 * x * y
    A = BernsteinBasis(2, kind).design(p)
    c, *_ = np.linalg.lstsq(A, f, rcond=None)
    assert np.abs(A @ c - f).max() < 1e-12


def test_degree_one_total_is_exactly_the_affine_model():
    """The claim the whole model ladder rests on: section 11's second baseline
    is the degree-1 rung, not a separate model."""
    p = RNG.random((300, 3))
    target = 0.7 - 1.3 * p[:, 0] + 0.2 * p[:, 1] + 4.0 * p[:, 2]
    A = BernsteinBasis(1, "total").design(p)
    assert A.shape[1] == 4
    c, *_ = np.linalg.lstsq(A, target, rcond=None)
    assert np.abs(A @ c - target).max() < 1e-12


def test_degree_zero_total_is_exactly_a_constant():
    A = BernsteinBasis(0, "total").design(RNG.random((50, 3)))
    assert A.shape[1] == 1
    assert np.abs(A - 1.0).max() < 1e-15


def test_the_ladder_is_nested():
    """Anything a degree-d model can represent, degree d+1 can too."""
    p = RNG.random((400, 3))
    for d in range(3):
        lo, hi = BernsteinBasis(d, "total"), BernsteinBasis(d + 1, "total")
        f = lo.design(p) @ RNG.normal(size=lo.n_params)
        A = hi.design(p)
        c, *_ = np.linalg.lstsq(A, f, rcond=None)
        assert np.abs(A @ c - f).max() < 1e-9


def test_total_degree_sits_inside_the_tensor_space():
    """The lift is what makes the total-degree family a Bernstein model."""
    p = RNG.random((200, 3))
    for d in (1, 2, 3):
        b = BernsteinBasis(d, "total")
        c = RNG.normal(size=(b.n_params, 1))
        direct = b.design(p) @ c
        via_tensor = BernsteinBasis(d, "tensor").design(p) @ (b.lift() @ c)
        assert np.abs(direct - via_tensor).max() < 1e-12


# ---------------------------------------------------------------------------
# the curvature penalty
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["total", "tensor"])
@pytest.mark.parametrize("degree", [2, 3])
def test_penalty_annihilates_exactly_the_affine_functions(kind, degree):
    b = BernsteinBasis(degree, kind)
    Omega = b.curvature_penalty()
    p = RNG.random((600, 3))
    A = b.design(p)

    affine = 0.3 + 1.1 * p[:, 0] - 2.0 * p[:, 1] + 0.7 * p[:, 2]
    c_aff, *_ = np.linalg.lstsq(A, affine, rcond=None)
    assert abs(c_aff @ Omega @ c_aff) < 1e-10

    # and the null space is EXACTLY four dimensional -- no more, no less
    rank = np.linalg.matrix_rank(Omega, tol=1e-8 * np.abs(Omega).max())
    assert rank == b.n_params - 4


@pytest.mark.parametrize("kind", ["total", "tensor"])
def test_penalty_value_is_the_true_curvature_energy(kind):
    """``u^2`` has ``f_uu = 2`` everywhere, so the energy is exactly 4."""
    b = BernsteinBasis(2, kind)
    p = RNG.random((600, 3))
    c, *_ = np.linalg.lstsq(b.design(p), p[:, 0] ** 2, rcond=None)
    assert abs(c @ b.curvature_penalty() @ c - 4.0) < 1e-8


def test_penalty_is_zero_below_degree_two():
    for d in (0, 1):
        for kind in ("total", "tensor"):
            Omega = BernsteinBasis(d, kind).curvature_penalty()
            assert Omega.shape[0] == n_coefficients(d, kind)
            assert np.abs(Omega).max() == 0.0


def test_penalty_is_positive_semidefinite():
    for d in (2, 3):
        vals = np.linalg.eigvalsh(BernsteinBasis(d, "total").curvature_penalty())
        assert vals.min() > -1e-8 * max(vals.max(), 1.0)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_conditioning_is_reported_and_stays_usable():
    """Bernstein is optimally stable for evaluation; the least-squares problem
    is a different question, and one worth being able to look at."""
    p = RNG.random((300, 3))
    conds = [BernsteinBasis(d, "total").conditioning(p) for d in range(4)]
    assert all(np.isfinite(conds))
    assert conds == sorted(conds)          # higher degree, worse conditioned
    assert conds[-1] < 1e4                 # still far from trouble at degree 3


def test_refuses_a_nonsense_basis():
    with pytest.raises(ValueError):
        BernsteinBasis(-1, "total")
    with pytest.raises(ValueError):
        BernsteinBasis(MAX_DEGREE + 1, "total")
    with pytest.raises(ValueError):
        BernsteinBasis(2, "chebyshev")


def test_ladder_names_the_two_protocol_baselines():
    names = [name for name, _ in ladder(3)]
    assert names[0] == "constant"
    assert names[1] == "affine"


def test_basis_survives_the_corners_of_the_cube():
    """A needle sitting exactly on a face of the box must not produce a NaN."""
    corners = np.array(
        [[a, b, c] for a in (0.0, 1.0) for b in (0.0, 1.0) for c in (0.0, 1.0)]
    )
    for d in range(4):
        A = BernsteinBasis(d, "total").design(corners)
        assert np.isfinite(A).all()
        assert np.abs(A.sum(axis=1) - 1.0).max() < 1e-12 or d > 0
