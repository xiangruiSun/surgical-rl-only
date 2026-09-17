"""The fitted model, its box, and the bound that makes it safe to publish.

The two things worth testing hardest are the ones that stop a bad model from
driving an arm: that the convex-hull bound really is a bound, and that a point
outside the calibrated box is refused rather than extrapolated.
"""

import json

import numpy as np
import pytest

from surgicai_rl_deploy.calib.bernstein import BernsteinBasis
from surgicai_rl_deploy.calib.models import (
    DEFAULT_MAX_CORRECTION_MM,
    FORMAT_VERSION,
    OutsideCalibratedRegion,
    ResidualModel,
    Workspace,
    zero_model,
)

RNG = np.random.default_rng(11)


def box():
    return Workspace([-0.02, 0.0, 0.07], [0.02, 0.04, 0.10])


def model(degree=2, kind="total", scale=0.002, ws=None):
    b = BernsteinBasis(degree, kind)
    return ResidualModel(
        basis=b,
        coefficients=RNG.normal(size=(b.n_params, 3)) * scale,
        workspace=ws or box(),
        metadata={"validated": True},
    )


# ---------------------------------------------------------------------------
# the box
# ---------------------------------------------------------------------------
def test_normalisation_round_trips():
    ws = box()
    p = RNG.uniform(ws.low_m, ws.high_m, size=(200, 3))
    assert np.abs(ws.denormalise(ws.normalise(p)) - p).max() < 1e-15


def test_corners_normalise_to_zero_and_one():
    ws = box()
    assert np.allclose(ws.normalise(ws.low_m), 0.0)
    assert np.allclose(ws.normalise(ws.high_m), 1.0)


def test_from_points_pads_and_names_the_axis_that_is_outside():
    p = RNG.uniform([-0.01, 0.0, 0.08], [0.01, 0.02, 0.09], size=(50, 3))
    ws = Workspace.from_points(p, pad_frac=0.05)
    assert all(ws.contains(q) for q in p)
    reasons = ws.outside([0.5, 0.0, 0.085])
    assert len(reasons) == 1 and reasons[0].startswith("x")


def test_from_points_survives_a_degenerate_axis():
    """Every placement sharing a coordinate must not produce a zero-width box."""
    p = np.column_stack([RNG.uniform(-0.01, 0.01, 40),
                         RNG.uniform(-0.01, 0.01, 40),
                         np.full(40, 0.085)])
    ws = Workspace.from_points(p)
    assert ws.span_m[2] > 0.0
    assert all(ws.contains(q) for q in p)


def test_rejects_an_inverted_box():
    with pytest.raises(ValueError, match="positive extent"):
        Workspace([0.0, 0.0, 0.0], [0.0, 1.0, 1.0])


def test_box_round_trips_through_json():
    ws = box()
    back = Workspace.from_dict(json.loads(json.dumps(ws.as_dict())))
    assert np.allclose(back.low_m, ws.low_m) and np.allclose(back.high_m, ws.high_m)


# ---------------------------------------------------------------------------
# the convex-hull bound
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["total", "tensor"])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_max_correction_is_a_true_bound_not_an_estimate(kind, degree):
    """Bernstein's convex-hull property, used as a safety guarantee.

    Nothing inside the box may exceed the bound -- checked against a dense
    sample including every corner, edge midpoint and face centre.
    """
    m = model(degree, kind)
    bound = m.max_correction_mm()
    grid = np.array(
        [[a, b, c] for a in (0, 0.5, 1) for b in (0, 0.5, 1) for c in (0, 0.5, 1)]
    )
    uvw = np.vstack([grid, RNG.random((4000, 3))])
    pts = m.workspace.denormalise(uvw)
    got = np.linalg.norm(m.residual(pts), axis=1) * 1000.0
    assert got.max() <= bound + 1e-9


def test_the_bound_is_not_vacuous():
    """A bound that is always a thousand times the truth is not a safety check."""
    m = model(2, "total")
    uvw = RNG.random((5000, 3))
    got = np.linalg.norm(m.residual(m.workspace.denormalise(uvw)), axis=1) * 1000.0
    assert got.max() > 0.2 * m.max_correction_mm()


def test_check_sane_flags_an_implausibly_large_correction():
    m = model(2, "total", scale=0.05)
    problems = m.check_sane()
    assert any("ceiling" in p for p in problems)


def test_check_sane_flags_a_flat_box():
    ws = Workspace([-0.02, 0.0, 0.08], [0.02, 0.04, 0.0801])
    assert any("only" in p for p in model(1, ws=ws).check_sane())


def test_check_sane_flags_an_unvalidated_model():
    b = BernsteinBasis(1, "total")
    m = ResidualModel(b, np.zeros((b.n_params, 3)), box())
    assert any("cross-validated" in p for p in m.check_sane())


# ---------------------------------------------------------------------------
# the refusal
# ---------------------------------------------------------------------------
def test_refuses_a_point_outside_the_box_by_default():
    m = model()
    with pytest.raises(OutsideCalibratedRegion) as exc:
        m.residual([[0.5, 0.0, 0.085]])
    assert exc.value.reasons and "x" in exc.value.reasons[0]
    assert exc.value.workspace is m.workspace


def test_clamp_keeps_the_correction_inside_the_bound():
    m = model().with_policy("clamp")
    far = np.array([[10.0, -10.0, 10.0]])
    assert np.linalg.norm(m.residual(far)) * 1000.0 <= m.max_correction_mm() + 1e-9


def test_allow_extrapolates_and_that_is_why_it_is_not_the_default():
    m = model(3).with_policy("allow")
    inside = np.linalg.norm(m.residual([m.workspace.denormalise([[0.5, 0.5, 0.5]])[0]]))
    outside = np.linalg.norm(m.residual([[0.5, 0.5, 0.5]]))
    assert outside > 100 * inside


def test_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="outside_policy"):
        model().with_policy("shrug")


# ---------------------------------------------------------------------------
# shape, io, versioning
# ---------------------------------------------------------------------------
def test_rejects_wrongly_shaped_coefficients():
    b = BernsteinBasis(2, "total")
    with pytest.raises(ValueError, match="coefficients must be"):
        ResidualModel(b, np.zeros((b.n_params, 2)), box())


def test_rejects_non_finite_coefficients():
    b = BernsteinBasis(1, "total")
    c = np.zeros((b.n_params, 3))
    c[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ResidualModel(b, c, box())


def test_correct_is_nominal_plus_residual():
    m = model()
    p = m.workspace.denormalise(RNG.random((20, 3)))
    assert np.allclose(m.correct(p), p + m.residual(p))


def test_round_trips_through_a_file(tmp_path):
    m = model(2, "tensor")
    path = m.save(tmp_path / "cal.json")
    back = ResidualModel.load(path)
    assert back.basis == m.basis
    assert np.allclose(back.coefficients, m.coefficients)
    assert np.allclose(back.workspace.low_m, m.workspace.low_m)
    p = m.workspace.denormalise(RNG.random((50, 3)))
    assert np.allclose(back.residual(p), m.residual(p))


def test_infinite_smoothing_survives_the_round_trip(tmp_path):
    b = BernsteinBasis(2, "total")
    m = ResidualModel(b, np.zeros((b.n_params, 3)), box(), smoothing=np.inf)
    back = ResidualModel.load(m.save(tmp_path / "c.json"))
    assert not np.isfinite(back.smoothing)


def test_refuses_a_file_from_another_format_version(tmp_path):
    m = model()
    d = m.as_dict()
    d["format_version"] = FORMAT_VERSION + 1
    path = tmp_path / "old.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="format version"):
        ResidualModel.load(path)


# ---------------------------------------------------------------------------
def test_zero_model_corrects_nothing_anywhere():
    z = zero_model(box())
    p = np.array([[0.0, 0.02, 0.085], [10.0, 10.0, 10.0]])
    assert np.abs(z.residual(p)).max() == 0.0
    assert np.allclose(z.correct(p), p)
    assert z.max_correction_mm() == 0.0
    assert z.check_sane() == []


def test_default_ceiling_is_documented_in_millimetres():
    assert 5.0 < DEFAULT_MAX_CORRECTION_MM < 100.0
