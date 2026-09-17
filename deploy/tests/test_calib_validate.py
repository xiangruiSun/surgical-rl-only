"""Fitting, cross-validation and model selection.

Two kinds of test here.  The first kind checks arithmetic against a known
answer: an exactly affine field is recovered exactly, the penalty path ends at
the affine model, a fold never splits a placement.  The second kind checks the
*procedure* -- that the selector adopts curvature when there is curvature and
declines when there is not.  The second kind matters more: a selection rule that
cannot say no is not a selection rule, and nothing else in this package would
notice.
"""

import numpy as np
import pytest

from surgicai_rl_deploy.calib.bernstein import BernsteinBasis
from surgicai_rl_deploy.calib.fit import fit, fit_dataset, smoothing_grid
from surgicai_rl_deploy.calib.models import Workspace
from surgicai_rl_deploy.calib.simulate import (
    SyntheticWorld,
    make_dataset,
    saddle_bias,
)
from surgicai_rl_deploy.calib.validate import (
    cross_validate,
    grouped_folds,
    paired_improvement,
    placements_needed,
    report,
    select_model,
    spatial_holdout,
    summarise,
    uncorrected,
    variance_budget,
)

RNG = np.random.default_rng(5)
FAST = np.array([0.0, 1e-4, 1e-2, 1.0, np.inf])


# ---------------------------------------------------------------------------
# the fit itself
# ---------------------------------------------------------------------------
def test_recovers_an_exactly_affine_field_exactly():
    """Which is the case section 2 argues is the physically expected one."""
    uvw = RNG.random((80, 3))
    A = np.array([[0.02, -0.01, 0.003], [0.0, 0.015, -0.002], [0.01, 0.0, 0.02]])
    b = np.array([0.004, -0.0025, 0.0055])
    r = uvw @ A.T + b
    res = fit(BernsteinBasis(1, "total"), uvw, r)
    assert res.train_rmse_mm < 1e-9
    assert res.effective_dof == pytest.approx(4.0, abs=1e-6)


def test_a_higher_degree_still_recovers_an_affine_field_exactly():
    uvw = RNG.random((120, 3))
    r = uvw @ np.eye(3) * 0.01 + 0.002
    for degree in (2, 3):
        assert fit(BernsteinBasis(degree, "total"), uvw, r).train_rmse_mm < 1e-8


def test_infinite_smoothing_is_exactly_the_affine_fit():
    """The regularisation path's far end is section 11's second baseline."""
    uvw = RNG.random((100, 3))
    r = RNG.normal(size=(100, 3)) * 0.001 + uvw * 0.005
    affine = fit(BernsteinBasis(1, "total"), uvw, r)
    for degree in (2, 3):
        penalised = fit(BernsteinBasis(degree, "total"), uvw, r, smoothing=np.inf)
        pa = BernsteinBasis(1, "total").design(uvw) @ affine.coefficients
        pp = BernsteinBasis(degree, "total").design(uvw) @ penalised.coefficients
        assert np.abs(pa - pp).max() < 1e-9
        assert penalised.effective_dof == pytest.approx(4.0, abs=1e-6)


def test_smoothing_monotonically_flattens_the_fit():
    uvw = RNG.random((150, 3))
    r = np.column_stack([uvw[:, 0] ** 2, uvw[:, 1] ** 2, uvw[:, 2] ** 2]) * 0.01
    basis = BernsteinBasis(2, "total")
    energies = []
    for lam in (0.0, 1e-3, 1e-1, 10.0, 1e4):
        c = fit(basis, uvw, r, smoothing=lam).coefficients
        energies.append(float(np.trace(c.T @ basis.curvature_penalty() @ c)))
    assert energies == sorted(energies, reverse=True)


def test_effective_dof_falls_as_smoothing_rises():
    uvw = RNG.random((120, 3))
    r = RNG.normal(size=(120, 3)) * 0.001
    basis = BernsteinBasis(3, "total")
    dofs = [fit(basis, uvw, r, smoothing=lam).effective_dof
            for lam in (0.0, 1e-3, 1.0, 1e5)]
    assert dofs[0] == pytest.approx(basis.n_params, abs=1e-6)
    assert dofs == sorted(dofs, reverse=True)
    assert dofs[-1] == pytest.approx(4.0, abs=0.2)


def test_huber_disbelieves_a_flipped_placement():
    """One 180-degree pose failure inside least squares drags the whole field."""
    uvw = RNG.random((60, 3))
    r = uvw * 0.004 + 0.002
    corrupted = r.copy()
    corrupted[7] += np.array([0.02, -0.02, 0.02])     # a flipped estimate

    plain = fit(BernsteinBasis(1, "total"), uvw, corrupted)
    robust = fit(BernsteinBasis(1, "total"), uvw, corrupted, robust="huber")
    clean = fit(BernsteinBasis(1, "total"), uvw, r)

    err = lambda f: np.abs(f.coefficients - clean.coefficients).max()  # noqa: E731
    assert err(robust) < 0.25 * err(plain)
    assert robust.weights[7] < 0.5
    assert robust.weights[np.arange(60) != 7].min() > 0.9


def test_rejects_an_unknown_robust_loss():
    with pytest.raises(ValueError, match="robust loss"):
        fit(BernsteinBasis(1, "total"), RNG.random((10, 3)),
            RNG.random((10, 3)), robust="tukey")


def test_smoothing_grid_contains_both_endpoints():
    grid = smoothing_grid()
    assert grid[0] == 0.0 and not np.isfinite(grid[-1])


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def test_folds_partition_the_placements_exactly_once():
    for k in (3, 5, 7, 50):
        folds = grouped_folds(20, k, seed=0)
        joined = np.concatenate(folds)
        assert sorted(joined) == list(range(20))


def test_leave_one_out_is_the_default_when_k_exceeds_n():
    assert len(grouped_folds(9, 99)) == 9


def test_spatial_holdout_takes_the_outer_shell():
    p = RNG.normal(size=(40, 3)) * 0.01
    (held,) = spatial_holdout(p, 0.25)
    d = np.linalg.norm(p - p.mean(axis=0), axis=1)
    assert d[held].min() >= np.delete(d, held).max() - 1e-12


def test_cross_validation_scores_every_placement_once():
    ds = make_dataset(SyntheticWorld(), 16, seed=2, n_grasps=2)
    cv = cross_validate(ds, BernsteinBasis(1, "total"), 0.0)
    assert cv.summary.n == len(ds)
    assert sorted(cv.scored_index) == list(range(len(ds)))


def test_an_underdetermined_fold_is_refused_rather_than_reported():
    """lstsq would happily return a minimum-norm solution and a small number."""
    ds = make_dataset(SyntheticWorld(), 14, seed=2, n_grasps=2)
    with pytest.raises(ValueError, match="underdetermined"):
        cross_validate(ds, BernsteinBasis(2, "tensor"), 0.0)


def test_paired_comparison_refuses_results_scored_on_different_placements():
    ds = make_dataset(SyntheticWorld(), 20, seed=2, n_grasps=2)
    full = cross_validate(ds, BernsteinBasis(1, "total"), 0.0)
    shell = cross_validate(ds, BernsteinBasis(1, "total"), 0.0,
                           folds=spatial_holdout(ds.p_nom(), 0.5))
    assert len(shell.scored_index) < len(full.scored_index)
    with pytest.raises(ValueError, match="same placements"):
        paired_improvement(full, shell)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_summary_reports_the_section_13_table():
    e = np.zeros((100, 3))
    e[:, 0] = 0.001                       # 1 mm, all in x
    s = summarise(e)
    assert s.rmse_3d_mm == pytest.approx(1.0)
    assert s.median_3d_mm == pytest.approx(1.0)
    assert s.max_3d_mm == pytest.approx(1.0)
    assert s.bias_mm[0] == pytest.approx(1.0)
    assert s.rmse_axis_mm[1] == pytest.approx(0.0)


def test_uncorrected_baseline_is_the_raw_residual():
    ds = make_dataset(SyntheticWorld(), 12, seed=1, n_grasps=2)
    base = uncorrected(ds)
    expected = np.linalg.norm(ds.residuals(), axis=1) * 1000.0
    assert np.allclose(sorted(base.per_placement_mm), sorted(expected))


# ---------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------
def test_variance_budget_marks_an_unestimable_model():
    rows = {r["model"]: r for r in variance_budget(40)}
    assert not rows["Bernstein tensor degree 3"]["estimable"]
    assert rows["Bernstein total degree 2"]["estimable"]


def test_variance_cost_rises_with_parameters_and_falls_with_placements():
    a = {r["model"]: r["variance_cost_mm"] for r in variance_budget(60)}
    b = {r["model"]: r["variance_cost_mm"] for r in variance_budget(240)}
    assert a["Bernstein total degree 2"] > a["Bernstein total degree 1"]
    assert b["Bernstein total degree 2"] < a["Bernstein total degree 2"]


def test_placements_needed_scales_with_the_coefficient_count():
    assert placements_needed(2, "total") > placements_needed(1, "total")
    assert placements_needed(2, "tensor") > placements_needed(2, "total")


# ---------------------------------------------------------------------------
# the procedure: can it say no, and can it say yes
# ---------------------------------------------------------------------------
def _selected_degree(label: str) -> int:
    if "constant" in label:
        return 0
    if "affine" in label:
        return 1
    if "no correction" in label:
        return -1
    return int(label.split("degree")[1].split(",")[0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_negative_control_does_not_adopt_curvature(seed):
    """An affine truth -- what the physics predicts -- must not buy a polynomial."""
    ds = make_dataset(SyntheticWorld(), 60, seed=100 + seed, n_grasps=3).usable()
    floor = ds.noise_floor()["residual_sd_3d_mm"]
    sel = select_model(ds, max_degree=3, smoothings=FAST, seed=seed,
                       robust="huber", margin_mm=floor / np.sqrt(len(ds)))
    assert _selected_degree(sel.best.label) <= 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_positive_control_does_adopt_curvature(seed):
    """A genuinely curved perception bias must be found."""
    world = SyntheticWorld(fp_extra_bias=saddle_bias())
    ds = make_dataset(world, 60, seed=100 + seed, n_grasps=3).usable()
    floor = ds.noise_floor()["residual_sd_3d_mm"]
    sel = select_model(ds, max_degree=3, smoothings=FAST, seed=seed,
                       robust="huber", margin_mm=floor / np.sqrt(len(ds)))
    assert _selected_degree(sel.best.label) >= 2


def test_any_correction_beats_no_correction_by_a_wide_margin():
    ds = make_dataset(SyntheticWorld(), 40, seed=9, n_grasps=3).usable()
    sel = select_model(ds, max_degree=2, smoothings=FAST, seed=0, robust="huber")
    assert sel.steps[0]["adopted"]
    assert sel.steps[0]["rmse_delta_mm"] > 2.0
    assert sel.steps[0]["ci95_mm"][0] > 0.0


def test_selection_reaches_the_noise_floor_on_an_affine_world():
    ds = make_dataset(SyntheticWorld(), 80, seed=11, n_grasps=5).usable()
    floor = ds.noise_floor()["residual_sd_3d_mm"]
    sel = select_model(ds, max_degree=2, smoothings=FAST, seed=0, robust="huber")
    assert sel.best.rmse_mm < 1.6 * floor


def test_selection_skips_a_model_with_more_coefficients_than_placements():
    ds = make_dataset(SyntheticWorld(), 20, seed=4, n_grasps=2).usable()
    sel = select_model(ds, kind="tensor", max_degree=3, smoothings=FAST, seed=0)
    assert any("skipped" in n for n in sel.notes)
    assert all(_selected_degree(r.label) < 3 for r in sel.results)


# ---------------------------------------------------------------------------
def test_report_runs_and_names_the_things_it_must_name():
    ds = make_dataset(SyntheticWorld(fp_flip_rate=0.05), 30, seed=6,
                      n_grasps=3).usable()
    text = report(ds, max_degree=2)
    for phrase in [
        "SECTION 6", "noise floor", "no correction", "constant offset", "affine",
        "SPATIAL HOLD-OUT", "real grasps", "never used for calibration",
    ]:
        assert phrase.lower() in text.lower(), phrase


def test_fit_dataset_records_the_conventions_it_was_fitted_under():
    ds = make_dataset(SyntheticWorld(), 30, seed=8, n_grasps=2).usable()
    m = fit_dataset(ds, BernsteinBasis(1, "total"))
    assert m.metadata["convention_digest"] == ds.convention_digest()
    assert m.metadata["validated"] is False      # nothing has validated it yet
    assert m.metadata["taught_orientation_quat_xyzw"] is not None


def test_fit_dataset_honours_an_explicit_workspace():
    ds = make_dataset(SyntheticWorld(), 20, seed=8, n_grasps=2).usable()
    ws = Workspace(ds.p_nom().min(axis=0) - 0.05, ds.p_nom().max(axis=0) + 0.05)
    m = fit_dataset(ds, BernsteinBasis(1, "total"), workspace=ws)
    assert np.allclose(m.workspace.low_m, ws.low_m)


def test_fit_dataset_names_the_placements_it_downweighted():
    world = SyntheticWorld(fp_flip_rate=0.0)
    ds = make_dataset(world, 40, seed=12, n_grasps=2).usable()
    ds.placements[5].grasp_poses[0][:3, 3] += 0.01
    m = fit_dataset(ds, BernsteinBasis(1, "total"), robust="huber")
    assert any(pid == "p005" for pid, _ in m.metadata["downweighted"])
