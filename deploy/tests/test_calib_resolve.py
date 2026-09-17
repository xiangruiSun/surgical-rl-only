"""The runtime path, and every refusal that stands between it and the arm.

An end-to-end check that the correction actually helps at a needle it never saw
comes first, because if it does not, the refusals are guarding nothing.  The
rest of the file is the refusals: a changed convention, a needle outside the
calibrated box, a wrist at the wrong angle, a model nobody validated.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from surgicai_rl_deploy.calib.bernstein import BernsteinBasis
from surgicai_rl_deploy.calib.fit import fit_dataset
from surgicai_rl_deploy.calib.perception import (
    DEFAULT_T_EC,
    GraspPointSpec,
    HandEye,
)
from surgicai_rl_deploy.calib.resolve import GraspResolver
from surgicai_rl_deploy.calib.simulate import SyntheticWorld, make_dataset


def build(tmp_path, n=50, seed=0, degree=1):
    """A world, a fitted model on disk, and a resolver that loads it."""
    world = SyntheticWorld()
    ds = make_dataset(world, n, seed=seed, n_grasps=3).usable()
    model = fit_dataset(
        ds, BernsteinBasis(degree, "total"),
        metadata={"validated": True, "cv_rmse_mm": 0.5, "cv_baseline_mm": 3.9},
    )
    path = model.save(tmp_path / "cal.json")
    return world, ds, GraspResolver.from_files(path), path


def observe(world, seed=99, frames=5):
    rng = np.random.default_rng(seed)
    T_true = world.place_needle(rng)
    return T_true, [world.estimate(T_true, rng) for _ in range(frames)], rng


# ---------------------------------------------------------------------------
# does it actually help
# ---------------------------------------------------------------------------
def test_the_correction_helps_at_a_needle_the_model_never_saw(tmp_path):
    world, ds, resolver, _ = build(tmp_path)
    corrected, nominal = [], []
    for s in range(40):
        T_true, frames, rng = observe(world, seed=500 + s)
        truth = world.true_grasp_pose(T_true, ds.grasp_point, rng)[:3, 3]
        tgt = resolver.resolve(frames, orientation=resolver.taught_orientation)
        if not tgt.report.ok:
            continue
        corrected.append(np.linalg.norm(tgt.p_target_m - truth))
        nominal.append(np.linalg.norm(tgt.p_nominal_m - truth))
    corrected, nominal = np.array(corrected) * 1000, np.array(nominal) * 1000
    assert len(corrected) > 20
    assert corrected.mean() < 0.25 * nominal.mean()
    assert corrected.mean() < 1.0


def test_target_is_nominal_plus_correction(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    t = resolver.resolve(frames, orientation=resolver.taught_orientation)
    assert np.allclose(t.p_target_m, t.p_nominal_m + t.correction_m)
    assert np.allclose(t.pose.p, t.p_target_m)


def test_averaging_frames_is_reported_and_reduces_the_scatter(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    spread = {}
    for n_frames in (1, 25):
        targets = []
        for s in range(30):
            T_true = world.place_needle(np.random.default_rng(7))
            rng = np.random.default_rng(1000 + s)
            frames = [world.estimate(T_true, rng) for _ in range(n_frames)]
            t = resolver.resolve(frames, orientation=resolver.taught_orientation)
            assert t.n_frames == n_frames
            targets.append(t.p_target_m)
        spread[n_frames] = np.std(np.array(targets), axis=0).mean()
    assert spread[25] < 0.5 * spread[1]


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------
def test_refuses_a_needle_outside_the_calibrated_box(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    far = [f.copy() for f in frames]
    for f in far:
        f[0, 3] += 0.08
    t = resolver.resolve(far, orientation=resolver.taught_orientation)
    assert not t.report.ok
    region = [c for c in t.report.checks if c.name == "calibration.region"][0]
    assert region.status == "fail"
    assert "outside calibrated" in region.message


def test_a_refused_needle_still_reports_a_bounded_correction(tmp_path):
    """The command is refused, not silently replaced with an extrapolation."""
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    far = [f.copy() for f in frames]
    for f in far:
        f[0, 3] += 0.08
    t = resolver.resolve(far, orientation=resolver.taught_orientation)
    assert np.linalg.norm(t.correction_mm) <= resolver.model.max_correction_mm() + 1e-9


def test_refuses_a_changed_hand_eye_transform(tmp_path):
    _, _, resolver, path = build(tmp_path)
    moved = DEFAULT_T_EC.copy()
    moved[0, 3] += 0.003
    resolver.hand_eye = HandEye(moved, source="recalibrated")
    rep = resolver.precheck()
    check = [c for c in rep.checks if c.name == "calibration.conventions"][0]
    assert check.status == "fail"
    assert "recollect" in check.message


def test_refuses_a_changed_needle_point(tmp_path):
    """Section 7's mistake, caught before the arm moves rather than after."""
    _, _, resolver, _ = build(tmp_path)
    resolver.grasp_point = GraspPointSpec(mode="mesh_origin")
    assert not resolver.precheck().ok


def test_refuses_a_model_nobody_validated(tmp_path):
    world = SyntheticWorld()
    ds = make_dataset(world, 30, seed=1, n_grasps=2).usable()
    path = fit_dataset(ds, BernsteinBasis(1, "total")).save(tmp_path / "raw.json")
    assert not GraspResolver.from_files(path).precheck().ok
    # ...but it can be run deliberately, with the warning still recorded
    lenient = GraspResolver.from_files(path, require_validated=False)
    rep = lenient.precheck()
    assert rep.ok and rep.warnings


def test_refuses_a_model_that_commands_an_implausible_correction(tmp_path):
    _, _, resolver, _ = build(tmp_path)
    resolver.max_correction_mm = 0.1
    check = [c for c in resolver.precheck().checks if c.name == "calibration.bound"][0]
    assert check.status == "fail"


def test_refuses_frames_that_disagree(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    flipped = frames[0].copy()
    flipped[:3, :3] = flipped[:3, :3] @ Rotation.from_rotvec(
        [0, 0, np.deg2rad(175)]
    ).as_matrix()
    t = resolver.resolve(frames + [flipped], orientation=resolver.taught_orientation)
    check = [c for c in t.report.checks if c.name == "calibration.frames"][0]
    assert check.status == "fail"


# ---------------------------------------------------------------------------
# the orientation freeze
# ---------------------------------------------------------------------------
def test_warns_when_the_wrist_is_not_where_the_calibration_was_taught(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    turned = resolver.taught_orientation * Rotation.from_rotvec([0, 0, np.deg2rad(40)])
    t = resolver.resolve(frames, orientation=turned)
    check = [c for c in t.report.checks if c.name == "calibration.orientation"][0]
    assert check.status == "warn"
    assert check.detail["delta_deg"] == pytest.approx(40.0, abs=0.5)
    assert "jaw offset" in check.message


def test_does_not_warn_at_the_taught_orientation(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    t = resolver.resolve(frames, orientation=resolver.taught_orientation)
    check = [c for c in t.report.checks if c.name == "calibration.orientation"][0]
    assert check.status == "pass"


def test_strict_promotes_the_orientation_warning_to_a_failure(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    turned = resolver.taught_orientation * Rotation.from_rotvec([0, 0, 1.0])
    assert not resolver.resolve(frames, orientation=turned, strict=True).report.ok


# ---------------------------------------------------------------------------
# what it carries
# ---------------------------------------------------------------------------
def test_target_records_every_intermediate_for_the_trace(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    d = resolver.resolve(frames, orientation=resolver.taught_orientation).as_dict()
    for key in ("p_camera_cm", "p_nominal_cm", "correction_mm", "p_target_cm",
                "target_quat_xyzw", "n_frames", "precheck"):
        assert key in d
    assert "->" in resolver.resolve(
        frames, orientation=resolver.taught_orientation
    ).describe()


def test_taught_orientation_comes_back_off_the_file(tmp_path):
    world, ds, resolver, _ = build(tmp_path)
    expected = Rotation.from_matrix(
        np.array([p.grasp_rotation().as_matrix() for p in ds.placements])
    ).mean()
    delta = np.degrees((expected.inv() * resolver.taught_orientation).magnitude())
    assert delta < 1e-6


def test_a_single_pose_is_accepted_without_wrapping_it(tmp_path):
    world, _, resolver, _ = build(tmp_path)
    _, frames, _ = observe(world)
    t = resolver.resolve(frames[0], orientation=resolver.taught_orientation)
    assert t.n_frames == 1
