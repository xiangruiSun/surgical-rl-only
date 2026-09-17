"""The dataset: grouping by placement, and the audits that guard a session.

The property worth defending hardest is that the residual is never *stored*.  It
is recomputed from the raw measurements under the dataset's own conventions
every time, so a change of grasp point or hand-eye transform cannot leave a
stale residual behind -- which is how a section 7 mistake would otherwise
survive a refit and look like a modelling problem.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from surgicai_rl_deploy.calib.dataset import CalibrationDataset, Placement
from surgicai_rl_deploy.calib.perception import (
    GraspPointSpec,
    HandEye,
    nominal_position,
)

RNG = np.random.default_rng(3)


def pose(t, rotvec=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


def placement(pid="p000", n_frames=5, n_grasps=2, noise=3e-4, seed=0,
              needle_t=(0.0, 0.0, 0.08), grasp_t=(0.01, 0.01, 0.09)):
    rng = np.random.default_rng(seed)
    return Placement(
        pid,
        [pose(np.array(needle_t) + rng.normal(0, noise, 3)) for _ in range(n_frames)],
        [pose(np.array(grasp_t) + rng.normal(0, noise, 3), (2.2, 0.4, -0.3))
         for _ in range(n_grasps)],
    )


def dataset(n=12, **kw):
    ds = CalibrationDataset()
    for i in range(n):
        ds.add(placement(
            f"p{i:03d}",
            seed=i,
            needle_t=(RNG.uniform(-0.02, 0.02), RNG.uniform(-0.02, 0.02),
                      RNG.uniform(0.07, 0.09)),
            **kw,
        ))
    return ds


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def test_a_placement_needs_both_kinds_of_measurement():
    with pytest.raises(ValueError, match="no pose estimates"):
        Placement("p", [], [pose([0, 0, 0])])
    with pytest.raises(ValueError, match="no grasp poses"):
        Placement("p", [pose([0, 0, 0])], [])


def test_duplicate_placement_ids_are_refused():
    ds = CalibrationDataset()
    ds.add(placement("p000"))
    with pytest.raises(ValueError, match="already in this dataset"):
        ds.add(placement("p000"))


def test_recorded_at_is_filled_in():
    assert placement().recorded_at


# ---------------------------------------------------------------------------
# the residual is derived, never stored
# ---------------------------------------------------------------------------
def test_residual_follows_the_grasp_point_convention():
    """Switching the needle point must move the residual by the lever arm."""
    ds = dataset(6)
    r_arc = ds.residuals().copy()
    ds.grasp_point = GraspPointSpec(mode="mesh_origin")
    r_origin = ds.residuals()
    moved = np.linalg.norm(r_arc - r_origin, axis=1)
    assert np.allclose(moved, 0.01018, atol=1e-6)


def test_residual_follows_the_hand_eye_transform():
    ds = dataset(6)
    before = ds.residuals().copy()
    shifted = HandEye().T.copy()
    shifted[0, 3] += 0.002
    ds.hand_eye = HandEye(shifted)
    after = ds.residuals()
    assert np.allclose(after[:, 0] - before[:, 0], -0.002, atol=1e-9)


def test_p_nom_matches_the_perception_pipeline():
    ds = dataset(4)
    for p, want in zip(ds.placements, ds.p_nom()):
        got = nominal_position(p.mean_pose_estimate(), ds.hand_eye, ds.grasp_point)
        assert np.allclose(got, want)


def test_residual_is_grasp_minus_nominal():
    ds = dataset(5)
    assert np.allclose(ds.residuals(), ds.p_grasp() - ds.p_nom())


# ---------------------------------------------------------------------------
# section 6
# ---------------------------------------------------------------------------
def test_noise_floor_is_complete_only_with_both_kinds_of_repeat():
    assert dataset(6, n_frames=5, n_grasps=3).noise_floor()["complete"]
    assert not dataset(6, n_frames=5, n_grasps=1).noise_floor()["complete"]
    assert not dataset(6, n_frames=1, n_grasps=3).noise_floor()["complete"]


def test_noise_floor_tracks_the_injected_noise():
    lo = dataset(10, noise=1e-4).noise_floor()["residual_sd_3d_mm"]
    hi = dataset(10, noise=6e-4).noise_floor()["residual_sd_3d_mm"]
    assert hi > 3.0 * lo


def test_averaging_more_frames_lowers_the_floor():
    few = dataset(10, n_frames=2, n_grasps=3).noise_floor()["residual_sd_3d_mm"]
    many = dataset(10, n_frames=20, n_grasps=3).noise_floor()["residual_sd_3d_mm"]
    assert many < few


# ---------------------------------------------------------------------------
# section 8
# ---------------------------------------------------------------------------
def test_a_frozen_wrist_reports_no_spread_and_no_leakage():
    ds = dataset(8)
    assert ds.orientation_spread_deg() < 1e-6
    assert ds.jaw_offset_leakage_mm() < 1e-6


def test_a_wandering_wrist_is_reported_as_millimetres_of_leakage():
    ds = CalibrationDataset()
    for i in range(8):
        p = placement(f"p{i:03d}", seed=i)
        turn = Rotation.from_rotvec([0, 0, np.deg2rad(4.0 * i)]).as_matrix()
        p.grasp_poses = [
            np.block([[turn @ g[:3, :3], g[:3, 3:4]], [np.zeros((1, 3)), np.ones((1, 1))]])
            for g in p.grasp_poses
        ]
        ds.add(p)
    assert ds.orientation_spread_deg() > 10.0
    assert ds.jaw_offset_leakage_mm(5.0) > 1.0
    assert any("varies by" in s for s in ds.problems())


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------
def test_an_unverified_grasp_is_dropped_with_a_reason():
    ds = dataset(6)
    ds.placements[2].verified = False
    dropped = dict(ds.dropped())
    assert "p002" in dropped
    assert any("never confirmed" in r for r in dropped["p002"])
    assert len(ds.usable()) == 5


def test_a_flipped_pose_estimate_is_dropped():
    ds = dataset(6)
    bad = ds.placements[1].pose_estimates[0].copy()
    bad[:3, :3] = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
    ds.placements[1].pose_estimates[0] = bad
    assert "p001" in dict(ds.dropped())
    assert len(ds.usable()) == 5


def test_grasps_that_disagree_are_dropped():
    ds = dataset(6)
    far = ds.placements[3].grasp_poses[0].copy()
    far[:3, 3] += 0.02
    ds.placements[3].grasp_poses[0] = far
    assert any("disagree" in r for r in dict(ds.dropped())["p003"])


def test_a_small_dataset_says_so():
    assert any("not enough" in s for s in dataset(4).problems())


def test_a_thin_axis_is_named():
    ds = CalibrationDataset()
    for i in range(12):
        ds.add(placement(f"p{i:03d}", seed=i,
                         needle_t=(0.001 * i, 0.0, 0.08)))   # y and z frozen
    problems = " ".join(ds.problems())
    assert "barely move" in problems and "y" in problems


def test_usable_keeps_the_conventions():
    ds = dataset(6)
    ds.grasp_point = GraspPointSpec(angle_deg=105.0)
    assert ds.usable().grasp_point == ds.grasp_point
    assert ds.usable().convention_digest() == ds.convention_digest()


def test_subset_selects_by_id():
    ds = dataset(6)
    sub = ds.subset(["p001", "p004"])
    assert sub.ids == ["p001", "p004"]


# ---------------------------------------------------------------------------
def test_coverage_reports_span_and_uniformity():
    cov = dataset(30).coverage()
    assert cov["n"] == 30
    assert all(s > 0 for s in cov["span_cm"])
    assert all(0.4 < u < 2.0 for u in cov["uniformity"])


def test_round_trips_through_a_file(tmp_path):
    ds = dataset(8)
    ds.grasp_point = GraspPointSpec(angle_deg=45.0)
    ds.provenance = {"arm": "PSM1"}
    back = CalibrationDataset.load(ds.save(tmp_path / "d.json"))
    assert back.ids == ds.ids
    assert back.convention_digest() == ds.convention_digest()
    assert np.allclose(back.residuals(), ds.residuals())
    assert back.provenance == ds.provenance


def test_refuses_a_file_from_another_format_version(tmp_path):
    ds = dataset(3)
    d = ds.as_dict()
    d["format_version"] = 99
    import json
    p = tmp_path / "d.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="format version"):
        CalibrationDataset.load(p)


def test_from_measurements_builds_grasps_from_position_and_quaternion():
    p = Placement.from_measurements(
        "p",
        [pose([0, 0, 0.08])],
        [[0.01, 0.0, 0.09]],
        [[0.0, 0.0, 0.0, 1.0]],
    )
    assert np.allclose(p.p_grasp(), [0.01, 0.0, 0.09])
