"""The two conventions that have to be right before anything else matters.

Section 7 (which point on the needle) and section 15 (which hand-eye transform).
Both are worth more millimetres than the polynomial degree is, so both are
pinned to the numbers in SurgicAI's own source rather than to anything this
package chose.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from surgicai_rl_deploy.calib.perception import (
    DEFAULT_GRASP_ANGLE_DEG,
    DEFAULT_T_EC,
    NEEDLE_ANGLES_DEG,
    NEEDLE_RADIUS_M,
    GraspPointSpec,
    HandEye,
    average_poses,
    convention_digest,
    flip_reasons,
    needle_frame_N,
    needle_point_N,
    nominal_position,
)

RNG = np.random.default_rng(7)


def _pose(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# needle geometry, against SurgicAI's own numbers
# ---------------------------------------------------------------------------
def test_radius_matches_surgicai():
    """``RL/utils/needle_kinematics_new.py``: ``Radius = 0.1018``, /10 to metres."""
    assert NEEDLE_RADIUS_M == pytest.approx(0.01018)


def test_named_points_match_needle_kinematics():
    """base, bm, mid, tip as ``T_bINn``, ``T_bmINn``, ``T_mINn``, ``T_tINn``."""
    R = NEEDLE_RADIUS_M
    expected = {
        "base": [-R, 0.0, 0.0],
        "bm": [-R * np.cos(np.pi / 6), R * np.sin(np.pi / 6), 0.0],
        "mid": [-R * np.cos(np.pi / 3), R * np.sin(np.pi / 3), 0.0],
        "tip": [-R * np.cos(2 * np.pi / 3), R * np.sin(2 * np.pi / 3), 0.0],
    }
    for name, want in expected.items():
        got = needle_point_N(NEEDLE_ANGLES_DEG[name])
        assert np.allclose(got, want, atol=1e-12)


def test_every_named_point_is_on_the_arc():
    for ang in NEEDLE_ANGLES_DEG.values():
        assert np.linalg.norm(needle_point_N(ang)) == pytest.approx(NEEDLE_RADIUS_M)


def test_the_mesh_origin_is_not_on_the_needle():
    """The fact section 7 exists for: the FoundationPose translation is the
    centre of the arc, ten millimetres from any part of the wire."""
    assert np.linalg.norm(needle_point_N(0.0)) > 0.01


def test_frame_carries_surgicai_s_yaw_convention():
    """``get_pose_angle`` rotates by ``Rz(-theta)`` at each arc angle."""
    for ang in (0.0, 30.0, 105.0):
        T = needle_frame_N(ang)
        yaw = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")[2]
        assert yaw == pytest.approx(-np.deg2rad(ang))


def test_default_grasp_angle_is_surgicai_s_own_target():
    """``needle_goal_evaluator`` defaults to ``get_bm_pose()``, theta = 30."""
    assert DEFAULT_GRASP_ANGLE_DEG == NEEDLE_ANGLES_DEG["bm"] == 30.0


# ---------------------------------------------------------------------------
# what the convention costs
# ---------------------------------------------------------------------------
def test_mesh_origin_carries_no_orientation_sensitivity():
    s = GraspPointSpec(mode="mesh_origin").orientation_sensitivity_mm(30.0)
    assert s["rms_mm"] == 0.0


def test_arc_point_sensitivity_grows_with_the_yaw_envelope():
    spec = GraspPointSpec()
    vals = [spec.orientation_sensitivity_mm(y)["rms_mm"] for y in (5, 10, 20, 30)]
    assert vals == sorted(vals)
    # at this repository's own +-30 deg needle cap the term is millimetres,
    # which is the size of the whole residual being modelled
    assert 2.0 < vals[-1] < 3.5


def test_grasp_point_follows_the_needle_orientation():
    """The point is on the arc, so it moves when the needle turns even if the
    mesh origin does not.  This is the term a position-only model cannot see."""
    spec = GraspPointSpec()
    T = _pose([0, 0, 0], [0.0, 0.0, 0.08])
    T_turned = _pose([0, 0, np.deg2rad(30)], [0.0, 0.0, 0.08])
    moved = np.linalg.norm(spec.in_camera(T) - spec.in_camera(T_turned))
    assert moved > 0.004
    origin = GraspPointSpec(mode="mesh_origin")
    assert np.allclose(origin.in_camera(T), origin.in_camera(T_turned))


def test_grasp_point_spec_validates():
    with pytest.raises(ValueError):
        GraspPointSpec(mode="middle")
    with pytest.raises(ValueError):
        GraspPointSpec(radius_m=0.0)
    with pytest.raises(ValueError):
        GraspPointSpec(angle_deg=np.nan)


def test_grasp_point_round_trips():
    spec = GraspPointSpec(mode="arc", angle_deg=105.0)
    assert GraspPointSpec.from_dict(spec.as_dict()) == spec


# ---------------------------------------------------------------------------
# hand-eye
# ---------------------------------------------------------------------------
def test_default_transform_is_orthonormal():
    he = HandEye()
    assert np.abs(he.R.T @ he.R - np.eye(3)).max() < 1e-9
    assert np.linalg.det(he.R) == pytest.approx(1.0, abs=1e-9)


def test_refuses_a_transform_that_is_not_a_rotation():
    bad = DEFAULT_T_EC.copy()
    bad[:3, :3] *= 1.01
    with pytest.raises(ValueError, match="orthonormal"):
        HandEye(bad)


def test_refuses_a_reflection():
    bad = DEFAULT_T_EC.copy()
    bad[:3, :3] = -np.eye(3)          # det = -1, and orthonormal
    with pytest.raises(ValueError, match="determinant"):
        HandEye(bad)


def test_refuses_a_broken_bottom_row():
    bad = DEFAULT_T_EC.copy()
    bad[3, 3] = 2.0
    with pytest.raises(ValueError, match="bottom row"):
        HandEye(bad)


def test_inverse_round_trips():
    he = HandEye()
    p = RNG.normal(size=(20, 3)) * 0.05
    # 1e-10 m is a tenth of an angstrom; the transform is stored to ten digits
    assert np.abs(he.inverse().point_to_ecm(he.point_to_ecm(p)) - p).max() < 1e-10


def test_rotation_error_to_mm_is_the_small_angle_lever_arm():
    assert HandEye().rotation_error_to_mm(1.0, 0.08) == pytest.approx(1.396, abs=1e-3)


def test_digest_is_stable_and_discriminating():
    a = HandEye()
    b = HandEye(DEFAULT_T_EC.copy())
    assert a.digest() == b.digest()
    moved = DEFAULT_T_EC.copy()
    moved[0, 3] += 0.001
    assert HandEye(moved).digest() != a.digest()


def test_convention_digest_reacts_to_both_halves():
    base = convention_digest(HandEye(), GraspPointSpec())
    assert base == convention_digest(HandEye(), GraspPointSpec())
    assert base != convention_digest(HandEye(), GraspPointSpec(angle_deg=105.0))
    moved = DEFAULT_T_EC.copy()
    moved[1, 3] += 0.0005
    assert base != convention_digest(HandEye(moved), GraspPointSpec())


# ---------------------------------------------------------------------------
# the nominal pipeline
# ---------------------------------------------------------------------------
def test_nominal_position_is_the_hand_eye_image_of_the_arc_point():
    he, spec = HandEye(), GraspPointSpec()
    T = _pose([0.1, -0.2, 0.3], [0.01, -0.005, 0.085])
    expected = he.R @ spec.in_camera(T) + he.t
    assert np.allclose(nominal_position(T, he, spec), expected)


# ---------------------------------------------------------------------------
# averaging and flip detection
# ---------------------------------------------------------------------------
def test_average_of_identical_poses_is_that_pose():
    T = _pose([0.2, 0.1, -0.4], [0.01, 0.02, 0.08])
    assert np.abs(average_poses([T] * 5) - T).max() < 1e-12


def test_averaging_reduces_the_noise_as_root_m():
    T = _pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.08])
    single, averaged = [], []
    for _ in range(300):
        frames = []
        for _ in range(9):
            f = T.copy()
            f[:3, 3] += RNG.normal(0, 4e-4, 3)
            frames.append(f)
        single.append(np.linalg.norm(frames[0][:3, 3] - T[:3, 3]))
        averaged.append(np.linalg.norm(average_poses(frames)[:3, 3] - T[:3, 3]))
    assert np.mean(averaged) < np.mean(single) / 2.0


def test_flip_is_detected_and_noise_is_not():
    T = _pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.08])
    noisy = []
    for _ in range(6):
        f = T.copy()
        f[:3, 3] += RNG.normal(0, 3e-4, 3)
        f[:3, :3] = Rotation.from_rotvec(RNG.normal(0, 0.01, 3)).as_matrix() @ f[:3, :3]
        noisy.append(f)
    assert flip_reasons(noisy) == []

    flipped = list(noisy)
    bad = T.copy()
    bad[:3, :3] = bad[:3, :3] @ Rotation.from_rotvec([0, 0, np.deg2rad(175)]).as_matrix()
    flipped.append(bad)
    assert any("flip" in r for r in flip_reasons(flipped))


def test_averaging_refuses_to_average_a_flip():
    T = _pose([0.0, 0.0, 0.0], [0.0, 0.0, 0.08])
    bad = T.copy()
    bad[:3, :3] = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
    with pytest.raises(ValueError, match="not repeated measurements"):
        average_poses([T, bad])


def test_a_single_frame_is_never_a_flip():
    assert flip_reasons([np.eye(4)]) == []
