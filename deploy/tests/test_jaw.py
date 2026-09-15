"""The jaw mapping, and the fact that its output is never a grasp confirmation."""

import numpy as np
import pytest

from surgicai_rl_deploy.jaw import (
    JawBaseline,
    JawCalibration,
    JawCalibrationError,
    JawEvidenceWindow,
    evaluate_jaw_evidence,
)


# --- calibration ----------------------------------------------------------
def test_grip_must_be_a_squeeze():
    """A non-negative grip command applies no tendon tension and holds nothing."""
    with pytest.raises(JawCalibrationError, match="no tendon tension"):
        JawCalibration(grip_rad=float(np.deg2rad(5.0)))


def test_grip_floor_is_enforced():
    with pytest.raises(JawCalibrationError, match="tool damage"):
        JawCalibration(grip_rad=float(np.deg2rad(-40.0)))


def test_open_must_exceed_closed():
    with pytest.raises(JawCalibrationError):
        JawCalibration(open_rad=0.0, closed_rad=0.1)


def test_open_ceiling():
    with pytest.raises(JawCalibrationError, match="ceiling"):
        JawCalibration(open_rad=float(np.deg2rad(120.0)))


def test_approach_open_must_sit_between():
    with pytest.raises(JawCalibrationError):
        JawCalibration(approach_open_rad=float(np.deg2rad(90.0)))


def test_normalisation_matches_the_previous_contract():
    """The old node used jaw_norm = jaw_rad / jaw_open_rad; with closed at 0
    the new mapping must reproduce it exactly, so the approach path is
    unchanged."""
    cal = JawCalibration(open_rad=1.0, closed_rad=0.0)
    for angle in (0.0, 0.25, 0.5, 1.0):
        assert cal.normalise(angle) == pytest.approx(angle / 1.0)


def test_normalise_clamps_but_unclamped_reports_the_squeeze():
    cal = JawCalibration()
    assert cal.normalise(cal.grip_rad) == 0.0
    assert cal.normalise_unclamped(cal.grip_rad) < 0.0
    assert cal.normalise(cal.open_rad * 2) == 1.0


def test_to_rad_roundtrip():
    cal = JawCalibration()
    for norm in (0.0, 0.3, 1.0):
        assert cal.normalise(cal.to_rad(norm)) == pytest.approx(norm)


# --- evidence -------------------------------------------------------------
def test_no_jaw_feedback_says_nothing():
    ev = evaluate_jaw_evidence(-0.26, None)
    assert ev.jaw_blocked is None
    assert ev.verified is False
    assert "nothing claimed" in ev.note


def test_no_baseline_says_nothing():
    ev = evaluate_jaw_evidence(-0.26, 0.03)
    assert ev.jaw_blocked is None
    assert ev.residual_rad == pytest.approx(0.29)
    assert "no empty-jaw baseline" in ev.note


def test_blocked_when_the_jaw_stops_early(baseline):
    """Commanded -15 deg, stopped at +2 deg: something is between the fingers."""
    ev = evaluate_jaw_evidence(
        baseline.empty_close_rad, float(np.deg2rad(2.0)), 0.9, baseline
    )
    assert ev.jaw_blocked is True
    # the empty jaw settles at the commanded -15 deg, so stopping at +2 deg is
    # 17 deg of excess
    assert ev.residual_excess_rad == pytest.approx(np.deg2rad(17.0), abs=1e-6)
    assert ev.verified is False
    assert "not verified" in ev.note


def test_not_blocked_on_an_empty_close(baseline):
    ev = evaluate_jaw_evidence(
        baseline.empty_close_rad, baseline.empty_close_rad, 0.02, baseline
    )
    assert ev.jaw_blocked is False


def test_sub_margin_residual_is_not_blocked(baseline):
    """A 0.5 deg residual is inside the noise floor; it must not read as a grasp."""
    ev = evaluate_jaw_evidence(
        baseline.empty_close_rad,
        baseline.empty_close_rad + float(np.deg2rad(0.5)),
        0.02,
        baseline,
    )
    assert ev.jaw_blocked is False


def test_effort_alone_can_flag_a_block(baseline):
    """Angle looks empty but the motor is straining: still worth flagging."""
    ev = evaluate_jaw_evidence(
        baseline.empty_close_rad, baseline.empty_close_rad, 5.0, baseline
    )
    assert ev.jaw_blocked is True
    assert ev.effort_excess == pytest.approx(4.98)


def test_missing_effort_field_falls_back_to_angle(baseline):
    ev = evaluate_jaw_evidence(
        baseline.empty_close_rad, float(np.deg2rad(3.0)), None, baseline
    )
    assert ev.jaw_blocked is True
    assert ev.effort_excess is None
    assert "no effort reference" in ev.note


def test_evidence_dict_always_carries_the_disclaimer(baseline):
    ev = evaluate_jaw_evidence(
        baseline.empty_close_rad, float(np.deg2rad(2.0)), 0.9, baseline
    )
    assert ev.as_dict()["grasp_verified"] is False


# --- the streak window ----------------------------------------------------
def test_window_requires_a_streak(baseline):
    window = JawEvidenceWindow(required_streak=3)
    blocked = evaluate_jaw_evidence(
        baseline.empty_close_rad, float(np.deg2rad(2.0)), 0.9, baseline
    )
    empty = evaluate_jaw_evidence(
        baseline.empty_close_rad, baseline.empty_close_rad, 0.02, baseline
    )
    window.update(blocked).update(blocked)
    assert not window.blocked_streak_met
    window.update(blocked)
    assert window.blocked_streak_met
    window.update(empty)
    assert not window.blocked_streak_met


def test_unknown_readings_break_the_streak():
    """Unknown is not evidence."""
    window = JawEvidenceWindow(required_streak=2)
    unknown = evaluate_jaw_evidence(-0.26, None)
    window.update(unknown).update(unknown)
    assert not window.blocked_streak_met


def test_window_summary_is_honest(baseline):
    window = JawEvidenceWindow(required_streak=2)
    window.update(
        evaluate_jaw_evidence(
            baseline.empty_close_rad, float(np.deg2rad(2.0)), 0.9, baseline
        )
    )
    assert window.summary()["grasp_verified"] is False


def test_baseline_roundtrip(baseline):
    assert JawBaseline.from_dict(baseline.as_dict()) == baseline
