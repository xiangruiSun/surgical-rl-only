"""Loading the empty-jaw baseline.

A baseline that cannot be trusted is worse than no baseline: every jaw reading
in the run is compared against it, so a file recorded at the wrong grip angle,
or in a dry run where the jaw never moved, quietly turns the evidence channel
into noise that still looks authoritative. These paths must refuse, not warn.
"""

import json

import numpy as np
import pytest


def write(tmp_path, payload, name="jaw_baseline.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def good_payload(grip_deg=-15.0):
    return {
        "empty_close_rad": float(np.deg2rad(grip_deg)),
        "empty_close_rad_noise": float(np.deg2rad(0.1)),
        "empty_close_effort": 0.02,
        "empty_close_effort_noise": 0.005,
        "source": "PSM1 empty-jaw close, test",
        "grip_command_deg": grip_deg,
        "executed": True,
    }


def test_no_spec_means_no_baseline_and_no_complaint(node_module):
    baseline, problem = node_module.resolve_jaw_baseline(None, grip_deg=-15.0)
    assert baseline is None and problem is None


def test_a_good_baseline_loads(node_module, tmp_path):
    baseline, problem = node_module.resolve_jaw_baseline(
        write(tmp_path, good_payload()), grip_deg=-15.0
    )
    assert problem is None
    assert baseline.empty_close_rad == pytest.approx(np.deg2rad(-15.0))
    assert baseline.empty_close_effort == pytest.approx(0.02)


def test_a_missing_file_explains_how_to_make_one(node_module, tmp_path):
    baseline, problem = node_module.resolve_jaw_baseline(
        str(tmp_path / "nope.json"), grip_deg=-15.0, arm="/PSM1"
    )
    assert baseline is None
    assert "not found" in problem
    assert "calibrate_jaw.py" in problem
    assert "EMPTY gripper" in problem


def test_malformed_json_is_reported_not_raised(node_module, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    baseline, problem = node_module.resolve_jaw_baseline(str(path), grip_deg=-15.0)
    assert baseline is None and "not valid JSON" in problem


def test_a_json_array_is_rejected(node_module, tmp_path):
    path = tmp_path / "arr.json"
    path.write_text("[1, 2, 3]")
    baseline, problem = node_module.resolve_jaw_baseline(str(path), grip_deg=-15.0)
    assert baseline is None and "JSON object" in problem


def test_a_baseline_without_the_measurement_is_rejected(node_module, tmp_path):
    payload = good_payload()
    del payload["empty_close_rad"]
    baseline, problem = node_module.resolve_jaw_baseline(
        write(tmp_path, payload), grip_deg=-15.0
    )
    assert baseline is None and "unusable" in problem


def test_a_dry_run_baseline_is_refused(node_module, tmp_path):
    """The jaw never moved, so the numbers describe wherever it already was."""
    payload = good_payload()
    payload["executed"] = False
    baseline, problem = node_module.resolve_jaw_baseline(
        write(tmp_path, payload), grip_deg=-15.0
    )
    assert baseline is None
    assert "DRY RUN" in problem and "--execute" in problem


def test_a_mismatched_grip_angle_is_refused(node_module, tmp_path):
    """Recorded closing to -15, run commands -20: every residual would be off
    by 5 degrees against a reference that looks authoritative."""
    baseline, problem = node_module.resolve_jaw_baseline(
        write(tmp_path, good_payload(-15.0)), grip_deg=-20.0
    )
    assert baseline is None
    assert "wrong reference" in problem
    assert "-15.0" in problem and "-20.0" in problem


def test_a_small_grip_difference_is_tolerated(node_module, tmp_path):
    baseline, problem = node_module.resolve_jaw_baseline(
        write(tmp_path, good_payload(-15.0)), grip_deg=-15.3
    )
    assert problem is None and baseline is not None


def test_an_old_baseline_without_provenance_still_loads(node_module, tmp_path):
    """Files predating the executed/grip_command_deg fields must not break."""
    payload = {
        "empty_close_rad": float(np.deg2rad(-15.0)),
        "empty_close_rad_noise": float(np.deg2rad(0.1)),
        "source": "legacy",
    }
    baseline, problem = node_module.resolve_jaw_baseline(
        write(tmp_path, payload), grip_deg=-15.0
    )
    assert problem is None
    assert baseline.empty_close_effort is None
