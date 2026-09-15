"""The command line of the full pipeline, on both the offline tool and the node.

Anything that can be got wrong from a terminal at two in the morning, with an
arm powered up, belongs here.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REAL_GOAL_POS, REAL_START_POS, REAL_START_QUAT

DEPLOY = Path(__file__).resolve().parents[1]
SUTURE = ["-0.040", "0.005", "0.040"]
SUTURE_QUAT = ["0", "0", "0", "1"]


def offline(*extra, expect=0):
    cmd = [
        sys.executable, str(DEPLOY / "tools" / "offline_grasp_lift.py"),
        "--start-pos", *[str(v) for v in REAL_START_POS],
        "--start-quat", *[str(v) for v in REAL_START_QUAT],
        "--grasp-pos", *[str(v) for v in REAL_GOAL_POS],
        "--lift-sign", "-1", "--grasp-gate", "always",
        "--jaw-stops-at-deg", "-5", "--max-cycles", "3000",
        *extra,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=DEPLOY, timeout=300)
    assert out.returncode == expect, out.stdout + out.stderr
    return out.stdout + out.stderr


def test_the_whole_pipeline_runs_offline(tmp_path):
    js = tmp_path / "run.json"
    text = offline("--suture-pos", *SUTURE, "--suture-quat", *SUTURE_QUAT,
                   "--suture-confirmed", "--json-out", str(js))
    for phase in ("approach", "settle", "close", "observe", "lift",
                  "transport", "place", "hold"):
        assert f"--- phase: {phase}" in text, f"{phase} never ran"
    assert "reason      : success" in text

    data = json.loads(js.read_text())
    assert data["summary"]["reached_suture_pose"] is True
    assert data["summary"]["grasp_verified"] is False
    assert "suture_cm" in data["plan"]
    assert "via_cm" in data["plan"]
    # every phase transition is in the record, not just the printed trace
    events = {e.get("event") for e in data["summary"]["events"]}
    assert {"lift_reached", "transport_begin", "transport_reached",
            "suture_pose_reached"} <= events


def test_a_suture_position_without_orientation_is_refused():
    text = offline("--suture-pos", *SUTURE, expect=1)
    assert "needs --suture-quat" in text


def test_without_a_suture_pose_it_is_still_a_grasp_and_lift():
    text = offline()
    assert "--- phase: transport" not in text
    assert "reason      : success" in text


def test_the_precheck_refuses_an_unconfirmed_suture_pose_only_when_executing():
    # offline never executes, so it is a warning and the run proceeds
    text = offline("--suture-pos", *SUTURE, "--suture-quat", *SUTURE_QUAT)
    assert "suture_pose" in text
    assert "reason      : success" in text


def test_staging_needs_a_contract():
    text = offline("--stage", expect=1)
    assert "--stage needs a checkpoint contract" in text


def test_a_named_contract_is_enough_to_stage(tmp_path):
    js = tmp_path / "run.json"
    text = offline("--contract", "approach_upstream", "--stage",
                   "--json-out", str(js))
    assert "--- phase: stage" in text
    data = json.loads(js.read_text())
    assert "staged_cm" in data["plan"]


def test_direct_transport_is_flagged():
    text = offline("--suture-pos", *SUTURE, "--suture-quat", *SUTURE_QUAT,
                   "--suture-confirmed", "--transport-via", "direct")
    assert "transport_clearance" in text
    assert "can pass below the tissue plane" in text


def test_the_transport_clearance_is_configurable(tmp_path):
    js = tmp_path / "run.json"
    offline("--suture-pos", *SUTURE, "--suture-quat", *SUTURE_QUAT,
            "--suture-confirmed", "--transport-clearance-cm", "3.0",
            "--json-out", str(js))
    data = json.loads(js.read_text())
    via = np.asarray(data["plan"]["via_cm"])
    suture = np.asarray(data["plan"]["suture_cm"])
    assert np.linalg.norm(via - suture) == pytest.approx(3.0, abs=1e-6)


# ----------------------------------------------------------------------
# the ROS node's argument surface, with rclpy stubbed
# ----------------------------------------------------------------------
def test_the_node_accepts_the_pipeline_arguments(node_module):
    args = node_module.parse_args([
        "--grasp-pos", "0.1", "0.2", "0.3",
        "--suture-pos", "0.4", "0.5", "0.6",
        "--suture-quat", "0", "0", "0", "1",
        "--suture-confirmed", "--stage", "--contract", "approach_upstream",
        "--transport-via", "direct", "--transport-clearance-cm", "2.5",
        "--on-approach-failure", "servo",
        "--shadow-model", "/nowhere/place.zip",
        "--shadow-contract", "place_upstream",
        "--lift-sign", "-1",
    ])
    assert args.suture_pos == [0.4, 0.5, 0.6]
    assert args.suture_confirmed is True
    assert args.stage is True
    assert args.contract == "approach_upstream"
    assert args.transport_via == "direct"
    assert args.transport_clearance_cm == 2.5
    assert args.on_approach_failure == "servo"
    assert args.shadow_contract == "place_upstream"


def test_the_node_defaults_are_the_cautious_ones(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.suture_pos is None
    assert args.suture_confirmed is False
    assert args.stage is False
    assert args.shadow_model is None
    assert args.on_approach_failure == "hold"
    assert args.transport_via == "lift_height"
    assert args.execute is False


@pytest.mark.parametrize("bad", [
    ["--transport-via", "sideways"],
    ["--on-approach-failure", "improvise"],
    ["--contract", "made_up"],
])
def test_the_node_rejects_nonsense(node_module, bad):
    with pytest.raises(SystemExit):
        node_module.parse_args(["--grasp-pos", "0", "0", "0", *bad])


def test_the_run_pipeline_entry_point_exists():
    path = DEPLOY / "run_pipeline.py"
    assert path.is_file()
    text = path.read_text()
    assert "--suture-pos" in text and "--stage" in text
