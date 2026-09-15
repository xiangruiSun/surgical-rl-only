"""The precheck is the thing that lets the workspace grow safely: it must fail
closed on every case where the episode is not clearly inside limits."""

import numpy as np

from surgicai_rl_deploy.feasibility import FAIL, PASS, WARN, precheck
from surgicai_rl_deploy.plan import LiftSpec, build_plan

from conftest import REAL_GOAL_POS


def status_of(report, name):
    for check in report.checks:
        if check.name == name:
            return check.status
    raise AssertionError(f"no check named {name!r} in {[c.name for c in report.checks]}")


def test_clean_plan_passes(plan, baseline):
    report = precheck(plan, controller="d2", jaw_baseline=baseline)
    assert report.ok
    assert status_of(report, "lift_direction") == PASS


def test_live_run_refuses_an_unconfirmed_lift_sign(start_pose, baseline):
    plan = build_plan(start_pose, REAL_GOAL_POS, lift=LiftSpec())  # explicit=False
    report = precheck(plan, controller="d2", execute=True, jaw_baseline=baseline)
    assert not report.ok
    assert status_of(report, "lift_direction") == FAIL


def test_dry_run_only_warns_about_the_lift_sign(start_pose, baseline):
    plan = build_plan(start_pose, REAL_GOAL_POS, lift=LiftSpec())
    report = precheck(plan, controller="d2", execute=False, jaw_baseline=baseline)
    assert report.ok
    assert status_of(report, "lift_direction") == WARN


def test_a_lift_that_continues_into_the_approach_is_flagged(start_pose):
    """+z here keeps going the way the gripper came in -- toward the tissue."""
    plan = build_plan(
        start_pose, REAL_GOAL_POS,
        lift=LiftSpec(axis="z", sign=+1, explicit=True),
    )
    report = precheck(plan, controller="d2")
    assert status_of(report, "lift_vs_approach") == WARN


def test_the_correct_lift_direction_passes(plan):
    report = precheck(plan, controller="d2")
    assert status_of(report, "lift_vs_approach") == PASS


def test_a_far_goal_fails_closed(start_pose):
    far = (np.asarray(REAL_GOAL_POS) + np.array([0.30, 0.0, 0.0])).tolist()
    plan = build_plan(start_pose, far, lift=LiftSpec(sign=-1, explicit=True))
    report = precheck(plan, controller="d2")
    assert not report.ok
    assert status_of(report, "path_radius") == FAIL


def test_the_radius_limit_can_be_raised_deliberately(start_pose):
    far = (np.asarray(REAL_GOAL_POS) + np.array([0.10, 0.0, 0.0])).tolist()
    plan = build_plan(start_pose, far, lift=LiftSpec(sign=-1, explicit=True))
    assert not precheck(plan, controller="d2").ok
    assert precheck(plan, controller="d2", max_path_radius_cm=25.0).ok


def test_hard_limit_box_is_enforced(plan):
    report = precheck(
        plan, controller="d2",
        limit_low_m=[-0.01, -0.01, -0.01], limit_high_m=[0.01, 0.01, 0.01],
    )
    assert not report.ok
    assert status_of(report, "hard_limits") == FAIL


def test_hard_limit_box_passes_when_generous(plan):
    report = precheck(
        plan, controller="d2",
        limit_low_m=[-0.20, -0.20, -0.20], limit_high_m=[0.20, 0.20, 0.20],
    )
    assert status_of(report, "hard_limits") == PASS


def test_inverted_hard_limits_fail(plan):
    report = precheck(
        plan, controller="d2",
        limit_low_m=[0.1, 0.1, 0.1], limit_high_m=[-0.1, -0.1, -0.1],
    )
    assert not report.ok


def test_step_budget_fails_when_the_cap_is_too_tight(plan):
    report = precheck(plan, controller="d2", approach_max_steps=5)
    assert not report.ok
    assert status_of(report, "step_budget_approach") == FAIL


def test_a_tolerance_wider_than_the_lift_is_rejected(plan):
    """A 1 cm tolerance on a 1.5 cm lift would report success after 0.5 cm."""
    report = precheck(plan, controller="d2", success_trans_cm=1.5)
    assert not report.ok
    assert status_of(report, "lift_tolerance") == FAIL


def test_evidence_gate_without_a_baseline_is_refused(plan):
    report = precheck(plan, controller="d2", grasp_gate="evidence", jaw_baseline=None)
    assert not report.ok
    assert status_of(report, "grasp_gate") == FAIL


def test_evidence_gate_with_a_baseline_warns_but_runs(plan, baseline):
    report = precheck(
        plan, controller="d2", grasp_gate="evidence", jaw_baseline=baseline
    )
    assert report.ok
    assert status_of(report, "grasp_gate") == WARN


def test_manual_gate_is_the_quiet_default(plan, baseline):
    report = precheck(plan, controller="d2", grasp_gate="manual", jaw_baseline=baseline)
    assert status_of(report, "grasp_gate") == PASS


def test_always_gate_warns_on_a_live_run(plan, baseline):
    report = precheck(
        plan, controller="d2", grasp_gate="always", jaw_baseline=baseline, execute=True
    )
    assert status_of(report, "grasp_gate") == WARN


def test_unknown_gate_fails(plan):
    assert not precheck(plan, controller="d2", grasp_gate="vibes").ok


def test_rl_is_bounded_by_the_r6_support(plan, baseline):
    """The real task is out of distribution: the RL path must say so."""
    report = precheck(plan, controller="rl", jaw_baseline=baseline)
    assert status_of(report, "r6_training_support") == WARN
    assert report.ok  # a warning, not a refusal, unless --strict


def test_strict_turns_the_r6_warning_into_a_refusal(plan, baseline):
    report = precheck(plan, controller="rl", jaw_baseline=baseline, strict=True)
    assert not report.ok


def test_the_servo_is_not_bounded_by_the_trained_region(plan, baseline):
    """This is the whole point: geometry has no training support."""
    report = precheck(plan, controller="d2", jaw_baseline=baseline)
    assert status_of(report, "r6_training_support") == PASS
    for check in report.checks:
        if check.name == "r6_training_support":
            assert check.detail["applies"] is False
            # the RL-relevant numbers are still reported, for comparison
            assert check.detail["offenders"]


def test_report_renders_and_serialises(plan, baseline):
    report = precheck(plan, controller="d2", jaw_baseline=baseline)
    assert "PRECHECK PASS" in report.render()
    payload = report.as_dict()
    assert payload["ok"] is True
    assert {c["name"] for c in payload["checks"]} >= {
        "inputs", "lift_direction", "path_radius", "grasp_gate", "jaw_calibration"
    }
