"""The state machine, driven against the kinematic mock.

These are the tests that stand in for a robot: every transition, every abort
path, and the guarantee that no commanded step ever exceeds the safety caps.
"""

import numpy as np
import pytest

from surgicai_rl_deploy.controllers import D2Controller
from surgicai_rl_deploy.frames import rotation_error_rad
from surgicai_rl_deploy.loop import SafetyLimits
from surgicai_rl_deploy.mock import MockArm, MockJaw
from surgicai_rl_deploy.plan import LiftSpec, build_plan
from surgicai_rl_deploy.sequence import (
    PHASE_ABORTED,
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_DONE,
    PHASE_HOLD,
    PHASE_LIFT,
    PHASE_OBSERVE,
    PHASE_SETTLE,
    PHASE_WAIT_OPERATOR,
    GraspLiftSequencer,
    SequenceConfig,
)

from conftest import REAL_GOAL_POS


def run_episode(
    plan,
    baseline,
    *,
    gate="evidence",
    block_deg=2.0,
    drop_at=None,
    max_cycles=600,
    confirm=None,
    on_slip="abort",
    lag=0.0,
    noise_mm=0.0,
    limits=None,
    config=None,
):
    cfg = config or SequenceConfig(grasp_gate=gate, on_slip=on_slip)
    limits = limits or SafetyLimits()
    sequencer = GraspLiftSequencer(
        plan, D2Controller(staged=True), cfg, limits, baseline,
        confirm_callback=confirm,
    )
    arm = MockArm(
        plan.start,
        MockJaw(
            angle_rad=plan.jaw.approach_open_rad,
            block_at_rad=None if block_deg <= 0 else float(np.deg2rad(block_deg)),
            drop_at_step=drop_at,
        ),
        lag=lag,
        noise_mm=noise_mm,
        jaw_calibration=plan.jaw,
    )
    arm.prime_jaw()
    sequencer.begin(arm.state())

    steps = []
    for _ in range(max_cycles):
        step = sequencer.step(arm.state())
        steps.append(step)
        if step.done:
            break
        arm.apply(step.command)
    return sequencer, arm, steps


def lift_start_cycle(plan, baseline):
    """Cycle index at which a clean run first enters the lift.

    Computed rather than hardcoded: the phase timings depend on the settle
    window and the jaw ramp, and a test that pins them would break every time
    one of those is tuned, for no good reason.
    """
    _, _, steps = run_episode(plan, baseline)
    for step in steps:
        if step.phase == PHASE_LIFT:
            return step.index
    raise AssertionError("the clean run never reached the lift")


def phases_seen(steps):
    seen = []
    for step in steps:
        if not seen or seen[-1] != step.phase:
            seen.append(step.phase)
    return seen


# --- the happy path -------------------------------------------------------
def test_full_sequence_completes(plan, baseline):
    sequencer, arm, steps = run_episode(plan, baseline)
    assert sequencer.phase == PHASE_DONE
    assert sequencer.reason == "success"
    assert phases_seen(steps) == [
        PHASE_APPROACH, PHASE_SETTLE, PHASE_CLOSE, PHASE_OBSERVE, PHASE_LIFT, PHASE_HOLD
    ]


def test_the_arm_ends_15mm_above_the_grasp_pose(plan, baseline):
    _, arm, _ = run_episode(plan, baseline)
    lifted_cm = float(np.linalg.norm(arm.pose.p - plan.grasp.p) * 100.0)
    assert lifted_cm == pytest.approx(1.5, abs=0.05)


def test_the_lift_goes_the_way_it_was_told(plan, baseline):
    _, arm, _ = run_episode(plan, baseline)
    delta = arm.pose.p - plan.grasp.p
    assert delta[2] < 0  # sign=-1
    assert abs(delta[0]) < 5e-4 and abs(delta[1]) < 5e-4


def test_orientation_is_frozen_through_the_grasp_and_lift(plan, baseline):
    _, arm, steps = run_episode(plan, baseline)
    after_settle = [s for s in steps if s.phase != PHASE_APPROACH]
    for step in after_settle:
        assert rotation_error_rad(step.command.pose, plan.grasp) < np.deg2rad(1.0)


def test_the_jaw_opens_for_the_approach_and_squeezes_after(plan, baseline):
    _, _, steps = run_episode(plan, baseline)
    approach = [s for s in steps if s.phase == PHASE_APPROACH]
    lift = [s for s in steps if s.phase == PHASE_LIFT]
    assert all(s.command.jaw_rad == plan.jaw.approach_open_rad for s in approach)
    assert all(s.command.jaw_rad == pytest.approx(plan.jaw.grip_rad) for s in lift)


def test_the_jaw_ramps_rather_than_snapping_shut(plan, baseline):
    cfg = SequenceConfig(grasp_gate="evidence")
    _, _, steps = run_episode(plan, baseline, config=cfg)
    close = [s.command.jaw_rad for s in steps if s.phase == PHASE_CLOSE]
    assert len(close) > 5
    deltas = np.diff(close)
    assert np.all(np.abs(deltas) <= cfg.jaw_ramp_rad + 1e-9)
    assert close[-1] == pytest.approx(plan.jaw.grip_rad)


def test_success_never_claims_a_verified_grasp(plan, baseline):
    sequencer, _, _ = run_episode(plan, baseline)
    summary = sequencer.summary()
    assert summary["grasp_verified"] is False
    assert "no grasp sensor" in summary["grasp_verification_note"]


# --- the gate -------------------------------------------------------------
def test_evidence_gate_refuses_an_empty_gripper(plan, baseline):
    sequencer, arm, steps = run_episode(plan, baseline, block_deg=0.0)
    assert sequencer.phase == PHASE_ABORTED
    assert "empty gripper" in sequencer.reason
    # and it never left the grasp pose
    assert np.linalg.norm(arm.pose.p - plan.grasp.p) < 1e-3


def test_gate_never_stops_after_the_close(plan, baseline):
    sequencer, arm, steps = run_episode(plan, baseline, gate="never")
    assert sequencer.phase == PHASE_DONE
    assert "stopped after the jaw closed" in sequencer.reason
    assert PHASE_LIFT not in phases_seen(steps)
    assert np.linalg.norm(arm.pose.p - plan.grasp.p) < 1e-3


def test_gate_always_lifts_an_empty_gripper(plan, baseline):
    """Rehearsal mode: useful with nothing in the jaws, dangerous otherwise."""
    sequencer, _, steps = run_episode(plan, baseline, gate="always", block_deg=0.0)
    assert sequencer.phase == PHASE_DONE
    assert PHASE_LIFT in phases_seen(steps)


def test_manual_gate_waits_for_a_human(plan, baseline):
    sequencer, _, steps = run_episode(
        plan, baseline, gate="manual", confirm=lambda: None, max_cycles=200
    )
    assert sequencer.phase == PHASE_WAIT_OPERATOR
    assert PHASE_LIFT not in phases_seen(steps)


def test_manual_gate_lifts_once_confirmed(plan, baseline):
    answers = iter([None, None, True])

    def confirm():
        return next(answers, True)

    sequencer, arm, steps = run_episode(plan, baseline, gate="manual", confirm=confirm)
    assert sequencer.phase == PHASE_DONE
    assert PHASE_LIFT in phases_seen(steps)
    assert float(np.linalg.norm(arm.pose.p - plan.grasp.p) * 100.0) == pytest.approx(
        1.5, abs=0.05
    )


def test_manual_gate_stops_when_declined(plan, baseline):
    sequencer, arm, steps = run_episode(
        plan, baseline, gate="manual", confirm=lambda: False
    )
    assert sequencer.phase == PHASE_DONE
    assert "declined" in sequencer.reason
    assert PHASE_LIFT not in phases_seen(steps)


def test_operator_timeout_aborts(plan, baseline):
    cfg = SequenceConfig(grasp_gate="manual", operator_timeout_steps=5)
    sequencer, _, _ = run_episode(
        plan, baseline, gate="manual", confirm=lambda: None, config=cfg
    )
    assert sequencer.phase == PHASE_ABORTED
    assert "timed out" in sequencer.reason


# --- losing the needle ----------------------------------------------------
def test_a_slip_during_the_lift_aborts(plan, baseline):
    drop_at = lift_start_cycle(plan, baseline) + 2
    sequencer, _, steps = run_episode(plan, baseline, drop_at=drop_at)
    assert sequencer.phase == PHASE_ABORTED
    assert "evidence disappeared" in sequencer.reason
    assert any(
        e.get("event") == "jaw_evidence_lost" for s in steps for e in s.events
    )


def test_a_slip_can_be_logged_without_aborting(plan, baseline):
    sequencer, _, steps = run_episode(
        plan, baseline, drop_at=lift_start_cycle(plan, baseline) + 2,
        on_slip="continue",
    )
    assert sequencer.phase == PHASE_DONE
    assert any(
        e.get("event") == "jaw_evidence_lost" for s in steps for e in s.events
    )


def test_on_slip_lower_returns_to_the_grasp_pose(plan, baseline):
    sequencer, arm, _ = run_episode(
        plan, baseline, drop_at=lift_start_cycle(plan, baseline) + 2,
        on_slip="lower",
    )
    assert sequencer.phase == PHASE_DONE
    assert "lowered back" in sequencer.reason
    assert np.linalg.norm(arm.pose.p - plan.grasp.p) < 3e-3


def test_slip_needs_a_streak_not_one_sample(plan, baseline):
    """One noisy reading must not tear down a good lift."""
    cfg = SequenceConfig(grasp_gate="evidence", slip_streak=4)
    sequencer = GraspLiftSequencer(
        plan, D2Controller(staged=True), cfg, SafetyLimits(), baseline
    )
    sequencer.evidence_established = True
    from surgicai_rl_deploy.jaw import evaluate_jaw_evidence

    empty = evaluate_jaw_evidence(
        baseline.empty_close_rad, baseline.empty_close_rad, 0.02, baseline
    )
    events = []
    for _ in range(3):
        assert sequencer._check_slip(empty, events) is None
    assert sequencer._check_slip(empty, events) == "abort"


# --- aborts ---------------------------------------------------------------
def test_an_unreachable_grasp_pose_aborts_the_approach(start_pose, baseline, jaw_cal):
    """A lagging arm that never converges must stop, not grind forever."""
    plan = build_plan(
        start_pose, REAL_GOAL_POS,
        lift=LiftSpec(sign=-1, explicit=True), jaw=jaw_cal,
    )
    cfg = SequenceConfig(grasp_gate="evidence", approach_max_steps=8)
    sequencer, _, _ = run_episode(plan, baseline, config=cfg, lag=0.95)
    assert sequencer.phase == PHASE_ABORTED
    assert "approach" in sequencer.reason


def test_settle_timeout_aborts(plan, baseline):
    """The arm arrives but never holds still: do not close on a moving wrist."""
    cfg = SequenceConfig(
        grasp_gate="evidence", settle_timeout_steps=6,
        settle_translation_tol_mm=1e-9, settle_rotation_tol_deg=1e-9,
    )
    sequencer, _, _ = run_episode(plan, baseline, config=cfg, noise_mm=0.5)
    assert sequencer.phase == PHASE_ABORTED
    assert "settle timeout" in sequencer.reason


def test_a_terminal_sequencer_stops_publishing_and_never_opens_the_jaw(plan, baseline):
    sequencer, arm, steps = run_episode(
        plan, baseline, drop_at=lift_start_cycle(plan, baseline) + 2
    )
    assert sequencer.phase == PHASE_ABORTED
    held = sequencer.step(arm.state())
    assert held.command.publish_pose is False
    assert held.command.publish_jaw is False
    # the jaw command still squeezes; it is never re-opened automatically
    assert held.command.jaw_rad == pytest.approx(plan.jaw.grip_rad)


# --- the safety envelope --------------------------------------------------
def test_no_command_ever_exceeds_the_per_step_caps(plan, baseline):
    limits = SafetyLimits(max_step_translation_mm=2.5, max_step_rotation_deg=5.0)
    _, _, steps = run_episode(plan, baseline, limits=limits)
    for step in steps:
        if not step.command.publish_pose:
            continue
        move_mm = float(
            np.linalg.norm(step.command.pose.p - step.measured.pose.p) * 1000.0
        )
        turn_deg = float(
            np.degrees(rotation_error_rad(step.command.pose, step.measured.pose))
        )
        assert move_mm <= limits.max_step_translation_mm + 1e-6
        assert turn_deg <= limits.max_step_rotation_deg + 1e-6


def test_every_command_stays_inside_the_padded_workspace_box(plan, baseline):
    limits = SafetyLimits(workspace_pad_cm=2.0)
    _, _, steps = run_episode(plan, baseline, limits=limits)
    low, high = plan.bounding_box_m(pad_cm=limits.workspace_pad_cm)
    for step in steps:
        assert np.all(step.command.pose.p >= low - 1e-9)
        assert np.all(step.command.pose.p <= high + 1e-9)


def test_the_jaw_command_never_passes_the_grip_angle(plan, baseline):
    _, _, steps = run_episode(plan, baseline)
    for step in steps:
        assert step.command.jaw_rad >= plan.jaw.grip_rad - 1e-9
        assert step.command.jaw_rad <= plan.jaw.open_rad + 1e-9


def test_tracking_lag_aborts_rather_than_chasing(plan, baseline):
    limits = SafetyLimits(max_tracking_error_cm=0.05)
    sequencer, _, _ = run_episode(plan, baseline, limits=limits, lag=0.9)
    assert sequencer.phase == PHASE_ABORTED


# --- config validation ----------------------------------------------------
def test_unknown_gate_is_rejected():
    with pytest.raises(ValueError, match="grasp gate"):
        SequenceConfig(grasp_gate="maybe")


def test_unknown_slip_policy_is_rejected():
    with pytest.raises(ValueError, match="on_slip"):
        SequenceConfig(on_slip="pray")


def test_nonpositive_streaks_are_rejected():
    with pytest.raises(ValueError):
        SequenceConfig(evidence_streak=0)


def test_step_before_begin_is_an_error(plan, baseline):
    sequencer = GraspLiftSequencer(
        plan, D2Controller(), SequenceConfig(), SafetyLimits(), baseline
    )
    with pytest.raises(RuntimeError, match="begin"):
        sequencer.step(
            __import__("surgicai_rl_deploy.sequence", fromlist=["ArmState"]).ArmState(
                pose=plan.start
            )
        )


# --- trace ----------------------------------------------------------------
def test_every_step_serialises_with_the_disclaimer(plan, baseline):
    _, _, steps = run_episode(plan, baseline)
    for step in steps:
        payload = step.as_dict()
        assert "phase" in payload
        if payload["jaw"] is not None:
            assert payload["jaw"]["grasp_verified"] is False
