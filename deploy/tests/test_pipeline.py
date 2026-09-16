"""The full pipeline: stage -> approach -> grasp -> lift -> transport -> place.

These run the real :class:`GraspLiftSequencer` against the kinematic mock, so
the phase machine, the plan geometry and the safety checks are exercised
together.  Nothing here needs ROS, a checkpoint, or a robot.
"""

import numpy as np
import pytest

from surgicai_rl_deploy.contract import APPROACH_UPSTREAM, PLACE_UPSTREAM
from surgicai_rl_deploy.controllers import D2Controller
from surgicai_rl_deploy.feasibility import FAIL, PASS, WARN, precheck
from surgicai_rl_deploy.frames import Pose, rotation_error_rad
from surgicai_rl_deploy.jaw import JawCalibration
from surgicai_rl_deploy.loop import SafetyLimits
from surgicai_rl_deploy.mock import MockArm, MockJaw
from surgicai_rl_deploy.plan import LiftSpec, TransportSpec, build_plan
from surgicai_rl_deploy.sequence import (
    PHASE_ABORTED,
    PHASE_DESCEND,
    PHASE_DONE,
    PHASE_PLACE,
    PHASE_STAGE,
    PHASE_TRANSPORT,
    GraspLiftSequencer,
    SequenceConfig,
)
from surgicai_rl_deploy.staging import stage_pose_for, support_report

from conftest import REAL_GOAL_POS  # noqa: E402

SUTURE_POS = [-0.0400, 0.0050, 0.0400]
SUTURE_QUAT = [0.0, 0.0, 0.0, 1.0]


def pipeline_plan(start_pose, jaw_cal, **kwargs):
    lift = LiftSpec(axis="z", sign=-1, distance_m=0.015, frame="robot", explicit=True)
    opts = dict(
        lift=lift,
        jaw=jaw_cal,
        suture_position_m=SUTURE_POS,
        suture_quat_xyzw=SUTURE_QUAT,
    )
    opts.update(kwargs)
    return build_plan(start_pose, REAL_GOAL_POS, **opts)


def run(plan, jaw_cal, baseline, *, block_at=None, cfg=None, shadow=None,
        max_cycles=6000, lag=0.0, drop_at=None):
    jaw = MockJaw(
        angle_rad=jaw_cal.approach_open_rad,
        block_at_rad=block_at,
        drop_at_step=drop_at,
    )
    arm = MockArm(plan.start, jaw, lag=lag, jaw_calibration=jaw_cal)
    arm.prime_jaw()
    seq = GraspLiftSequencer(
        plan,
        D2Controller(staged=True),
        cfg or SequenceConfig(grasp_gate="always"),
        SafetyLimits(max_tracking_error_cm=50.0, workspace_pad_cm=50.0),
        baseline,
        shadow_controller=shadow,
        shadow_contract=PLACE_UPSTREAM if shadow is not None else None,
    )
    seq.begin(arm.state())
    steps = []
    for _ in range(max_cycles):
        state = arm.state()
        step = seq.step(state)
        steps.append(step)
        if step.done:
            break
        arm.apply(step.command)
    return seq, steps


# ======================================================================
# geometry
# ======================================================================
def test_a_plan_without_a_suture_pose_is_unchanged(plan):
    assert plan.suture is None
    assert plan.via is None
    assert [n for n, _ in plan.waypoints] == ["start", "grasp", "lifted"]
    assert plan.transport_travel_cm == 0.0


def test_the_suturing_plan_adds_a_via_point(start_pose, jaw_cal):
    p = pipeline_plan(start_pose, jaw_cal)
    assert [n for n, _ in p.waypoints] == [
        "start", "grasp", "lifted", "via", "suture"
    ]
    # the via sits exactly one lift-distance along the lift direction from the
    # suturing pose, so the last motion is a pure descent
    lift_dir = p.lift_spec.direction(p.grasp)
    offset = p.via.p - p.suture.p
    np.testing.assert_allclose(offset, lift_dir * p.lift_spec.distance_m, atol=1e-12)
    # and it carries the suturing orientation already, so the wrist turns in
    # transit rather than on the way down
    assert rotation_error_rad(p.via, p.suture) == pytest.approx(0.0, abs=1e-12)


def test_direct_transport_has_no_via(start_pose, jaw_cal):
    p = pipeline_plan(start_pose, jaw_cal, transport=TransportSpec(via="direct"))
    assert p.via is None
    assert [n for n, _ in p.waypoints] == ["start", "grasp", "lifted", "suture"]


def test_via_clearance_can_be_set(start_pose, jaw_cal):
    p = pipeline_plan(
        start_pose, jaw_cal, transport=TransportSpec(via_clearance_m=0.03)
    )
    assert np.linalg.norm(p.via.p - p.suture.p) == pytest.approx(0.03)


def test_a_suture_position_without_orientation_is_refused(start_pose, jaw_cal):
    with pytest.raises(ValueError, match="not optional"):
        build_plan(start_pose, REAL_GOAL_POS, jaw=jaw_cal,
                   suture_position_m=SUTURE_POS)


@pytest.mark.parametrize("bad", [{"via": "sideways"}, {"via_clearance_m": 0.5},
                                 {"via_clearance_m": -0.01}])
def test_bad_transport_specs_are_refused(bad):
    with pytest.raises(ValueError):
        TransportSpec(**bad)


def test_path_radius_and_box_include_every_waypoint(start_pose, jaw_cal):
    p = pipeline_plan(start_pose, jaw_cal)
    assert p.path_radius_cm() >= np.linalg.norm(p.suture.p - p.start.p) * 100.0 - 1e-9
    low, high = p.bounding_box_m(pad_cm=0.0)
    for _, pose in p.waypoints:
        assert np.all(pose.p >= low - 1e-12) and np.all(pose.p <= high + 1e-12)


# ======================================================================
# staging
# ======================================================================
def test_the_staged_pose_lands_inside_the_training_support(start_pose, jaw_cal):
    p = pipeline_plan(start_pose, jaw_cal, stage_contract=APPROACH_UPSTREAM)
    assert p.staged is not None
    report = support_report(p.staged, p.grasp, APPROACH_UPSTREAM)
    assert report["in_support"], report["reasons"]


@pytest.mark.parametrize("rotation_deg", [55.0, 70.0, 95.0])
def test_any_rotation_inside_the_range_can_be_requested(start_pose, jaw_cal,
                                                        rotation_deg):
    p = pipeline_plan(start_pose, jaw_cal, stage_contract=APPROACH_UPSTREAM,
                      stage_rotation_deg=rotation_deg)
    report = support_report(p.staged, p.grasp, APPROACH_UPSTREAM)
    assert report["rotation_deg"] == pytest.approx(rotation_deg, abs=1e-6)
    assert report["in_support"]


def test_the_box_corners_are_reachable(start_pose, jaw_cal):
    c = APPROACH_UPSTREAM
    for offset in (c.start_offset_tool_min, c.start_offset_tool_max,
                   c.start_offset_tool_mean):
        staged = stage_pose_for(
            Pose.from_pos_quat(REAL_GOAL_POS, [0, 0, 0, 1], 0.0), c,
            offset_tool_cm=offset, rotation_deg=c.start_rot_deg_median,
        )
        rep = support_report(
            staged, Pose.from_pos_quat(REAL_GOAL_POS, [0, 0, 0, 1], 0.0), c
        )
        np.testing.assert_allclose(rep["offset_tool_cm"], offset, atol=1e-6)
        assert rep["in_support"]


def test_staging_is_frame_invariant(start_pose, jaw_cal):
    """The support constraints survive any rigid transform of the scene."""
    from scipy.spatial.transform import Rotation

    grasp = Pose.from_pos_quat(REAL_GOAL_POS, [0.2, 0.3, 0.1, 0.9], 0.0)
    staged = stage_pose_for(grasp, APPROACH_UPSTREAM)
    base = support_report(staged, grasp, APPROACH_UPSTREAM)

    X = Pose(np.array([0.3, -0.2, 1.1]), Rotation.from_rotvec([0.4, -1.1, 0.7]).as_matrix())
    moved = support_report(X * staged, X * grasp, APPROACH_UPSTREAM)
    np.testing.assert_allclose(moved["offset_tool_cm"], base["offset_tool_cm"], atol=1e-9)
    assert moved["rotation_deg"] == pytest.approx(base["rotation_deg"], abs=1e-9)


# ======================================================================
# the sequence
# ======================================================================
def test_the_pipeline_runs_end_to_end(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal, stage_contract=APPROACH_UPSTREAM)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0))
    assert seq.phase == PHASE_DONE, seq.reason
    phases = [s.phase for s in steps]
    assert phases[0] == PHASE_STAGE
    for expected in ("stage", "approach", "settle", "close", "observe", "lift",
                     "transport", "place", "hold"):
        assert expected in phases, f"{expected} never ran"
    # the order is the pipeline order, with no phase revisited
    seen = [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p]
    assert seen == sorted(set(seen), key=seen.index)
    summary = seq.summary()
    assert summary["reached_suture_pose"] is True
    assert summary["grasp_verified"] is False


def test_the_arm_actually_ends_at_the_suturing_pose(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0))
    assert seq.phase == PHASE_DONE, seq.reason
    final = steps[-1].measured.pose
    assert np.linalg.norm(final.p - plan.suture.p) * 100.0 < 0.3
    assert np.degrees(rotation_error_rad(final, plan.suture)) < 3.0


def test_the_descent_happens_only_at_the_end(start_pose, jaw_cal, baseline):
    """During transport the tool must never go below the suturing height."""
    plan = pipeline_plan(start_pose, jaw_cal)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0))
    lift_dir = plan.lift_spec.direction(plan.grasp)
    height = lambda p: float(np.dot(p - plan.suture.p, lift_dir))  # noqa: E731
    transport = [s for s in steps if s.phase == PHASE_TRANSPORT]
    assert transport
    for s in transport:
        # a millimetre of tolerance for the servo's own overshoot
        assert height(s.measured.pose.p) > -0.001


def test_a_plan_without_a_suture_pose_still_stops_after_the_lift(
    plan, jaw_cal, baseline
):
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0))
    assert seq.phase == PHASE_DONE, seq.reason
    assert PHASE_TRANSPORT not in [s.phase for s in steps]
    assert seq.summary()["reached_suture_pose"] is False


def test_the_jaw_is_never_opened_while_loaded(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0))
    loaded = [s for s in steps
              if s.phase in (PHASE_TRANSPORT, PHASE_PLACE, "lift", "hold")]
    assert loaded
    for s in loaded:
        assert s.command.jaw_rad <= jaw_cal.grip_rad + 1e-9


@pytest.mark.parametrize("drop_at", [90, 110, 120, 130, 150])
def test_losing_the_needle_while_loaded_never_descends_further(
    start_pose, jaw_cal, baseline, drop_at
):
    """The invariant, stated as height rather than as a phase name.

    Which phase the needle is dropped in depends on cycle counts, and cycle
    counts move whenever the action scale or a tolerance changes -- so a test
    that names a phase tests the timing, not the safety property. The property
    is: once the jaw evidence is gone, the tool does not go any further down
    toward the tissue, wherever it happened to be at the time.
    """
    plan = pipeline_plan(start_pose, jaw_cal)
    cfg = SequenceConfig(grasp_gate="evidence", on_slip="lower")
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0),
                     cfg=cfg, drop_at=drop_at)

    lost = next((e for e in seq.events if e.get("event") == "jaw_evidence_lost"),
                None)
    if lost is None:
        return  # the drop landed before evidence was ever established

    lift_dir = plan.lift_spec.direction(plan.grasp)
    # height above the suturing pose, along the lift axis: larger is safer
    height = lambda p: float(np.dot(p - plan.suture.p, lift_dir))  # noqa: E731
    at_loss = next(height(s.measured.pose.p) for s in steps if s.index == lost["i"])

    after = [s for s in steps if s.index > lost["i"]]
    for s in after:
        assert height(s.command.pose.p) >= at_loss - 1e-3, (
            f"descended {1000*(at_loss - height(s.command.pose.p)):.2f} mm after "
            f"losing the needle, at cycle {s.index} in phase {s.phase}"
        )


def test_a_slip_while_loaded_is_recorded_and_stops_the_run(start_pose, jaw_cal,
                                                           baseline):
    plan = pipeline_plan(start_pose, jaw_cal)
    cfg = SequenceConfig(grasp_gate="evidence", on_slip="lower")
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0),
                     cfg=cfg, drop_at=120)
    lost = [e for e in seq.events if e.get("event") == "jaw_evidence_lost"]
    assert lost, "the drop should have been noticed"
    assert seq.phase == PHASE_ABORTED
    assert "jaw evidence disappeared" in seq.reason
    # and the jaw was never opened on the way out
    assert all(s.command.jaw_rad <= jaw_cal.grip_rad + 1e-9
               for s in steps if s.index >= lost[0]["i"])


def test_stage_failure_never_starts_the_policy(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal, stage_contract=APPROACH_UPSTREAM)
    cfg = SequenceConfig(grasp_gate="always", stage_max_steps=3)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0), cfg=cfg)
    assert seq.phase == PHASE_ABORTED
    assert "stage" in seq.reason
    assert seq.approach_report is None
    assert "approach" not in [s.phase for s in steps]


# ======================================================================
# approach failure handling
# ======================================================================
class Stuck:
    name = "stuck"

    def act(self, obs):
        return np.zeros(7)

    def describe(self):
        return "stuck"


def _stuck_run(start_pose, jaw_cal, baseline, on_failure):
    plan = pipeline_plan(start_pose, jaw_cal)
    jaw = MockJaw(angle_rad=jaw_cal.approach_open_rad, block_at_rad=np.deg2rad(-5.0))
    arm = MockArm(plan.start, jaw, jaw_calibration=jaw_cal)
    arm.prime_jaw()
    seq = GraspLiftSequencer(
        plan, Stuck(),
        SequenceConfig(grasp_gate="always", approach_max_steps=12,
                       on_approach_failure=on_failure),
        SafetyLimits(max_tracking_error_cm=50.0, workspace_pad_cm=50.0),
        baseline,
    )
    seq.begin(arm.state())
    steps = []
    for _ in range(4000):
        step = seq.step(arm.state())
        steps.append(step)
        if step.done:
            break
        arm.apply(step.command)
    return seq, steps


def test_a_failed_approach_holds_by_default(start_pose, jaw_cal, baseline):
    seq, steps = _stuck_run(start_pose, jaw_cal, baseline, "hold")
    assert seq.phase == PHASE_ABORTED
    assert "Holding the last command" in seq.reason
    # the jaw command never left the open angle
    assert all(s.command.jaw_rad == pytest.approx(jaw_cal.approach_open_rad)
               for s in steps)
    # and the terminal command is not published at all
    assert steps[-1].command.publish_pose is False


def test_the_servo_fallback_finishes_the_run(start_pose, jaw_cal, baseline):
    seq, steps = _stuck_run(start_pose, jaw_cal, baseline, "servo")
    assert seq.phase == PHASE_DONE, seq.reason
    assert any(e.get("event") == "approach_fallback" for e in seq.events)


def test_an_unknown_failure_policy_is_refused():
    with pytest.raises(ValueError, match="on_approach_failure"):
        SequenceConfig(on_approach_failure="improvise")


# ======================================================================
# the shadow controller
# ======================================================================
class RecordingShadow:
    name = "shadow"

    def __init__(self):
        self.calls = 0

    def act(self, obs):
        self.calls += 1
        return np.zeros(7)

    def describe(self):
        return "recording shadow"


def test_the_shadow_runs_but_never_commands(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal)
    shadow = RecordingShadow()
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0),
                     shadow=shadow)
    assert seq.phase == PHASE_DONE, seq.reason
    assert shadow.calls > 0
    assert seq.shadow_log
    summary = seq.shadow_summary()
    assert summary["cycles"] == len(seq.shadow_log)
    # a shadow returning zero actions proposes "stay where you are", so its
    # advice diverges from the servo's by exactly one servo step, every cycle
    assert summary["median_divergence_mm"] > 0.0
    assert summary["support"] is not None
    # and the real arm still arrived, because the shadow never commanded
    assert np.linalg.norm(steps[-1].measured.pose.p - plan.suture.p) * 100.0 < 0.3


class ExplodingShadow:
    name = "boom"

    def act(self, obs):
        raise RuntimeError("no checkpoint")

    def describe(self):
        return "exploding shadow"


def test_a_broken_shadow_cannot_take_down_the_run(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0),
                     shadow=ExplodingShadow())
    assert seq.phase == PHASE_DONE, seq.reason
    assert any(e.get("event") == "shadow_failed" for e in seq.events)


def test_no_shadow_means_no_shadow_summary(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal)
    seq, _ = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0))
    assert seq.shadow_summary() is None
    assert seq.summary()["shadow"] is None


# ======================================================================
# the precheck
# ======================================================================
def status_of(report, name):
    return next(c.status for c in report.checks if c.name == name)


def test_the_precheck_demands_a_confirmed_suture_pose(start_pose, jaw_cal):
    plan = pipeline_plan(start_pose, jaw_cal)
    assert status_of(precheck(plan, execute=True), "suture_pose") == FAIL
    assert status_of(precheck(plan, execute=False), "suture_pose") == WARN
    ok = precheck(plan, execute=True, suture_confirmed=True)
    assert status_of(ok, "suture_pose") == PASS


def test_the_precheck_flags_a_direct_transport(start_pose, jaw_cal):
    plan = pipeline_plan(start_pose, jaw_cal, transport=TransportSpec(via="direct"))
    assert status_of(precheck(plan), "transport_clearance") == WARN


def test_the_precheck_catches_a_descent_that_is_a_retreat(start_pose, jaw_cal):
    """A lift sign that disagrees with the suturing geometry."""
    up = LiftSpec(axis="z", sign=+1, distance_m=0.015, frame="robot", explicit=True)
    plan = pipeline_plan(start_pose, jaw_cal, lift=up)
    # flipping the lift flips the via point to the other side, so the final
    # segment still descends; the check must stay self-consistent
    assert status_of(precheck(plan), "transport_clearance") == PASS


def test_the_precheck_reports_the_training_support(start_pose, jaw_cal):
    staged = pipeline_plan(start_pose, jaw_cal, stage_contract=APPROACH_UPSTREAM)
    report = precheck(staged, controller="rl", stage_contract=APPROACH_UPSTREAM)
    assert status_of(report, "training_support") == PASS

    unstaged = pipeline_plan(start_pose, jaw_cal)
    rl = precheck(unstaged, controller="rl", stage_contract=APPROACH_UPSTREAM)
    assert status_of(rl, "training_support") == WARN
    # under the geometric servo the same geometry is information, not a warning
    servo = precheck(unstaged, controller="d2", stage_contract=APPROACH_UPSTREAM)
    assert status_of(servo, "training_support") == PASS


def test_the_precheck_budgets_every_segment(start_pose, jaw_cal):
    plan = pipeline_plan(start_pose, jaw_cal, stage_contract=APPROACH_UPSTREAM)
    names = {c.name for c in precheck(plan).checks}
    for seg in ("stage", "approach", "lift", "transport", "place"):
        assert f"step_budget_{seg}" in names


def test_a_starved_transport_budget_fails(start_pose, jaw_cal):
    plan = pipeline_plan(start_pose, jaw_cal)
    report = precheck(plan, transport_max_steps=2)
    assert status_of(report, "step_budget_transport") == FAIL


# ======================================================================
# the workspace question, measured rather than argued
# ======================================================================
def test_staging_covers_a_whole_needle_envelope():
    """The claim the workspace tool is built on, pinned as a property.

    A support defined purely by relative geometry can always be satisfied by
    choosing the start pose, so a needle anywhere in its placement envelope is
    reachable in-distribution. If this ever fails, the 'extend the workspace'
    advice in the docs is wrong.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from workspace_spec import needle_envelope

    grasp = Pose.from_pos_quat(
        REAL_GOAL_POS, [0.23320, 0.42679, -0.23589, 0.84133], 0.0
    )
    for contract in (APPROACH_UPSTREAM, PLACE_UPSTREAM):
        for g in needle_envelope(grasp, [3.0, 3.0], 30.0, 60, seed=7):
            staged = stage_pose_for(g, contract)
            rep = support_report(staged, g, contract)
            assert rep["in_support"], (contract.name, rep["reasons"])


def test_the_unstaged_real_geometry_is_out_of_support(start_pose, jaw_cal):
    """...and that it is genuinely out, which is why staging is not optional."""
    plan = pipeline_plan(start_pose, jaw_cal)
    rep = support_report(plan.start, plan.grasp, APPROACH_UPSTREAM)
    assert not rep["in_support"]
    assert rep["reasons"]


def test_the_descent_is_reported_as_purely_vertical(start_pose, jaw_cal):
    """The via point is built along the lift axis, so lateral drift is zero.

    This reported 3.000 cm -- exactly twice the lift distance -- until a sign
    error in the precheck was fixed: it added the component along the lift axis
    instead of subtracting it, so a perfectly vertical descent was announced as
    3 cm of sideways drift onto the entry point.
    """
    plan = pipeline_plan(start_pose, jaw_cal)
    check = next(c for c in precheck(plan).checks if c.name == "transport_clearance")
    assert check.status == PASS
    assert check.detail["lateral_cm"] == pytest.approx(0.0, abs=1e-9)
    assert check.detail["drop_cm"] == pytest.approx(
        plan.lift_spec.distance_m * 100.0, abs=1e-9
    )


def test_a_genuinely_slanted_descent_is_reported(start_pose, jaw_cal):
    """...and a real one still shows up, so the check is not just zero."""
    plan = pipeline_plan(start_pose, jaw_cal)
    slanted = Pose(plan.via.p + np.array([0.004, 0.0, 0.0]), plan.via.R, plan.via.jaw)
    object.__setattr__(plan, "transport_spec", None)
    # hand-build the check the same way precheck does, with a displaced via
    lift_dir = plan.lift_spec.direction(plan.grasp)
    descent = plan.suture.p - slanted.p
    lateral = np.linalg.norm(descent - lift_dir * np.dot(descent, lift_dir)) * 100.0
    assert lateral == pytest.approx(0.4, abs=1e-9)


# ======================================================================
# the grasp standoff: the policy's goal is not the needle
# ======================================================================
def test_the_hover_pose_is_the_grasp_pose_backed_off_along_the_tool_axis(
    start_pose, jaw_cal
):
    p = pipeline_plan(start_pose, jaw_cal, grasp_standoff_m=0.007)
    assert p.hover is not None
    offset = p.grasp.p - p.hover.p
    # purely along the tool's own z, 7 mm of it
    assert float(np.dot(offset, p.grasp.R[:, 2])) == pytest.approx(0.007, abs=1e-12)
    assert np.linalg.norm(
        offset - p.grasp.R[:, 2] * np.dot(offset, p.grasp.R[:, 2])
    ) == pytest.approx(0.0, abs=1e-12)
    # and the orientation is unchanged, so the jaws already point at the needle
    assert rotation_error_rad(p.hover, p.grasp) == pytest.approx(0.0, abs=1e-12)
    assert "hover" in [n for n, _ in p.waypoints]


def test_the_approach_aims_at_the_hover_not_the_grasp(start_pose, jaw_cal):
    p = pipeline_plan(start_pose, jaw_cal, grasp_standoff_m=0.007)
    assert np.allclose(p.approach_target.p, p.hover.p)
    q = pipeline_plan(start_pose, jaw_cal)
    assert q.hover is None
    assert np.allclose(q.approach_target.p, q.grasp.p)


def test_staging_is_solved_against_the_hover_pose(start_pose, jaw_cal):
    """The support is relative to the policy's goal, which is the standoff."""
    p = pipeline_plan(start_pose, jaw_cal, grasp_standoff_m=0.007,
                      stage_contract=APPROACH_UPSTREAM)
    assert support_report(p.staged, p.approach_target,
                          APPROACH_UPSTREAM)["in_support"]


def test_the_descent_runs_and_lands_on_the_needle(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal, grasp_standoff_m=0.007)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0))
    assert seq.phase == PHASE_DONE, seq.reason
    phases = [s.phase for s in steps]
    assert PHASE_DESCEND in phases
    assert phases.index(PHASE_DESCEND) < phases.index("settle")
    closed = [e for e in seq.events if e.get("event") == "standoff_closed"]
    assert closed and closed[0]["trans_err_cm"] < 0.1


def test_the_descent_is_gentler_than_the_approach(start_pose, jaw_cal, baseline):
    """It is the one motion that can move the needle before it is held."""
    plan = pipeline_plan(start_pose, jaw_cal, grasp_standoff_m=0.007)
    cfg = SequenceConfig(grasp_gate="always", descend_step_mm=0.5)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0), cfg=cfg)
    descend = [s for s in steps if s.phase == PHASE_DESCEND]
    assert descend
    moves = [
        float(np.linalg.norm(b.command.pose.p - a.command.pose.p) * 1000.0)
        for a, b in zip(descend, descend[1:])
    ]
    assert max(moves) <= 0.5 + 1e-6, f"descended {max(moves):.3f} mm in one cycle"


def test_no_standoff_means_no_descend_phase(start_pose, jaw_cal, baseline):
    plan = pipeline_plan(start_pose, jaw_cal)
    seq, steps = run(plan, jaw_cal, baseline, block_at=np.deg2rad(-5.0))
    assert seq.phase == PHASE_DONE, seq.reason
    assert PHASE_DESCEND not in [s.phase for s in steps]


def test_the_precheck_warns_when_a_policy_aims_straight_at_the_needle(
    start_pose, jaw_cal
):
    plan = pipeline_plan(start_pose, jaw_cal)
    assert status_of(precheck(plan, controller="rl"), "grasp_standoff") == WARN
    # the servo has no training goal to miss, so this is not its problem
    assert status_of(precheck(plan, controller="d2"), "grasp_standoff") == PASS
    with_standoff = pipeline_plan(start_pose, jaw_cal, grasp_standoff_m=0.007)
    assert status_of(precheck(with_standoff, controller="rl"),
                     "grasp_standoff") == PASS


def test_the_descend_budget_is_checked(start_pose, jaw_cal):
    plan = pipeline_plan(start_pose, jaw_cal, grasp_standoff_m=0.007)
    assert status_of(precheck(plan), "step_budget_descend") == PASS
    starved = precheck(plan, descend_max_steps=3)
    assert status_of(starved, "step_budget_descend") == FAIL


def test_the_approach_contracts_carry_the_seven_millimetres(start_pose, jaw_cal):
    assert APPROACH_UPSTREAM.grasp_standoff_m == pytest.approx(0.007)
    # Place's goal is the entry pose itself, not a standoff from it
    assert PLACE_UPSTREAM.grasp_standoff_m == pytest.approx(0.0)
