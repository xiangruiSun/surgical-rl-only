"""The ROS entry point's argument surface, checked without ROS installed.

``rclpy`` and the message packages are stubbed so the module imports on any
host.  What is exercised is the wiring: defaults that matter for safety, and
the fact that every knob the README documents actually exists.
"""


import numpy as np
import pytest


def test_dry_run_is_the_default(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.execute is False


def test_manual_gate_is_the_default(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.grasp_gate == "manual"


def test_lift_sign_has_no_default(node_module):
    """It must be stated, so the precheck can refuse a live run without it."""
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.lift_sign is None


def test_lift_defaults_to_15mm_in_z(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.lift_distance_cm == pytest.approx(1.5)
    assert args.lift_axis == "z"
    assert args.lift_frame == "robot"


def test_the_servo_is_the_default_controller(node_module):
    """The RL policy never converged on this geometry; it is opt-in."""
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.controller == "d2"


def test_grip_default_is_a_squeeze_inside_the_dvrk_range(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.jaw_grip_deg < 0
    assert args.jaw_grip_deg >= -25.0  # dvrk's own close() is -20


def test_abort_policy_defaults_to_stopping(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.on_slip == "abort"


def test_grasp_position_is_required(node_module):
    with pytest.raises(SystemExit):
        node_module.parse_args([])


def test_lift_sign_rejects_nonsense(node_module):
    with pytest.raises(SystemExit):
        node_module.parse_args(["--grasp-pos", "0", "0", "0", "--lift-sign", "0"])


def test_safety_caps_match_the_approach_entry_point(node_module):
    """run_approach.py and run_grasp_lift.py must guard identically."""
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.max_step_translation_mm == pytest.approx(2.5)
    assert args.max_step_rotation_deg == pytest.approx(5.0)
    assert args.max_tracking_error_cm == pytest.approx(1.5)
    assert args.workspace_pad_cm == pytest.approx(2.0)


def test_lift_success_tolerance_is_tighter_than_the_lift(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.lift_success_trans_cm < args.lift_distance_cm


def test_subscriptions_default_to_best_effort(node_module):
    """A RELIABLE subscriber does not match a BEST_EFFORT publisher, which is
    how /PSM1/jaw/measured_js echoed fine on the command line while the node
    saw nothing. BEST_EFFORT matches publishers of either kind."""
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.sub_reliability == "best_effort"


def test_dry_runs_walk_the_sequence_by_default(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.dry_run_static is False
    assert args.dry_run_simulate is True


def test_dry_run_can_be_made_static(node_module):
    args = node_module.parse_args(
        ["--grasp-pos", "0", "0", "0", "--dry-run-static"]
    )
    assert args.dry_run_simulate is False


# --- the dry-run walkthrough ---------------------------------------------
def _build_node(node_module, plan, baseline, argv):
    from surgicai_rl_deploy.controllers import D2Controller
    from surgicai_rl_deploy.loop import SafetyLimits
    from surgicai_rl_deploy.sequence import SequenceConfig, GraspLiftSequencer

    args = node_module.parse_args(argv)
    sequencer = GraspLiftSequencer(
        plan, D2Controller(staged=True),
        SequenceConfig(grasp_gate="always"), node_module.build_limits(args),
        baseline,
    )
    node = node_module.GraspLiftNode(args, plan, sequencer, plan.jaw)
    node._measured = (plan.start.p, plan.start.quat_xyzw())
    node._measured_frame = "ECM"
    node._jaw_rad = plan.jaw.approach_open_rad
    node._jaw_effort = 0.02
    return node, sequencer


def test_simulated_dry_run_advances_past_the_start_pose(node_module, plan, baseline):
    """With nothing published the arm cannot move, so a static dry run sits at
    the start and always dies at max_steps. The simulated one must progress."""
    import time as _time

    node, sequencer = _build_node(
        node_module, plan, baseline, ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1"]
    )
    assert node.simulate is True
    node._measured_stamp = _time.monotonic()
    sequencer.begin(node._state())
    node._started = True

    for _ in range(40):
        node._measured_stamp = _time.monotonic()
        node.tick()
        if node.finished:
            break

    moved_cm = float(
        np.linalg.norm(node._sim_pose.p - plan.start.p) * 100.0
    )
    assert moved_cm > 0.5, "the simulated arm never left the start pose"


def test_a_static_dry_run_never_moves(node_module, plan, baseline):
    import time as _time

    node, sequencer = _build_node(
        node_module, plan, baseline,
        ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1", "--dry-run-static"],
    )
    assert node.simulate is False
    node._measured_stamp = _time.monotonic()
    sequencer.begin(node._state())
    node._started = True
    for _ in range(10):
        node._measured_stamp = _time.monotonic()
        node.tick()
    assert node._sim_pose is None
    assert sequencer.phase == "approach"


def test_a_dry_run_publishes_nothing(node_module, plan, baseline):
    import time as _time

    node, sequencer = _build_node(
        node_module, plan, baseline, ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1"]
    )
    node._measured_stamp = _time.monotonic()
    sequencer.begin(node._state())
    node._started = True
    for _ in range(30):
        node._measured_stamp = _time.monotonic()
        node.tick()
    assert node.published == []


def test_a_stale_pose_still_aborts_a_live_run(node_module, plan, baseline):
    """Relaxing the freshness guards for dry runs must not relax them for
    --execute: that guard is the one that stops motion against a dead feed."""
    import time as _time

    node, sequencer = _build_node(
        node_module, plan, baseline,
        ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1", "--execute"],
    )
    assert node.simulate is False  # never simulate while publishing
    node._measured_stamp = _time.monotonic()
    node._jaw_stamp = _time.monotonic()
    sequencer.begin(node._state())
    node._started = True

    node._measured_stamp = _time.monotonic() - 10.0  # feed went dead
    node.tick()
    assert node.finished
    assert "stale measured_cp" in sequencer.reason or node.finished


def test_a_stale_jaw_still_aborts_a_live_run(node_module, plan, baseline):
    import time as _time

    node, sequencer = _build_node(
        node_module, plan, baseline,
        ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1", "--execute"],
    )
    node._measured_stamp = _time.monotonic()
    node._jaw_stamp = _time.monotonic() - 10.0
    sequencer.begin(node._state())
    node._started = True
    node.tick()
    assert node.finished


def test_a_dry_run_tolerates_a_silent_jaw_topic(node_module, plan, baseline):
    """A dry run should still walk the sequence and say the feed is silent."""
    import time as _time

    node, sequencer = _build_node(
        node_module, plan, baseline, ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1"]
    )
    node._measured_stamp = _time.monotonic()
    node._jaw_stamp = _time.monotonic() - 10.0  # silent
    sequencer.begin(node._state())
    node._started = True
    for _ in range(20):
        node._measured_stamp = _time.monotonic()
        node.tick()
        if node.finished:
            break
    assert node._sim_pose is not None
    assert any("stale" in msg for _, msg in node.get_logger().lines)


def test_jaw_calibration_flags_build_a_valid_calibration(node_module):
    from surgicai_rl_deploy.jaw import JawCalibration

    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    JawCalibration(
        open_rad=float(np.deg2rad(args.jaw_open_deg)),
        closed_rad=float(np.deg2rad(args.jaw_closed_deg)),
        grip_rad=float(np.deg2rad(args.jaw_grip_deg)),
        approach_open_rad=float(np.deg2rad(args.jaw_approach_open_deg)),
    )


def test_r6_support_is_a_warning_only_for_the_learned_policy(
    node_module, plan, baseline
):
    """A WARN on a live arm should mean "consider stopping". A checkpoint's
    trained region does not bind the geometric servo, so under d2 it is logged
    as information, matching what the precheck already says."""
    node, sequencer = _build_node(
        node_module, plan, baseline, ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1"]
    )
    node._measured_stamp = __import__("time").monotonic()
    node.start_episode(_FakeReport())
    levels = {
        level for level, msg in node.get_logger().lines
        if "demonstration support" in msg
    }
    assert levels == {"info"}


def test_r6_support_warns_under_the_rl_controller(node_module, plan, baseline):
    from surgicai_rl_deploy.controllers import RLController

    node, sequencer = _build_node(
        node_module, plan, baseline, ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1"]
    )

    class _Policy:
        def act(self, obs):
            return np.zeros(7)

        def describe(self):
            return "stub"

    sequencer.approach_controller = RLController(_Policy())
    node._measured_stamp = __import__("time").monotonic()
    node.start_episode(_FakeReport())
    levels = {
        level for level, msg in node.get_logger().lines
        if "demonstration support" in msg
    }
    assert levels == {"warn"}


class _FakeReport:
    """Stands in for feasibility.PrecheckReport in start_episode's trace line."""

    def as_dict(self):
        return {"ok": True, "strict": False, "checks": []}


# ----------------------------------------------------------------------
# the operator gate has to be visible
# ----------------------------------------------------------------------
def test_the_gate_prompt_is_reprinted(node_module, plan, baseline):
    """It used to print once and then scroll away under a 10 Hz status line."""
    import time as _time

    node, sequencer = _build_node(
        node_module, plan, baseline,
        ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1",
         "--confirm-reprompt-s", "0", "--dry-run-confirm-after-s", "0"],
    )
    node._measured_stamp = _time.monotonic()
    sequencer.begin(node._state())
    for _ in range(3):
        node._confirm_callback()
    prompts = [m for lvl, m in node.get_logger().lines if "waiting for you" in m]
    assert len(prompts) == 3, "the prompt must repeat while it waits"


def test_a_dry_run_releases_the_gate_so_the_walkthrough_finishes(
    node_module, plan, baseline
):
    node, sequencer = _build_node(
        node_module, plan, baseline,
        ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1",
         "--dry-run-confirm-after-s", "0.0001"],
    )
    sequencer.begin(node._state())
    assert node._confirm_callback() is None  # first call only prompts
    import time as _time

    _time.sleep(0.01)
    assert node._confirm_callback() is True
    assert any("DRY RUN: nobody answered" in m for _, m in node.get_logger().lines)


def test_a_live_run_never_releases_the_gate_by_itself(node_module, plan, baseline):
    node, sequencer = _build_node(
        node_module, plan, baseline,
        ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1", "--execute",
         "--dry-run-confirm-after-s", "0.0001"],
    )
    sequencer.begin(node._state())
    import time as _time

    for _ in range(5):
        _time.sleep(0.005)
        assert node._confirm_callback() is None, "a live gate must wait for a person"


def test_the_gate_still_answers_the_topic(node_module, plan, baseline):
    node, sequencer = _build_node(
        node_module, plan, baseline,
        ["--grasp-pos", "0", "0", "0", "--lift-sign", "-1", "--execute"],
    )
    sequencer.begin(node._state())
    node._confirm_callback()
    node._confirm = True
    assert node._confirm_callback() is True
    node._confirm = False
    assert node._confirm_callback() is False


def test_a_slow_servo_rate_is_refused(node_module):
    args = node_module.parse_args(
        ["--grasp-pos", "0", "0", "0", "--interface", "servo_cp", "--rate", "5"]
    )
    assert args.rate < args.min_servo_rate, (
        "5 Hz must be below the servo_cp floor; the two live runs that never "
        "moved the arm were at 2 Hz and 5 Hz"
    )


def test_the_default_rate_clears_the_servo_floor(node_module):
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.interface == "servo_cp"
    assert args.rate >= args.min_servo_rate


def test_a_discovery_timeout_exists_and_is_generous(node_module):
    """Publishing before DDS matching completes drops the messages silently."""
    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    assert args.discovery_timeout_s >= 2.0


def test_command_subscribers_reports_both_topics(node_module, plan, baseline):
    node, _ = _build_node(
        node_module, plan, baseline, ["--grasp-pos", "0", "0", "0"]
    )
    counts = node.command_subscribers()
    assert set(counts) == {"/PSM1/servo_cp", "/PSM1/jaw/servo_jp"}


@pytest.mark.parametrize('reply', ['yes', 'no', 'timeout'])
def test_terminal_confirmation_controls_full_pipeline(
    node_module, start_pose, jaw_cal, baseline, monkeypatch, reply
):
    """Drive real phase transitions and terminal callback against the mock arm."""
    import io
    from surgicai_rl_deploy.mock import MockArm, MockJaw
    from surgicai_rl_deploy.plan import LiftSpec, build_plan
    from surgicai_rl_deploy.sequence import PHASE_WAIT_OPERATOR

    p = build_plan(
        start_pose, [-0.050726357, 0.015332369, 0.049514053], jaw=jaw_cal,
        lift=LiftSpec(axis='z', sign=-1, distance_m=0.015, explicit=True),
        suture_position_m=[-0.040, 0.005, 0.040],
        suture_quat_xyzw=[0, 0, 0, 1],
    )
    node, seq = _build_node(node_module, p, baseline,
                           ['--grasp-pos', '0', '0', '0', '--execute'])
    seq.cfg.grasp_gate = 'manual'
    seq.cfg.operator_timeout_steps = 8
    seq.limits.max_path_radius_cm = 30
    arm = MockArm(p.start, MockJaw(angle_rad=jaw_cal.approach_open_rad),
                  jaw_calibration=jaw_cal)
    arm.prime_jaw()

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal(reply + '\n')
    monkeypatch.setattr(node_module.sys, 'stdin', terminal)
    calls = []

    def ready(*args):
        calls.append(seq.phase)
        assert seq.phase == PHASE_WAIT_OPERATOR
        assert seq.jaw_command_rad == pytest.approx(jaw_cal.grip_rad)
        # No motion into lift before the operator has answered.
        return ([terminal], [], []) if len(calls) >= 4 and reply != 'timeout' else ([], [], [])

    monkeypatch.setattr(node_module.select, 'select', ready)
    seq.confirm_callback = node._confirm_callback
    seq.begin(arm.state())
    phases = []
    for _ in range(3000):
        step = seq.step(arm.state())
        phases.append(step.phase)
        arm.apply(step.command)
        if step.done:
            break
    assert step.done
    assert len(calls) >= 4
    assert 'close' in phases and 'wait_operator' in phases
    assert any('Was the grasp successful?' in m for _, m in node.get_logger().lines)
    assert any('transport, then descend' in m for _, m in node.get_logger().lines)
    if reply == 'yes':
        assert seq.reason == 'success'
        assert 'lift' in phases and 'transport' in phases and 'place' in phases
        assert np.linalg.norm(arm.pose.p - seq.suture_target.p) <= seq.cfg.place_success_trans_cm / 100
    else:
        assert 'lift' not in phases and 'transport' not in phases
        expected = 'operator declined the lift' if reply == 'no' else 'operator confirmation timed out'
        assert seq.reason == expected
    assert arm.jaw.angle_rad == pytest.approx(jaw_cal.grip_rad)


def test_topic_confirmation_before_grasp_is_ignored(node_module, plan, baseline):
    from types import SimpleNamespace
    from surgicai_rl_deploy.sequence import PHASE_WAIT_OPERATOR
    node, seq = _build_node(node_module, plan, baseline, ['--grasp-pos', '0', '0', '0'])
    seq.begin(node._state())
    node._on_confirm(SimpleNamespace(data=True))
    assert node._confirm is None
    seq.phase = PHASE_WAIT_OPERATOR
    node._on_confirm(SimpleNamespace(data=True))
    assert node._confirm is True


def test_manual_live_run_requires_a_confirmation_channel(node_module, monkeypatch, capsys):
    monkeypatch.setattr(node_module.sys, 'stdin', None)
    assert node_module.main(['--grasp-pos', '0', '0', '0', '--execute']) == 2
    assert 'interactive terminal or --confirm-topic' in capsys.readouterr().err


@pytest.mark.parametrize('argv, expected', [
    ([], 1200),
    (['--rate', '5'], 600),
    (['--operator-timeout-s', '0'], 0),
    (['--operator-timeout-steps', '7'], 7),
])
def test_operator_wait_budget(node_module, argv, expected):
    args = node_module.parse_args(['--grasp-pos', '0', '0', '0'] + argv)
    assert args.operator_timeout_steps == expected
