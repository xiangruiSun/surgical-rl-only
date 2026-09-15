"""The ROS entry point's argument surface, checked without ROS installed.

``rclpy`` and the message packages are stubbed so the module imports on any
host.  What is exercised is the wiring: defaults that matter for safety, and
the fact that every knob the README documents actually exists.
"""

import sys
import types

import numpy as np
import pytest


@pytest.fixture(scope="module")
def node_module():
    stubs = {}

    def stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        stubs[name] = module
        return module

    class _Node:
        def __init__(self, *a, **k):
            pass

    class _QoS:
        def __init__(self, depth=10):
            self.depth = depth
            self.reliability = None

    rclpy = stub("rclpy", init=lambda *a, **k: None, shutdown=lambda: None,
                 create_node=lambda *a, **k: None, spin_once=lambda *a, **k: None,
                 ok=lambda: True)
    rclpy.logging = types.SimpleNamespace(get_logger=lambda name: None)
    stub("rclpy.node", Node=_Node)
    stub("rclpy.qos", QoSProfile=_QoS,
         ReliabilityPolicy=types.SimpleNamespace(RELIABLE="reliable"))
    stub("geometry_msgs", )
    stub("geometry_msgs.msg", PoseStamped=object)
    stub("sensor_msgs", )
    stub("sensor_msgs.msg", JointState=object)
    stub("std_msgs", )
    stub("std_msgs.msg", Bool=object)

    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    sys.modules.pop("surgicai_rl_deploy.grasp_lift_node", None)
    from surgicai_rl_deploy import grasp_lift_node as module

    yield module

    sys.modules.pop("surgicai_rl_deploy.grasp_lift_node", None)
    for key, value in saved.items():
        if value is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = value


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


def test_jaw_calibration_flags_build_a_valid_calibration(node_module):
    from surgicai_rl_deploy.jaw import JawCalibration

    args = node_module.parse_args(["--grasp-pos", "0", "0", "0"])
    JawCalibration(
        open_rad=float(np.deg2rad(args.jaw_open_deg)),
        closed_rad=float(np.deg2rad(args.jaw_closed_deg)),
        grip_rad=float(np.deg2rad(args.jaw_grip_deg)),
        approach_open_rad=float(np.deg2rad(args.jaw_approach_open_deg)),
    )
