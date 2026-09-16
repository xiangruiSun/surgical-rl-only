import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgicai_rl_deploy.frames import Pose  # noqa: E402
from surgicai_rl_deploy.jaw import JawBaseline, JawCalibration  # noqa: E402
from surgicai_rl_deploy.plan import LiftSpec, build_plan  # noqa: E402

#: the pose echoed from /PSM1/measured_cp on lcsr-dvrk-15, frame ECM
REAL_START_POS = [-0.05639860616831881, 0.03366166453830251, 0.024455994074878362]
REAL_START_QUAT = [
    0.23319925218484056,
    0.4267863636861243,
    -0.23588767438897446,
    0.841325450478807,
]
REAL_GOAL_POS = [-0.050726357, 0.015332369, 0.049514053]


@pytest.fixture
def jaw_cal():
    return JawCalibration()


@pytest.fixture
def start_pose(jaw_cal):
    return Pose.from_pos_quat(
        REAL_START_POS, REAL_START_QUAT, jaw_cal.normalise(jaw_cal.approach_open_rad)
    )


@pytest.fixture
def lift_down():
    return LiftSpec(axis="z", sign=-1, distance_m=0.015, frame="robot", explicit=True)


@pytest.fixture
def plan(start_pose, lift_down, jaw_cal):
    return build_plan(start_pose, REAL_GOAL_POS, lift=lift_down, jaw=jaw_cal)


@pytest.fixture
def baseline(jaw_cal):
    return JawBaseline(
        empty_close_rad=jaw_cal.grip_rad,
        empty_close_effort=0.02,
        empty_close_rad_noise=float(np.deg2rad(0.1)),
        empty_close_effort_noise=0.005,
        source="test fixture",
    )


# --- the ROS node, with rclpy stubbed so it imports on any host ---------
import types  # noqa: E402


@pytest.fixture(scope="session")
def node_module():
    stubs = {}

    def stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        stubs[name] = module
        return module

    class _Logger:
        def __init__(self):
            self.lines = []

        def info(self, msg):
            self.lines.append(("info", msg))

        def warn(self, msg):
            self.lines.append(("warn", msg))

        def error(self, msg):
            self.lines.append(("error", msg))

    class _Clock:
        def now(self):
            return types.SimpleNamespace(to_msg=lambda: None)

    class _Node:
        def __init__(self, *a, **k):
            self._logger = _Logger()
            self.published = []

        def create_subscription(self, *a, **k):
            return None

        def create_publisher(self, *a, **k):
            node = self

            class _Pub:
                #: tests can set this to rehearse "nobody is listening"
                subscription_count = 1

                def publish(self, msg):
                    node.published.append(msg)

                def get_subscription_count(self):
                    return self.subscription_count

            return _Pub()

        def create_timer(self, *a, **k):
            return None

        def get_logger(self):
            return self._logger

        def get_clock(self):
            return _Clock()

        def destroy_node(self):
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
         ReliabilityPolicy=types.SimpleNamespace(
             RELIABLE="reliable", BEST_EFFORT="best_effort"))
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
