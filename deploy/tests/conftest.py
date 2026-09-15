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
