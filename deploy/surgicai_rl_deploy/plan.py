"""Geometry of a grasp-and-lift episode: the three poses the arm must hit.

    start  --(approach)-->  grasp  --(close jaw)-->  grasp  --(lift)-->  lifted

Everything here is pure geometry in the **robot frame** -- the frame
``<arm>/measured_cp`` reports, ``ECM`` on lcsr-dvrk-15.  No controller, no ROS,
no policy.  The frame bridge into the policy's training frame happens later and
only for the approach segment.

The lift
--------
The user's contract for this deployment is: *lift the needle 1.5 cm up, in z*.
"Up" is a claim about the world, and a frame's z axis is not guaranteed to
point away from the tissue -- on the ECM frame it usually points **into** the
scene, along the camera's view direction.  Getting the sign wrong drives the
gripper into the pad with the needle clamped in it.

So the sign is never guessed: :class:`LiftSpec` carries an ``explicit`` flag,
and :mod:`.feasibility` refuses to run a live episode unless the sign was
stated on the command line by a human who looked at the scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .frames import Pose, rotation_error_rad
from .jaw import JawCalibration

AXES = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class LiftSpec:
    """How far, along what, and in which frame the lift goes."""

    #: 'x' | 'y' | 'z'
    axis: str = "z"
    #: +1 or -1
    sign: int = 1
    #: metres; the deployment contract is 1.5 cm
    distance_m: float = 0.015
    #: 'robot' -- the frame measured_cp reports in (ECM here);
    #: 'tool'  -- the gripper frame at the grasp pose, so the needle backs out
    #:            along the axis the jaw came in on
    frame: str = "robot"
    #: True once a human has stated the sign explicitly.  Live runs require it.
    explicit: bool = False

    def __post_init__(self):
        if self.axis not in AXES:
            raise ValueError(f"lift axis must be one of x, y, z; got {self.axis!r}")
        if int(self.sign) not in (-1, 1):
            raise ValueError(f"lift sign must be +1 or -1; got {self.sign!r}")
        if not np.isfinite(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError("lift distance must be a positive, finite number of metres")
        if self.distance_m > 0.05:
            raise ValueError(
                f"lift distance {self.distance_m * 100:.1f} cm exceeds the 5 cm "
                "ceiling for this package; a longer retreat is a separate, "
                "planned motion, not a lift"
            )
        if self.frame not in ("robot", "tool"):
            raise ValueError(f"lift frame must be 'robot' or 'tool'; got {self.frame!r}")

    def direction(self, grasp: Pose) -> np.ndarray:
        """Unit vector of the lift, expressed in the robot frame."""
        unit = np.zeros(3)
        unit[AXES[self.axis]] = float(self.sign)
        if self.frame == "tool":
            return grasp.R @ unit
        return unit

    def displacement(self, grasp: Pose) -> np.ndarray:
        return self.direction(grasp) * float(self.distance_m)

    def describe(self) -> str:
        return (
            f"{self.distance_m * 100:.2f} cm along {'+' if self.sign > 0 else '-'}"
            f"{self.axis} of the {self.frame} frame"
            + ("" if self.explicit else "  [SIGN NOT CONFIRMED BY OPERATOR]")
        )


@dataclass
class GraspLiftPlan:
    """The frozen geometry of one episode."""

    start: Pose
    grasp: Pose
    lifted: Pose
    lift_spec: LiftSpec
    jaw: JawCalibration

    # -- derived -----------------------------------------------------------
    @property
    def approach_travel_cm(self) -> float:
        return float(np.linalg.norm(self.grasp.p - self.start.p) * 100.0)

    @property
    def approach_rotation_deg(self) -> float:
        return float(np.degrees(rotation_error_rad(self.start, self.grasp)))

    @property
    def lift_travel_cm(self) -> float:
        return float(np.linalg.norm(self.lifted.p - self.grasp.p) * 100.0)

    def path_radius_cm(self) -> float:
        """Furthest any waypoint sits from the measured start pose."""
        return float(
            max(
                np.linalg.norm(self.grasp.p - self.start.p),
                np.linalg.norm(self.lifted.p - self.start.p),
            )
            * 100.0
        )

    def bounding_box_m(self, pad_cm: float = 2.0):
        """Axis-aligned box covering all three waypoints, plus padding."""
        pts = np.stack([self.start.p, self.grasp.p, self.lifted.p])
        pad = float(pad_cm) / 100.0
        return pts.min(axis=0) - pad, pts.max(axis=0) + pad

    def as_dict(self) -> dict:
        return {
            "start_cm": (self.start.p * 100.0).tolist(),
            "grasp_cm": (self.grasp.p * 100.0).tolist(),
            "lifted_cm": (self.lifted.p * 100.0).tolist(),
            "start_quat_xyzw": self.start.quat_xyzw().tolist(),
            "grasp_quat_xyzw": self.grasp.quat_xyzw().tolist(),
            "approach_travel_cm": self.approach_travel_cm,
            "approach_rotation_deg": self.approach_rotation_deg,
            "lift_travel_cm": self.lift_travel_cm,
            "lift": {
                "axis": self.lift_spec.axis,
                "sign": int(self.lift_spec.sign),
                "distance_cm": self.lift_spec.distance_m * 100.0,
                "frame": self.lift_spec.frame,
                "operator_confirmed_sign": bool(self.lift_spec.explicit),
                "direction_robot_frame": self.lift_spec.direction(self.grasp).tolist(),
            },
            "jaw": self.jaw.as_dict(),
        }


def resolve_grasp_rotation(
    start: Pose,
    mode: str,
    goal_quat_xyzw: Optional[tuple] = None,
) -> np.ndarray:
    """Pick the orientation the gripper holds at the grasp pose.

    ``hold``
        Keep the orientation the arm is already in.  Nothing rotates, so the
        wrist cannot tumble into a singularity on the way in -- and, per the
        R6 findings, this is also the case the policy never saw in training.
    ``explicit``
        Use ``--goal-quat``.  This is the right mode when a needle-grasp
        orientation comes from perception.
    ``trained_relative``
        Rotate the wrist by the mean start->goal rotation of the R6
        demonstrations (~65 deg), which is the only setting that puts the
        policy's rotation channels back in distribution.
    """
    if mode == "hold":
        return start.R.copy()
    if mode == "explicit":
        if goal_quat_xyzw is None:
            raise ValueError("goal_orientation='explicit' needs goal_quat_xyzw")
        return Pose.from_pos_quat([0.0, 0.0, 0.0], goal_quat_xyzw).R
    if mode == "trained_relative":
        from scipy.spatial.transform import Rotation

        from .contract import R6_START_TO_GOAL_ROTVEC_MEAN

        rel = Rotation.from_rotvec(R6_START_TO_GOAL_ROTVEC_MEAN).as_matrix()
        return start.R @ rel
    raise ValueError(f"unknown goal_orientation {mode!r}")


def build_plan(
    start: Pose,
    grasp_position_m,
    *,
    goal_orientation: str = "hold",
    goal_quat_xyzw: Optional[tuple] = None,
    lift: Optional[LiftSpec] = None,
    jaw: Optional[JawCalibration] = None,
) -> GraspLiftPlan:
    """Freeze the episode geometry from a measured start pose and a grasp point."""
    lift = lift or LiftSpec()
    jaw = jaw or JawCalibration()

    grasp_p = np.asarray(grasp_position_m, dtype=np.float64).reshape(3)
    if not np.isfinite(grasp_p).all():
        raise ValueError("grasp position must be finite")

    grasp_R = resolve_grasp_rotation(start, goal_orientation, goal_quat_xyzw)

    # The jaw rides along with each pose as the *normalised* value the policy
    # observation expects.  The real squeeze command is a separate channel and
    # is applied by the sequencer, not carried here.
    grasp_pose = Pose(grasp_p, grasp_R, jaw.normalise(jaw.approach_open_rad))
    lifted_pose = Pose(
        grasp_p + lift.displacement(Pose(grasp_p, grasp_R, 0.0)),
        grasp_R.copy(),
        jaw.normalise(jaw.grip_rad),
    )
    return GraspLiftPlan(
        start=start, grasp=grasp_pose, lifted=lifted_pose, lift_spec=lift, jaw=jaw
    )
