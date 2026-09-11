"""Pose conversions and the frame-bridging strategies.

Rotation convention
-------------------
Training used ``PyKDL.Frame.M.GetRPY()`` and the controllers used
``scipy...Rotation.from_euler("xyz", rpy)``.  These agree: KDL's RPY is the
fixed-axis (extrinsic) x-y-z convention, i.e. ``R = Rz(yaw) Ry(pitch) Rx(roll)``,
which is exactly scipy's lowercase ``"xyz"``.  Everything below uses that.

Units
-----
``vec7`` is the RAW pose vector ``[x_m, y_m, z_m, roll, pitch, yaw, jaw_norm]``.
Scaling to the network's cm-based observation happens in :mod:`obs`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------
# basic conversions
# --------------------------------------------------------------------------
def rpy_from_matrix(mat: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(mat, dtype=np.float64)).as_euler("xyz")


def matrix_from_rpy(rpy) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_matrix()


def rpy_from_quat_xyzw(quat) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat, dtype=np.float64)).as_euler("xyz")


def quat_xyzw_from_rpy(rpy) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_quat()


def wrap_to_pi(value):
    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class Pose:
    """A rigid transform plus the scalar jaw value that rides along with it."""

    p: np.ndarray  # (3,) metres
    R: np.ndarray  # (3,3)
    jaw: float = 0.0

    # -- constructors ------------------------------------------------------
    @staticmethod
    def from_vec7(vec7) -> "Pose":
        v = np.asarray(vec7, dtype=np.float64).reshape(7)
        return Pose(v[:3].copy(), matrix_from_rpy(v[3:6]), float(v[6]))

    @staticmethod
    def from_pos_quat(pos, quat_xyzw, jaw: float = 0.0) -> "Pose":
        return Pose(
            np.asarray(pos, dtype=np.float64).reshape(3).copy(),
            Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix(),
            float(jaw),
        )

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.zeros(3), np.eye(3), 0.0)

    # -- accessors ---------------------------------------------------------
    def to_vec7(self) -> np.ndarray:
        return np.concatenate([self.p, rpy_from_matrix(self.R), [self.jaw]]).astype(
            np.float64
        )

    def quat_xyzw(self) -> np.ndarray:
        return Rotation.from_matrix(self.R).as_quat()

    # -- algebra -----------------------------------------------------------
    def __mul__(self, other: "Pose") -> "Pose":
        return Pose(self.R @ other.p + self.p, self.R @ other.R, other.jaw)

    def inverse(self) -> "Pose":
        Rt = self.R.T
        return Pose(-Rt @ self.p, Rt, self.jaw)

    def with_jaw(self, jaw: float) -> "Pose":
        return Pose(self.p.copy(), self.R.copy(), float(jaw))


def translation_error_cm(a: Pose, b: Pose) -> float:
    return float(np.linalg.norm(a.p - b.p) * 100.0)


def rotation_error_rad(a: Pose, b: Pose) -> float:
    rel = Rotation.from_matrix(a.R.T @ b.R)
    return float(rel.magnitude())


# --------------------------------------------------------------------------
# frame bridging
# --------------------------------------------------------------------------
class FrameBridge:
    """Maps between the robot's working frame and the policy's training frame.

    The policy was trained on absolute poses in the PSM base frame, around one
    frozen goal.  A real ECM-frame goal is nowhere near that goal, so feeding
    raw ECM numbers puts every absolute block of the observation outside the
    training support.  A bridge fixes that by choosing a rigid transform ``X``
    with ``policy_frame_pose = X * robot_frame_pose``.

    Modes
    -----
    ``rebase``
        ``X = T_trained_goal * inv(T_goal_robot)``.  The real goal is mapped
        exactly onto the checkpoint's own goal, so ``desired_goal`` is always
        the vector the policy was trained on, and the start pose keeps its true
        relative offset (rotated into the training frame).  This is the
        "relative servo" idea from the R6 report, §7.1.
    ``translate``
        Position-only version of ``rebase``: the origin is shifted so the goal
        position lands on the trained goal position, while the axes stay in
        robot coordinates.  Orientation is therefore left untouched and the
        approach *direction* keeps its robot-frame sense, which is usually
        worse, not better.  Kept as an A/B baseline.
    ``identity``
        No bridging.  Raw robot-frame numbers go straight into the network.
        Expect out-of-distribution behaviour; useful only as a baseline.
    """

    def __init__(self, X: Pose, mode: str):
        self.X = X
        self.X_inv = X.inverse()
        self.mode = mode

    # -- factories ---------------------------------------------------------
    @classmethod
    def identity(cls) -> "FrameBridge":
        return cls(Pose.identity(), "identity")

    @classmethod
    def rebase(cls, goal_robot: Pose, trained_goal: Pose) -> "FrameBridge":
        return cls(trained_goal * goal_robot.inverse(), "rebase")

    @classmethod
    def translate(cls, goal_robot: Pose, trained_goal: Pose) -> "FrameBridge":
        # Pure translation: keep the robot's axes, shift the origin so the goal
        # position lands on the trained goal position.
        X = Pose(trained_goal.p - goal_robot.p, np.eye(3), 0.0)
        return cls(X, "translate")

    @classmethod
    def build(cls, mode: str, goal_robot: Pose, trained_goal: Pose) -> "FrameBridge":
        mode = str(mode)
        if mode == "identity":
            return cls.identity()
        if mode == "rebase":
            return cls.rebase(goal_robot, trained_goal)
        if mode == "translate":
            return cls.translate(goal_robot, trained_goal)
        raise ValueError(f"unknown frame mode {mode!r}")

    # -- use ---------------------------------------------------------------
    def to_policy(self, pose_robot: Pose) -> Pose:
        return (self.X * pose_robot).with_jaw(pose_robot.jaw)

    def to_robot(self, pose_policy: Pose) -> Pose:
        return (self.X_inv * pose_policy).with_jaw(pose_policy.jaw)
