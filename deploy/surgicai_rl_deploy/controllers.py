"""Action sources.

``RLController``  -- the released TD3+HER+BC checkpoint.
``D2Controller``  -- the gain-scheduled SE(3) goal servo ported verbatim from
                     ``src/control/controllers.py`` in the combined repo.  It
                     is the *validated* Reach path (SIM-S4 selects it), needs
                     no learned model, and is the honest baseline to compare
                     the policy against on real hardware.
``ResidualController`` -- policy + servo blend with the direction guard, also
                     ported from the combined repo.

All of them take the scaled observation dict and return an action in [-1, 1]^7.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import STEP_SIZE_RAW


def _wrap_to_pi(value):
    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _validated(obs: dict, step_size):
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64).reshape(7)
    desired = np.asarray(obs["desired_goal"], dtype=np.float64).reshape(7)
    steps = np.asarray(step_size, dtype=np.float64).reshape(7)
    if np.any(steps <= 0.0):
        raise ValueError("step_size must be positive")
    return achieved, desired, steps


def _rotation_action(current_rpy, desired_rpy, rotation_steps, *, gain, cap_fraction):
    current = Rotation.from_euler("xyz", current_rpy)
    desired = Rotation.from_euler("xyz", desired_rpy)
    rotvec = (current.inv() * desired).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1.0e-12:
        return np.zeros(3), 0.0
    max_geodesic_step = float(np.min(rotation_steps) * cap_fraction)
    commanded = min(max_geodesic_step, gain * angle)
    next_rot = current * Rotation.from_rotvec(rotvec * (commanded / angle))
    increment = _wrap_to_pi(next_rot.as_euler("xyz") - np.asarray(current_rpy))
    return np.clip(increment / rotation_steps, -cap_fraction, cap_fraction), angle


def adaptive_se3_action(obs: dict, step_size=STEP_SIZE_RAW, *, staged: bool = False):
    achieved, desired, steps = _validated(obs, step_size)

    # NOTE: observation xyz is in cm, step_size xyz is in metres.
    translation_error_m = (desired[:3] - achieved[:3]) / 100.0
    translation_norm_m = float(np.linalg.norm(translation_error_m))

    if translation_norm_m > 0.010:
        trans_gain, trans_cap = 0.85, 1.00
    elif translation_norm_m > 0.003:
        trans_gain, trans_cap = 0.65, 0.70
    elif translation_norm_m > 0.001:
        trans_gain, trans_cap = 0.50, 0.40
    else:
        trans_gain, trans_cap = 0.35, 0.20

    current = Rotation.from_euler("xyz", achieved[3:6])
    desired_rot = Rotation.from_euler("xyz", desired[3:6])
    rot_err = float(np.linalg.norm((current.inv() * desired_rot).as_rotvec()))

    if rot_err > np.deg2rad(30.0):
        rot_gain, rot_cap = 0.80, 1.00
    elif rot_err > np.deg2rad(10.0):
        rot_gain, rot_cap = 0.65, 0.75
    elif rot_err > np.deg2rad(3.0):
        rot_gain, rot_cap = 0.50, 0.45
    else:
        rot_gain, rot_cap = 0.35, 0.25

    translation_action = np.clip(
        trans_gain * translation_error_m / steps[:3], -trans_cap, trans_cap
    )
    if staged and rot_err > np.deg2rad(15.0):
        translation_action *= 0.20
    elif staged and rot_err > np.deg2rad(8.0):
        translation_action *= 0.55

    rotation_action, _ = _rotation_action(
        achieved[3:6], desired[3:6], steps[3:6], gain=rot_gain, cap_fraction=rot_cap
    )
    jaw_action = float(np.clip((desired[6] - achieved[6]) / steps[6], -1.0, 1.0))
    return np.asarray([*translation_action, *rotation_action, jaw_action], dtype=np.float32)


class D2Controller:
    name = "d2"

    def __init__(self, staged: bool = True, step_size=STEP_SIZE_RAW):
        self.staged = bool(staged)
        self.step_size = np.asarray(step_size, dtype=np.float32)

    def act(self, obs: dict) -> np.ndarray:
        return adaptive_se3_action(obs, self.step_size, staged=self.staged)

    def describe(self) -> str:
        return f"D2 SE(3) goal servo (staged={self.staged})"


class RLController:
    name = "rl"

    def __init__(self, policy):
        self.policy = policy

    def act(self, obs: dict) -> np.ndarray:
        return self.policy.act(obs)

    def describe(self) -> str:
        return f"RL checkpoint {self.policy.describe()}"


class ResidualController:
    """Policy action blended with the servo, with the opposing-axis guard."""

    name = "residual"

    def __init__(
        self,
        policy,
        step_size=STEP_SIZE_RAW,
        policy_weight: float = 0.50,
        servo_weight: float = 0.75,
        staged: bool = False,
        direction_guard: bool = True,
    ):
        self.policy = policy
        self.step_size = np.asarray(step_size, dtype=np.float32)
        self.policy_weight = float(policy_weight)
        self.servo_weight = float(servo_weight)
        self.staged = bool(staged)
        self.direction_guard = bool(direction_guard)

    def act(self, obs: dict) -> np.ndarray:
        policy = np.asarray(self.policy.act(obs), dtype=np.float32).reshape(7)
        servo = adaptive_se3_action(obs, self.step_size, staged=self.staged)
        guarded = policy.copy()
        if self.direction_guard:
            opposing = (np.abs(servo) > 0.05) & (guarded * servo < 0.0)
            guarded[opposing] = 0.0
        combined = self.policy_weight * guarded + self.servo_weight * servo
        return np.clip(combined, -1.0, 1.0).astype(np.float32)

    def describe(self) -> str:
        return (
            f"residual: {self.policy_weight}*policy + {self.servo_weight}*servo "
            f"({self.policy.describe()})"
        )
