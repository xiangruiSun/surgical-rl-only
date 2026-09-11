"""Measured closed-loop state machine for the Approach grasp task."""

from __future__ import annotations

import numpy as np


def wrap_to_pi(value):
    value = np.asarray(value, dtype=np.float64)
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class MeasuredGraspServo:
    """Approach, close, confirm, then stop using measured state and contact.

    The controller deliberately does not treat a closed jaw as a grasp.  The
    confirmation phase requires ``needle_grasped`` from the PSM finger-sensor
    actuator path for multiple consecutive control cycles.
    """

    PHASE_APPROACH = "approach"
    PHASE_CLOSE = "close_jaw"
    PHASE_CONFIRM = "confirm_grasp"
    PHASE_DONE = "done"

    def __init__(
        self,
        step_size,
        translation_threshold_cm=1.0,
        rotation_threshold_rad=np.deg2rad(10.0),
        approach_dwell_steps=3,
        confirmation_steps=3,
        open_jaw=0.8,
        closed_jaw=0.0,
        closed_jaw_threshold=0.1,
    ):
        self.step_size = np.asarray(step_size, dtype=np.float64)
        if self.step_size.shape != (7,) or np.any(self.step_size <= 0.0):
            raise ValueError("step_size must be a positive seven-vector")
        self.translation_threshold_cm = float(translation_threshold_cm)
        self.rotation_threshold_rad = float(rotation_threshold_rad)
        self.approach_dwell_steps = int(approach_dwell_steps)
        self.confirmation_steps = int(confirmation_steps)
        if self.approach_dwell_steps <= 0 or self.confirmation_steps <= 0:
            raise ValueError("servo dwell counts must be positive")
        self.open_jaw = float(open_jaw)
        self.closed_jaw = float(closed_jaw)
        self.closed_jaw_threshold = float(closed_jaw_threshold)
        self.reset()

    def reset(self):
        self.phase = self.PHASE_APPROACH
        self.approach_streak = 0
        self.confirmation_streak = 0
        self.last_diagnostic = None

    def _pose_action(self, measured, desired):
        delta = np.asarray(desired, dtype=np.float64) - np.asarray(
            measured, dtype=np.float64
        )
        delta[3:6] = wrap_to_pi(delta[3:6])
        delta_env = delta.copy()
        delta_env[:3] /= 100.0
        action = np.clip(delta_env / self.step_size, -1.0, 1.0)
        trans_error_cm = float(np.linalg.norm(delta[:3]))
        # The environment's corrected measured contract uses SO(3) geodesic
        # for scoring.  The servo uses the wrapped Euler tangent magnitude as a
        # local gate; it is valid here because the gate is only 10 degrees.
        rot_error_rad = float(np.linalg.norm(delta[3:6]))
        return action, trans_error_cm, rot_error_rad

    def action(self, measured, desired, grasp_status):
        measured = np.asarray(measured, dtype=np.float64)
        desired = np.asarray(desired, dtype=np.float64)
        if measured.shape != (7,) or desired.shape != (7,):
            raise ValueError("measured and desired states must be seven-vectors")

        phase_before = self.phase
        pose_action, trans_error_cm, rot_error_rad = self._pose_action(
            measured, desired
        )
        pose_ready = bool(
            trans_error_cm <= self.translation_threshold_cm
            and rot_error_rad <= self.rotation_threshold_rad
        )
        needle_grasped = bool((grasp_status or {}).get("needle_grasped", False))
        jaw_closed = bool(measured[6] <= self.closed_jaw_threshold)

        if self.phase == self.PHASE_APPROACH:
            self.approach_streak = self.approach_streak + 1 if pose_ready else 0
            if self.approach_streak >= self.approach_dwell_steps:
                self.phase = self.PHASE_CLOSE

        # A real object can physically stop the measured jaw well above the
        # empty-jaw threshold.  ``needle_grasped`` is already conditioned on a
        # close command crossing the actuator threshold plus a Needle ghost-
        # sensor hit, so it is the authoritative close/attachment signal.
        if self.phase == self.PHASE_CLOSE and pose_ready and needle_grasped:
            self.phase = self.PHASE_CONFIRM
            self.confirmation_streak = 1
        elif self.phase == self.PHASE_CONFIRM:
            if pose_ready and needle_grasped:
                self.confirmation_streak += 1
                if self.confirmation_streak >= self.confirmation_steps:
                    self.phase = self.PHASE_DONE
            else:
                self.phase = self.PHASE_CLOSE
                self.confirmation_streak = 0

        action = pose_action
        jaw_target = self.open_jaw if self.phase == self.PHASE_APPROACH else self.closed_jaw
        action[6] = np.clip(
            (jaw_target - measured[6]) / self.step_size[6], -1.0, 1.0
        )
        if self.phase == self.PHASE_DONE:
            action[:] = 0.0

        self.last_diagnostic = {
            "phase_before": phase_before,
            "phase_after": self.phase,
            "translation_error_cm": trans_error_cm,
            "rotation_error_rad": rot_error_rad,
            "pose_ready": pose_ready,
            "jaw_measured": float(measured[6]),
            "jaw_closed": jaw_closed,
            "needle_grasped": needle_grasped,
            "approach_streak": int(self.approach_streak),
            "confirmation_streak": int(self.confirmation_streak),
            "stop_requested": self.phase == self.PHASE_DONE,
        }
        return np.asarray(action, dtype=np.float32), dict(self.last_diagnostic)
