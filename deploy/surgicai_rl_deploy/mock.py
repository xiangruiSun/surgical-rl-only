"""A kinematic mock of the arm, including a jaw that can be blocked.

Used by ``tools/offline_grasp_lift.py`` and by the tests.  The point is not
fidelity -- it is that the *optimistic* case can be replayed deterministically:
if the sequencer cannot finish here, it certainly will not on hardware.

The jaw model is the interesting part.  ``block_at_rad`` is where the fingers
physically stop because something is between them; ``None`` means an empty
gripper that closes all the way to the command.  Setting ``drop_at_step``
releases the object mid-lift so the slip path can be exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .frames import Pose
from .sequence import ArmState


@dataclass
class MockJaw:
    """A first-order jaw with an optional physical stop."""

    angle_rad: float
    #: the fingers cannot close past this because an object is in the way
    block_at_rad: Optional[float] = None
    #: fraction of the remaining error closed per cycle
    follow: float = 0.6
    #: effort drawn once the jaw is pressing on its stop, per radian of overrun
    effort_gain: float = 4.0
    #: effort drawn by an unobstructed jaw (tendon friction)
    effort_floor: float = 0.02
    #: measurement noise, radians
    noise_rad: float = 0.0
    #: cycle index at which the object slips out of the jaws
    drop_at_step: Optional[int] = None

    _step: int = 0
    _dropped: bool = False

    def update(self, command_rad: float, rng=None):
        self._step += 1
        if (
            self.drop_at_step is not None
            and self._step >= self.drop_at_step
            and not self._dropped
        ):
            self._dropped = True
            self.block_at_rad = None

        target = float(command_rad)
        if self.block_at_rad is not None:
            target = max(target, float(self.block_at_rad))
        self.angle_rad += (target - self.angle_rad) * self.follow

        overrun = 0.0
        if self.block_at_rad is not None:
            overrun = max(0.0, float(self.block_at_rad) - float(command_rad))
        effort = self.effort_floor + self.effort_gain * overrun

        measured = self.angle_rad
        if self.noise_rad and rng is not None:
            measured = measured + rng.normal(0.0, self.noise_rad)
        return float(measured), float(effort)


class MockArm:
    """Cartesian pose that chases the command, plus a :class:`MockJaw`."""

    def __init__(self, pose: Pose, jaw: MockJaw, *, lag: float = 0.0,
                 noise_mm: float = 0.0, seed: int = 0,
                 jaw_calibration=None):
        self.pose = pose
        self.jaw = jaw
        self.lag = float(lag)
        self.noise_mm = float(noise_mm)
        self.rng = np.random.default_rng(seed)
        self.jaw_calibration = jaw_calibration

    def state(self) -> ArmState:
        jaw_norm = (
            0.0 if self.jaw_calibration is None
            else self.jaw_calibration.normalise(self.jaw.angle_rad)
        )
        measured_p = self.pose.p
        if self.noise_mm:
            measured_p = measured_p + self.rng.normal(0.0, self.noise_mm / 1000.0, 3)
        return ArmState(
            pose=Pose(measured_p, self.pose.R, jaw_norm),
            jaw_rad=self._last_jaw_measured,
            jaw_effort=self._last_jaw_effort,
        )

    _last_jaw_measured: Optional[float] = None
    _last_jaw_effort: Optional[float] = None

    def apply(self, command):
        """Advance one cycle towards the commanded pose and jaw."""
        if command.publish_pose:
            target = command.pose
            p = self.pose.p + (target.p - self.pose.p) * (1.0 - self.lag)
            rel = Rotation.from_matrix(self.pose.R.T @ target.R).as_rotvec() * (
                1.0 - self.lag
            )
            R = self.pose.R @ Rotation.from_rotvec(rel).as_matrix()
            self.pose = Pose(p, R, self.pose.jaw)
        if command.publish_jaw:
            measured, effort = self.jaw.update(command.jaw_rad, self.rng)
            self._last_jaw_measured = measured
            self._last_jaw_effort = effort

    def prime_jaw(self):
        """Seed the jaw feedback before the first command is issued."""
        self._last_jaw_measured = self.jaw.angle_rad
        self._last_jaw_effort = self.jaw.effort_floor
