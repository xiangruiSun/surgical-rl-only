"""A synthetic robot and camera, so the protocol can be rehearsed before it runs.

Nothing here is a model of the residual field.  The residual *emerges*: a needle
is placed, a true hand-eye transform carries it into the ECM frame, a jaw offset
and a kinematic error move the grasp pose, a deliberately-wrong hand-eye
transform and a biased pose estimator produce the nominal point, and
``r = p_grasp - p_nom`` falls out.  That is the only honest way to test a
calibration pipeline: if the residual were imposed as a polynomial, the fit
recovering it would prove nothing except that least squares works.

What it is for
--------------
1. **Testing.**  Every claim in this package is checkable against a world whose
   truth is known.  The affine claim, the flip detector, the degree selector,
   the refusal outside the box.

2. **Power analysis, before anyone stands at the robot.**  Given a repeatability
   and a needle envelope, how many placements does it take before the ladder
   reliably picks the right rung?  ``tools/rehearse_grasp_calibration.py``
   answers that by running the whole protocol a few hundred times.  It is a much
   cheaper way to discover that sixty placements cannot support a degree-3
   tensor model than collecting sixty placements.

3. **A negative control.**  Build a world whose residual is exactly affine --
   which section 2 of the protocol document argues is the physically expected
   case -- and check that the selector does *not* adopt a curved model.  A
   selection procedure that cannot say "no" is not a selection procedure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .dataset import CalibrationDataset, Placement
from .perception import (
    DEFAULT_T_EC,
    GraspPointSpec,
    HandEye,
    needle_frame_N,
)


def _rigid(rotvec, translation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


@dataclass
class SyntheticWorld:
    """A robot, a camera and an estimator, all with known faults.

    Defaults are chosen to match what the hardware literature reports rather
    than to make the method look good:

    * ``handeye_rotation_deg = 1.0`` -- the JHU registration package's README
      puts its own accuracy at "closer to 5mm to 10mm cube"; at an 8 cm working
      distance one degree is 1.4 mm, comfortably inside that.
    * ``jaw_offset_m`` 5 mm along the tool z -- the Large Needle Driver's
      pitch-to-yaw link alone is 9 mm, so this is conservative.
    * ``fp_noise_sd_m`` 0.4 mm per axis -- the markerless needle-tracking
      literature reports 0.6-1.2 mm total position error for a suture needle.
    * ``grasp_noise_sd_m`` 0.3 mm -- a hand-taught target, read off
      ``measured_cp``, on an arm whose own ``measured_cp`` noise this project
      has already measured at a few tenths of a millimetre.
    """

    # -- the true and the believed hand-eye transform ----------------------
    handeye_nominal: np.ndarray = field(default_factory=lambda: DEFAULT_T_EC.copy())
    handeye_rotation_deg: float = 1.0
    handeye_rotation_axis: tuple = (0.3, -0.7, 0.65)
    handeye_translation_m: tuple = (0.002, -0.0015, 0.001)

    # -- the gripper -------------------------------------------------------
    #: offset from the frame measured_cp reports to the point between the jaws,
    #: expressed in the TOOL frame.  Constant there, which is why section 8
    #: freezes the wrist.
    jaw_offset_tool_m: tuple = (0.0, 0.0, 0.005)
    #: fixed gripper orientation for the whole session (section 8)
    gripper_rotvec: tuple = (2.2, 0.4, -0.3)
    #: how far the wrist is allowed to wander between placements, degrees
    gripper_wander_deg: float = 0.0

    # -- the pose estimator ------------------------------------------------
    #: constant bias in the camera frame, metres
    fp_bias_m: tuple = (0.0005, -0.0003, 0.0012)
    #: fractional depth-scale error: z is estimated (1 + s) times too long
    fp_depth_scale: float = 0.004
    #: an optional extra bias field, camera frame -> metres.  This is the only
    #: place curvature can enter, and it is switched off by default because
    #: nothing in the physics predicts it.
    fp_extra_bias: Optional[Callable[[np.ndarray], np.ndarray]] = None
    fp_noise_sd_m: float = 0.0004
    #: probability that a frame lands on a flipped branch
    fp_flip_rate: float = 0.0
    fp_flip_deg: float = 170.0

    # -- the arm -----------------------------------------------------------
    #: a joint-offset-like error, applied as a small rotation of the ECM frame
    #: about the robot's own base.  Exactly affine, as section 2 derives.
    arm_rotation_deg: float = 0.1
    arm_rotation_axis: tuple = (0.1, 0.9, -0.4)
    grasp_noise_sd_m: float = 0.0003

    # -- the scene ---------------------------------------------------------
    #: where needles are placed, camera frame: centre and half-extent, metres
    needle_centre_C: tuple = (0.0, 0.0, 0.08)
    needle_half_extent_m: tuple = (0.02, 0.02, 0.015)
    needle_yaw_envelope_deg: float = 30.0

    def __post_init__(self):
        self.handeye_nominal = np.asarray(self.handeye_nominal, dtype=np.float64).reshape(4, 4)

    # -- derived -----------------------------------------------------------
    @property
    def handeye_true(self) -> np.ndarray:
        axis = np.asarray(self.handeye_rotation_axis, dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        dT = _rigid(axis * np.deg2rad(self.handeye_rotation_deg), self.handeye_translation_m)
        return dT @ self.handeye_nominal

    @property
    def arm_error(self) -> np.ndarray:
        axis = np.asarray(self.arm_rotation_axis, dtype=np.float64)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return _rigid(axis * np.deg2rad(self.arm_rotation_deg), np.zeros(3))

    def nominal_hand_eye(self) -> HandEye:
        return HandEye(self.handeye_nominal, source="synthetic world, nominal")

    # -- one placement -----------------------------------------------------
    def place_needle(self, rng) -> np.ndarray:
        """A true needle pose in the camera frame."""
        centre = np.asarray(self.needle_centre_C, dtype=np.float64)
        half = np.asarray(self.needle_half_extent_m, dtype=np.float64)
        p = centre + rng.uniform(-half, half)
        yaw = np.deg2rad(rng.uniform(-self.needle_yaw_envelope_deg,
                                     self.needle_yaw_envelope_deg))
        # A needle lying roughly in the image plane, rolled to face the camera.
        R = Rotation.from_euler("zyx", [yaw, 0.02, np.pi]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def estimate(self, T_CN_true, rng) -> np.ndarray:
        """What FoundationPose reports for that needle."""
        T = T_CN_true.copy()
        p = T[:3, 3]
        bias = np.asarray(self.fp_bias_m, dtype=np.float64).copy()
        bias[2] += self.fp_depth_scale * p[2]
        if self.fp_extra_bias is not None:
            bias = bias + np.asarray(self.fp_extra_bias(p), dtype=np.float64).reshape(3)
        T[:3, 3] = p + bias + rng.normal(0.0, self.fp_noise_sd_m, 3)
        if self.fp_flip_rate > 0.0 and rng.random() < self.fp_flip_rate:
            flip = Rotation.from_rotvec(
                np.deg2rad(self.fp_flip_deg) * np.array([0.0, 0.0, 1.0])
            ).as_matrix()
            T[:3, :3] = T[:3, :3] @ flip
        return T

    def true_grasp_pose(self, T_CN_true, grasp_point: GraspPointSpec, rng) -> np.ndarray:
        """Where ``measured_cp`` reads when the jaw actually holds the needle.

        Built physically: the true grasp *point* is on the needle arc, carried
        into the ECM frame by the TRUE hand-eye transform, then displaced by the
        jaw offset expressed in the tool frame, then moved by the arm's own
        kinematic error, then read with noise.
        """
        T_EN = self.handeye_true @ (T_CN_true @ needle_frame_N(grasp_point.angle_deg))
        p_needle_E = T_EN[:3, 3]

        rot = np.asarray(self.gripper_rotvec, dtype=np.float64)
        if self.gripper_wander_deg > 0.0:
            rot = rot + rng.normal(0.0, np.deg2rad(self.gripper_wander_deg), 3)
        R_tool = Rotation.from_rotvec(rot).as_matrix()

        # measured_cp sits jaw_offset BEHIND the pinch point, in the tool frame.
        p_cp = p_needle_E - R_tool @ np.asarray(self.jaw_offset_tool_m, dtype=np.float64)
        p_cp = self.arm_error[:3, :3] @ p_cp + self.arm_error[:3, 3]
        p_cp = p_cp + rng.normal(0.0, self.grasp_noise_sd_m, 3)

        T = np.eye(4)
        T[:3, :3] = R_tool
        T[:3, 3] = p_cp
        return T

    def placement(
        self,
        placement_id: str,
        rng,
        grasp_point: GraspPointSpec,
        n_frames: int = 5,
        n_grasps: int = 1,
    ) -> Placement:
        T_true = self.place_needle(rng)
        return Placement(
            placement_id=placement_id,
            pose_estimates=[self.estimate(T_true, rng) for _ in range(n_frames)],
            grasp_poses=[
                self.true_grasp_pose(T_true, grasp_point, rng) for _ in range(n_grasps)
            ],
            verified=True,
            note="synthetic",
        )


def make_dataset(
    world: SyntheticWorld,
    n_placements: int = 40,
    seed: int = 0,
    grasp_point: Optional[GraspPointSpec] = None,
    n_frames: int = 5,
    n_grasps: int = 1,
) -> CalibrationDataset:
    """Run the section 5 protocol against a synthetic world."""
    rng = np.random.default_rng(seed)
    gp = grasp_point or GraspPointSpec()
    ds = CalibrationDataset(
        hand_eye=world.nominal_hand_eye(),
        grasp_point=gp,
        provenance={
            "source": "synthetic",
            "seed": seed,
            "handeye_rotation_deg": world.handeye_rotation_deg,
            "fp_noise_sd_mm": world.fp_noise_sd_m * 1000.0,
            "grasp_noise_sd_mm": world.grasp_noise_sd_m * 1000.0,
            "fp_flip_rate": world.fp_flip_rate,
            "curved_bias": world.fp_extra_bias is not None,
        },
    )
    for i in range(int(n_placements)):
        ds.add(world.placement(f"p{i:03d}", rng, gp, n_frames, n_grasps))
    return ds


# ---------------------------------------------------------------------------
# a curved bias, for the positive control
# ---------------------------------------------------------------------------
def quadratic_depth_bias(amplitude_m: float = 0.0008, reference_m: float = 0.08):
    """A depth bias that grows as the square of range.

    Stands in for whatever a learned pose estimator might do that is genuinely
    not affine.  Used as the positive control: if the selector cannot find
    *this*, it will not find anything.
    """

    def bias(p_C):
        z = float(np.asarray(p_C).reshape(3)[2])
        return np.array([0.0, 0.0, amplitude_m * (z / reference_m) ** 2])

    return bias


def saddle_bias(amplitude_m: float = 0.001, scale_m: float = 0.02):
    """A saddle in the image plane -- curvature a polynomial can actually see."""

    def bias(p_C):
        p = np.asarray(p_C).reshape(3)
        s = float(scale_m)
        return np.array(
            [
                amplitude_m * (p[0] * p[1]) / (s * s),
                amplitude_m * (p[0] ** 2 - p[1] ** 2) / (s * s),
                0.0,
            ]
        )

    return bias
