"""The runtime path: a FoundationPose estimate in, a grasp target out.

Section 14, with the refusals section 15 asks for made mandatory rather than
advisory::

    T_CN  ->  needle point  ->  ^E T_C  ->  p_nom  ->  + f(p_nom)  ->  p_target

What this returns and what it does not
--------------------------------------
A **position**.  The calibration is a position-only correction and section 8
freezes the wrist precisely so that it can be: the offset between the frame
``measured_cp`` reports and the point between the jaws is fixed in the *tool*
frame, so it is only fixed in ECM coordinates while the tool frame is.  The
orientation of the grasp therefore has to come from somewhere else -- the taught
orientation the calibration was collected at, which is what
:class:`GraspResolver` carries and checks against.

Handing this a needle and getting back a pose whose orientation came from the
needle estimate would silently leave the regime the calibration was fitted in.
So the orientation is an input, the resolver compares it against the taught one,
and it warns when they differ by enough to matter.

The failure that this module exists to prevent
----------------------------------------------
A polynomial evaluated outside the box it was fitted in does not degrade, it
diverges -- the higher the degree, the faster.  Inside the box the Bernstein
convex-hull property bounds the correction by the largest control coefficient;
outside, nothing bounds it.  So :meth:`GraspResolver.resolve` refuses by
default, and the refusal names the axis and the distance rather than saying
"out of range".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from ..feasibility import FAIL, PASS, WARN, PrecheckReport
from ..frames import Pose
from .models import DEFAULT_MAX_CORRECTION_MM, OutsideCalibratedRegion, ResidualModel
from .perception import (
    GraspPointSpec,
    HandEye,
    average_poses,
    convention_digest,
    flip_reasons,
    nominal_position,
)


@dataclass
class GraspTarget:
    """Where the gripper should go, and every intermediate number that got there.

    Carrying the intermediates is not verbosity.  When a grasp misses, the only
    useful question is *which stage was wrong* -- the pose estimate, the
    hand-eye transform, or the correction -- and that question is unanswerable
    from the final number alone.  Every one of these lands in the trace.
    """

    #: the raw FoundationPose estimate used (averaged, if several frames)
    T_CN: np.ndarray
    #: the needle grasp point in the camera frame
    p_camera_m: np.ndarray
    #: after the hand-eye transform, before any correction
    p_nominal_m: np.ndarray
    #: the learned correction
    correction_m: np.ndarray
    #: nominal + correction
    p_target_m: np.ndarray
    #: the full commanded pose, with the taught orientation
    pose: Pose
    n_frames: int = 1
    report: PrecheckReport = field(default_factory=PrecheckReport)

    @property
    def correction_mm(self) -> np.ndarray:
        return self.correction_m * 1000.0

    @property
    def ok(self) -> bool:
        return self.report.ok

    def describe(self) -> str:
        c = self.correction_mm
        return (
            f"needle at camera ({self.p_camera_m[0]*100:+.2f}, "
            f"{self.p_camera_m[1]*100:+.2f}, {self.p_camera_m[2]*100:+.2f}) cm"
            f"  ->  nominal ECM ({self.p_nominal_m[0]*100:+.2f}, "
            f"{self.p_nominal_m[1]*100:+.2f}, {self.p_nominal_m[2]*100:+.2f}) cm"
            f"  ->  correction ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}) mm "
            f"[{np.linalg.norm(c):.2f} mm]"
            f"  ->  target ({self.p_target_m[0]*100:+.2f}, "
            f"{self.p_target_m[1]*100:+.2f}, {self.p_target_m[2]*100:+.2f}) cm"
        )

    def as_dict(self) -> dict:
        return {
            "T_CN": np.asarray(self.T_CN).tolist(),
            "p_camera_cm": (self.p_camera_m * 100.0).tolist(),
            "p_nominal_cm": (self.p_nominal_m * 100.0).tolist(),
            "correction_mm": self.correction_mm.tolist(),
            "p_target_cm": (self.p_target_m * 100.0).tolist(),
            "target_quat_xyzw": self.pose.quat_xyzw().tolist(),
            "n_frames": self.n_frames,
            "precheck": self.report.as_dict(),
        }


@dataclass
class GraspResolver:
    """Everything needed to turn a needle observation into a grasp command.

    The conventions are held here, once, and every model loaded through
    :meth:`from_files` is checked against them.  A model fitted under one
    grasp-point convention and applied under another is wrong by the needle
    radius -- 10.18 mm -- which is larger than the entire effect being
    corrected, and it is exactly the mistake section 7 is written to prevent.
    """

    model: ResidualModel
    hand_eye: HandEye = field(default_factory=HandEye)
    grasp_point: GraspPointSpec = field(default_factory=GraspPointSpec)
    #: the gripper orientation the calibration was taught at (section 8).
    #: ``None`` disables the check, which should only happen in a rehearsal.
    taught_orientation: Optional[Rotation] = None
    #: how far the commanded wrist may differ from the taught one before warning
    orientation_tol_deg: float = 10.0
    max_correction_mm: float = DEFAULT_MAX_CORRECTION_MM
    #: refuse to resolve at all if the model never passed validation
    require_validated: bool = True

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_files(cls, model_path, **kw) -> "GraspResolver":
        model = ResidualModel.load(model_path)
        meta = model.metadata
        hand_eye = kw.pop("hand_eye", None)
        grasp_point = kw.pop("grasp_point", None)
        if hand_eye is None and "hand_eye_T" in meta:
            hand_eye = HandEye(np.asarray(meta["hand_eye_T"]), source=str(model_path))
        if grasp_point is None and "grasp_point_spec" in meta:
            grasp_point = GraspPointSpec.from_dict(meta["grasp_point_spec"])
        if "taught_orientation" not in kw:
            quat = meta.get("taught_orientation_quat_xyzw")
            kw["taught_orientation"] = (
                None if quat is None else Rotation.from_quat(np.asarray(quat))
            )
        return cls(
            model=model,
            hand_eye=hand_eye or HandEye(),
            grasp_point=grasp_point or GraspPointSpec(),
            **kw,
        )

    # -- the static checks, run once before an episode ---------------------
    def precheck(self, strict: bool = False) -> PrecheckReport:
        rep = PrecheckReport(strict=strict)
        meta = self.model.metadata

        rep.add(
            "calibration.model", PASS, self.model.describe(),
            **{k: meta.get(k) for k in ("n_placements", "cv_rmse_mm", "robust")},
        )

        bound = self.model.max_correction_mm()
        rep.add(
            "calibration.bound",
            PASS if bound <= self.max_correction_mm else FAIL,
            f"the model can command at most {bound:.2f} mm of correction inside "
            f"its box (ceiling {self.max_correction_mm:.1f} mm)",
            bound_mm=bound,
        )

        expected = meta.get("convention_digest")
        actual = convention_digest(self.hand_eye, self.grasp_point)
        rep.add(
            "calibration.conventions",
            PASS if (expected is None or expected == actual) else FAIL,
            (
                f"hand-eye and needle-point conventions match the ones the model "
                f"was fitted under ({actual})"
                if expected is None or expected == actual
                else f"the model was fitted under conventions {expected} and is "
                f"being run under {actual}: recollect, or restore the "
                "transform and grasp point it was fitted with (section 15)"
            ),
            expected=expected, actual=actual,
        )

        validated = bool(meta.get("validated", False))
        rep.add(
            "calibration.validated",
            PASS if validated else (FAIL if self.require_validated else WARN),
            (
                f"cross-validated at {meta.get('cv_rmse_mm', float('nan')):.3f} mm "
                f"against {meta.get('cv_baseline_mm', float('nan')):.3f} mm uncorrected"
                if validated
                else "this model carries no cross-validated error: it has never "
                "been shown to help at a needle it did not see"
            ),
        )

        for problem in self.model.check_sane(self.max_correction_mm):
            rep.add("calibration.sanity", WARN, problem)

        return rep

    # -- the per-needle path -----------------------------------------------
    def resolve(
        self,
        pose_estimates,
        orientation: Optional[Rotation] = None,
        jaw: float = 0.0,
        strict: bool = False,
    ) -> GraspTarget:
        """Section 14, once, for one needle.

        ``pose_estimates`` is one 4x4 FoundationPose estimate or several frames
        of the same stationary needle -- section 6's averaging, which divides
        the random part of the perception error by ``sqrt(M)`` and leaves the
        systematic part, the part being corrected, alone.
        """
        Ts = np.asarray(pose_estimates, dtype=np.float64)
        if Ts.ndim == 2:
            Ts = Ts[None]
        rep = self.precheck(strict=strict)

        reasons = flip_reasons(Ts)
        rep.add(
            "calibration.frames",
            PASS if not reasons else FAIL,
            (
                f"{len(Ts)} pose estimate(s) agree"
                if not reasons
                else "; ".join(reasons)
            ),
            n_frames=len(Ts),
        )
        T_CN = average_poses(Ts) if not reasons else Ts[0]

        p_cam = self.grasp_point.in_camera(T_CN)
        p_nom = nominal_position(T_CN, self.hand_eye, self.grasp_point)

        outside = self.model.workspace.outside(p_nom)
        rep.add(
            "calibration.region",
            PASS if not outside else FAIL,
            (
                f"needle sits inside the calibrated {self.model.workspace.describe()}"
                if not outside
                else "; ".join(outside)
                + " -- the correction was never measured here and a Bernstein "
                "polynomial does not degrade gracefully outside its box (section 15)"
            ),
        )

        try:
            correction = self.model.with_policy(
                "clamp" if outside else self.model.outside_policy
            ).residual(p_nom)[0]
        except OutsideCalibratedRegion:
            correction = np.zeros(3)

        p_target = p_nom + correction

        R = (orientation or self.taught_orientation or Rotation.identity()).as_matrix()
        if orientation is not None and self.taught_orientation is not None:
            delta = np.degrees(
                (self.taught_orientation.inv() * orientation).magnitude()
            )
            rep.add(
                "calibration.orientation",
                PASS if delta <= self.orientation_tol_deg else WARN,
                f"commanded wrist is {delta:.1f} deg from the orientation the "
                f"calibration was taught at (tolerance {self.orientation_tol_deg:.0f} deg)"
                + (
                    ""
                    if delta <= self.orientation_tol_deg
                    else "; the jaw offset is fixed in the TOOL frame, so this "
                    "much rotation moves the grasp point by roughly "
                    f"{2 * 5.0 * np.sin(np.deg2rad(delta) / 2):.2f} mm for a 5 mm "
                    "offset and the position correction cannot see it (section 8)"
                ),
                delta_deg=float(delta),
            )
        elif self.taught_orientation is None:
            rep.add(
                "calibration.orientation", WARN,
                "no taught gripper orientation was recorded with this model, so "
                "the section 8 freeze cannot be checked",
            )

        return GraspTarget(
            T_CN=T_CN,
            p_camera_m=p_cam,
            p_nominal_m=p_nom,
            correction_m=correction,
            p_target_m=p_target,
            pose=Pose(p_target, R, float(jaw)),
            n_frames=len(Ts),
            report=rep,
        )
