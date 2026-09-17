"""Calibration samples, keyed by *physical needle placement*.

The grouping is the whole point.  Section 12 of the protocol: a placement, not a
frame, is the unit of independent evidence.  Ten FoundationPose frames of one
stationary needle and three repeated hand-taught grasps of it are thirty rows
and **one** sample.  Splitting them across a train/test boundary produces a
cross-validated error that measures how repeatable the perception is, not
whether the correction field generalises -- and it will look excellent.

So the dataset is a list of :class:`Placement` objects, each holding its own
repeats, and every split in :mod:`.validate` is a split of that list.  There is
no code path in this package that can separate two frames of the same placement,
because there is no code path that ever sees a loose frame.

What a placement records
------------------------
======================  ====================================================
``pose_estimates``      M FoundationPose 4x4 estimates, camera frame
``grasp_poses``         K verified successful ``measured_cp`` poses, ECM frame
``verified``            a human watched the jaw close on the needle
======================  ====================================================

and from those, deterministically, ``p_nom`` (section 5 step 3), ``p_grasp``
(step 5) and ``r = p_grasp - p_nom`` (step 8).  The residual is never stored as
an independent field: it is recomputed from the raw measurements every time the
dataset is loaded, so a change of grasp-point convention or hand-eye transform
cannot leave a stale residual behind.  That is the failure section 7 warns about
made structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .perception import (
    DEFAULT_GRASP_ANGLE_DEG,
    GraspPointSpec,
    HandEye,
    average_poses,
    convention_digest,
    flip_reasons,
    nominal_position,
)

FORMAT_VERSION = 1


def _pose_from_pos_quat(pos, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


@dataclass
class Placement:
    """One physical needle placement and every measurement taken of it."""

    placement_id: str
    #: M FoundationPose estimates of the stationary needle, camera frame, 4x4
    pose_estimates: List[np.ndarray]
    #: K poses read from <arm>/measured_cp at a verified successful grasp, ECM
    grasp_poses: List[np.ndarray]
    #: a human confirmed that closing the jaw here actually took the needle
    verified: bool = True
    note: str = ""
    recorded_at: str = ""

    def __post_init__(self):
        self.pose_estimates = [
            np.asarray(T, dtype=np.float64).reshape(4, 4) for T in self.pose_estimates
        ]
        self.grasp_poses = [
            np.asarray(T, dtype=np.float64).reshape(4, 4) for T in self.grasp_poses
        ]
        if not self.pose_estimates:
            raise ValueError(f"placement {self.placement_id!r} has no pose estimates")
        if not self.grasp_poses:
            raise ValueError(f"placement {self.placement_id!r} has no grasp poses")
        if not self.recorded_at:
            self.recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # -- derived quantities ------------------------------------------------
    @property
    def n_pose_frames(self) -> int:
        return len(self.pose_estimates)

    @property
    def n_grasps(self) -> int:
        return len(self.grasp_poses)

    def mean_pose_estimate(self) -> np.ndarray:
        return average_poses(self.pose_estimates)

    def p_grasp(self) -> np.ndarray:
        """Mean measured grasp position, ECM frame, metres."""
        return np.mean([T[:3, 3] for T in self.grasp_poses], axis=0)

    def grasp_rotation(self) -> Rotation:
        return Rotation.from_matrix(np.array([T[:3, :3] for T in self.grasp_poses])).mean()

    def p_nom(self, hand_eye: HandEye, grasp_point: GraspPointSpec) -> np.ndarray:
        return nominal_position(self.mean_pose_estimate(), hand_eye, grasp_point)

    def residual(self, hand_eye: HandEye, grasp_point: GraspPointSpec) -> np.ndarray:
        return self.p_grasp() - self.p_nom(hand_eye, grasp_point)

    # -- repeatability, section 6 -----------------------------------------
    def perception_spread_mm(self, hand_eye: HandEye, grasp_point: GraspPointSpec) -> np.ndarray:
        """Per-axis standard deviation of the nominal point over the M frames."""
        if self.n_pose_frames < 2:
            return np.full(3, np.nan)
        pts = np.array(
            [nominal_position(T, hand_eye, grasp_point) for T in self.pose_estimates]
        )
        return pts.std(axis=0, ddof=1) * 1000.0

    def grasp_spread_mm(self) -> np.ndarray:
        """Per-axis standard deviation of the taught grasp over the K repeats."""
        if self.n_grasps < 2:
            return np.full(3, np.nan)
        return np.array([T[:3, 3] for T in self.grasp_poses]).std(axis=0, ddof=1) * 1000.0

    def problems(self, hand_eye: HandEye, grasp_point: GraspPointSpec) -> list:
        """Reasons this placement should not go into a fit."""
        out = []
        if not self.verified:
            out.append("the grasp was never confirmed to have taken the needle")
        out += [f"pose repeats: {r}" for r in flip_reasons(self.pose_estimates)]
        if self.n_grasps >= 2:
            spread = np.linalg.norm(
                np.array([T[:3, 3] for T in self.grasp_poses]) - self.p_grasp(), axis=1
            ).max() * 1000.0
            if spread > 5.0:
                out.append(
                    f"the {self.n_grasps} taught grasps disagree by up to "
                    f"{spread:.1f} mm -- they are not repeats of one target"
                )
        return out

    # -- io ----------------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "placement_id": self.placement_id,
            "pose_estimates_T_CN": [T.tolist() for T in self.pose_estimates],
            "grasp_poses_T_E": [T.tolist() for T in self.grasp_poses],
            "verified": bool(self.verified),
            "note": self.note,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Placement":
        return cls(
            placement_id=str(d["placement_id"]),
            pose_estimates=[np.asarray(T) for T in d["pose_estimates_T_CN"]],
            grasp_poses=[np.asarray(T) for T in d["grasp_poses_T_E"]],
            verified=bool(d.get("verified", True)),
            note=str(d.get("note", "")),
            recorded_at=str(d.get("recorded_at", "")),
        )

    @classmethod
    def from_measurements(
        cls,
        placement_id: str,
        pose_estimates,
        grasp_positions,
        grasp_quats_xyzw,
        **kw,
    ) -> "Placement":
        """Convenience constructor from the shapes a ROS log actually has."""
        grasps = [
            _pose_from_pos_quat(p, q)
            for p, q in zip(grasp_positions, grasp_quats_xyzw)
        ]
        return cls(placement_id, list(pose_estimates), grasps, **kw)


# ---------------------------------------------------------------------------
@dataclass
class CalibrationDataset:
    """Every placement, under one set of conventions.

    The conventions travel with the data because they have to: a residual
    computed under one grasp-point convention and modelled under another is not
    wrong by a little, it is wrong by the needle radius.
    """

    placements: List[Placement] = field(default_factory=list)
    hand_eye: HandEye = field(default_factory=HandEye)
    grasp_point: GraspPointSpec = field(default_factory=GraspPointSpec)
    #: free-text: arm, instrument serial, camera, FoundationPose checkpoint, date
    provenance: dict = field(default_factory=dict)

    # -- building ----------------------------------------------------------
    def add(self, placement: Placement) -> "CalibrationDataset":
        if any(p.placement_id == placement.placement_id for p in self.placements):
            raise ValueError(
                f"placement id {placement.placement_id!r} is already in this dataset; "
                "ids are the grouping key for cross-validation and must be unique"
            )
        self.placements.append(placement)
        return self

    def __len__(self) -> int:
        return len(self.placements)

    def usable(self) -> "CalibrationDataset":
        """The subset that is fit to fit on, with the reasons for each drop."""
        keep = [
            p for p in self.placements
            if not p.problems(self.hand_eye, self.grasp_point)
        ]
        return CalibrationDataset(
            keep, self.hand_eye, self.grasp_point, dict(self.provenance)
        )

    def dropped(self) -> list:
        return [
            (p.placement_id, p.problems(self.hand_eye, self.grasp_point))
            for p in self.placements
            if p.problems(self.hand_eye, self.grasp_point)
        ]

    def subset(self, ids) -> "CalibrationDataset":
        wanted = set(ids)
        return CalibrationDataset(
            [p for p in self.placements if p.placement_id in wanted],
            self.hand_eye,
            self.grasp_point,
            dict(self.provenance),
        )

    # -- the arrays a fit consumes ----------------------------------------
    @property
    def ids(self) -> List[str]:
        return [p.placement_id for p in self.placements]

    def p_nom(self) -> np.ndarray:
        return np.array([p.p_nom(self.hand_eye, self.grasp_point) for p in self.placements])

    def p_grasp(self) -> np.ndarray:
        return np.array([p.p_grasp() for p in self.placements])

    def residuals(self) -> np.ndarray:
        return self.p_grasp() - self.p_nom()

    # -- the noise floor, section 6 ---------------------------------------
    def noise_floor(self) -> dict:
        """What the residual's random component is, measured from repeats.

        This is the number every other number in the report has to be read
        against.  A deterministic correction field can remove a systematic bias;
        it cannot predict this.  If the model's cross-validated error is already
        near it, the model is done -- and if the floor itself is several
        millimetres, no polynomial of any degree is going to help and the
        protocol should stop at section 6 rather than continue to section 9.
        """
        perc = np.array(
            [p.perception_spread_mm(self.hand_eye, self.grasp_point) for p in self.placements]
        )
        grasp = np.array([p.grasp_spread_mm() for p in self.placements])

        def pool(a):
            a = a[np.isfinite(a).all(axis=1)]
            if len(a) == 0:
                return None
            return np.sqrt((a ** 2).mean(axis=0))

        perc_sd, grasp_sd = pool(perc), pool(grasp)
        out = {
            "n_placements": len(self.placements),
            "n_with_pose_repeats": int(np.isfinite(perc).all(axis=1).sum()),
            "n_with_grasp_repeats": int(np.isfinite(grasp).all(axis=1).sum()),
            "perception_sd_mm": None if perc_sd is None else perc_sd.tolist(),
            "grasp_sd_mm": None if grasp_sd is None else grasp_sd.tolist(),
            # Both terms have to be measured for the floor to be a floor.  With
            # only one of them the number below is a LOWER bound, and every
            # adoption decision judged against it is too generous.  Measuring
            # the perception term needs >= 2 frames per placement; measuring the
            # teaching term needs >= 2 taught grasps, and the synthetic
            # rehearsal says the teaching term is the larger of the two.
            "complete": bool(perc_sd is not None and grasp_sd is not None),
        }
        # The residual's own noise is the two in quadrature, reduced by however
        # many repeats each was averaged over.
        if perc_sd is not None or grasp_sd is not None:
            mperc = np.median([p.n_pose_frames for p in self.placements])
            mgrasp = np.median([p.n_grasps for p in self.placements])
            a = np.zeros(3) if perc_sd is None else perc_sd ** 2 / max(mperc, 1)
            b = np.zeros(3) if grasp_sd is None else grasp_sd ** 2 / max(mgrasp, 1)
            sd = np.sqrt(a + b)
            out["residual_sd_mm"] = sd.tolist()
            out["residual_sd_3d_mm"] = float(np.linalg.norm(sd))
        return out

    # -- the orientation freeze, section 8 --------------------------------
    def orientation_spread_deg(self) -> float:
        """Geodesic spread of the taught gripper orientation across placements.

        Section 8 requires this to be small.  It is not a style point: the
        offset between ``measured_cp`` and the point between the jaws is fixed
        in the *tool* frame, so in ECM coordinates it is ``R_tool @ d``.  If the
        wrist turns between placements, that term moves without the position
        moving, and a position-only model has to explain it as noise.  A
        millimetre of jaw offset and thirty degrees of wrist spread is half a
        millimetre of unexplainable residual.
        """
        if len(self.placements) < 2:
            return 0.0
        rots = Rotation.from_matrix(
            np.array([p.grasp_rotation().as_matrix() for p in self.placements])
        )
        mean = rots.mean()
        return float(np.degrees((mean.inv() * rots).magnitude().max()))

    def jaw_offset_leakage_mm(self, jaw_offset_mm: float = 5.0) -> float:
        """How much residual an orientation spread injects, for a given offset.

        ``jaw_offset_mm`` is how far the physical pinch point sits from the
        frame ``measured_cp`` reports.  On a Large Needle Driver the pitch-to-yaw
        link alone is 9 mm, so five is a conservative default rather than a
        generous one.
        """
        spread = np.deg2rad(self.orientation_spread_deg())
        return float(2.0 * jaw_offset_mm * np.sin(spread / 2.0))

    # -- coverage ----------------------------------------------------------
    def coverage(self) -> dict:
        p = self.p_nom()
        if len(p) == 0:
            return {"span_cm": [0.0, 0.0, 0.0], "n": 0}
        span = (p.max(axis=0) - p.min(axis=0)) * 100.0
        # How much of the box the samples actually reach into, per axis: the
        # ratio of the sample standard deviation to that of a uniform fill.
        fill = (p.std(axis=0) * 100.0) / np.maximum(span / np.sqrt(12.0), 1e-9)
        return {
            "n": len(p),
            "span_cm": span.tolist(),
            "uniformity": fill.tolist(),
            "centroid_cm": (p.mean(axis=0) * 100.0).tolist(),
        }

    def problems(self) -> list:
        """Everything about this dataset that should be fixed before fitting."""
        out = []
        n = len(self.placements)
        if n < 8:
            out.append(
                f"{n} placements is not enough to cross-validate anything; "
                "see validate.variance_budget for what each model costs"
            )
        cov = self.coverage()
        if n:
            thin = [
                "xyz"[i] for i in range(3) if cov["span_cm"][i] < 0.5
            ]
            if thin:
                span_text = ", ".join(f"{s:.2f} cm" for s in cov["span_cm"])
                out.append(
                    "the placements barely move in "
                    + ", ".join(thin)
                    + f" (span {span_text}): a correction field fitted here cannot "
                    "be trusted to vary along those axes -- section 5 asks for "
                    "variation in x, y AND z"
                )
        spread = self.orientation_spread_deg()
        if spread > 10.0:
            out.append(
                f"the taught gripper orientation varies by {spread:.1f} deg across "
                f"placements, which leaks about "
                f"{self.jaw_offset_leakage_mm():.2f} mm of residual that no "
                "position-only model can represent (section 8)"
            )
        for pid, reasons in self.dropped():
            out.append(f"placement {pid}: " + "; ".join(reasons))
        return out

    # -- io ----------------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "hand_eye": self.hand_eye.as_dict(),
            "grasp_point": self.grasp_point.as_dict(),
            "provenance": self.provenance,
            "placements": [p.as_dict() for p in self.placements],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationDataset":
        version = int(d.get("format_version", 0))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"calibration dataset format version {version}, this package "
                f"reads {FORMAT_VERSION}"
            )
        return cls(
            placements=[Placement.from_dict(p) for p in d["placements"]],
            hand_eye=HandEye.from_dict(d["hand_eye"]),
            grasp_point=GraspPointSpec.from_dict(d["grasp_point"]),
            provenance=dict(d.get("provenance", {})),
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path) -> "CalibrationDataset":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # -- what the conventions were -----------------------------------------
    def convention_digest(self) -> str:
        """Fingerprint of hand-eye transform + grasp point; see
        :func:`~.perception.convention_digest`."""
        return convention_digest(self.hand_eye, self.grasp_point)
