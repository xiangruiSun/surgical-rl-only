"""From a FoundationPose estimate to a nominal ECM-frame point.

This is the *coarse geometric backbone* of section 2: the part the empirical
calibration refines rather than replaces.

    T_CN  (FoundationPose)  ->  a named point on the needle  ->  ^E T_C  ->  p_nom

Two conventions have to be nailed down before any of it means anything, and
getting either wrong costs more than the polynomial can ever win back.

Which point on the needle
-------------------------
The SurgicAI needle mesh has its origin at the **centre of the arc** -- a point
in empty space, ``10.18 mm`` from any part of the wire.  Straight from
``RL/utils/needle_kinematics_new.py``::

    Radius  = 0.1018            (AMBF units, /10 -> metres)
    T_bINn  = (-R,              0,              0)   yaw   0      "base"
    T_bmINn = (-R cos(pi/6),    R sin(pi/6),    0)   yaw -pi/6    "base-mid"
    T_mINn  = (-R cos(pi/3),    R sin(pi/3),    0)   yaw -pi/3    "mid"
    T_tINn  = (-R cos(2pi/3),   R sin(2pi/3),   0)   yaw -2pi/3   "tip"

and generally ``get_pose_angle(theta)`` puts a frame at
``R(-cos t, sin t, 0)`` rotated by ``Rz(-t)``.  SurgicAI's own grasp target is
``get_bm_pose()`` -- theta = 30 degrees -- which is what
``needle_goal_evaluator`` composes with a 7 mm standoff to build the Approach
policy's goal.

So if the FoundationPose translation (the mesh origin) is used as "the needle
position" while the arm is taught to grasp at ``bm``, the residual carries
``R_needle @ q_bm``, which **rotates with the needle**.  Over the +-30 degree
needle-yaw envelope this project already assumes, the part of that term a
position-only model cannot represent is 2.7 mm RMS and 5.2 mm worst case --
the same size as the residual being modelled.  Section 7 of the protocol is not
a tidy-up; it is the difference between a learnable field and a hopeless one.

Which hand-eye transform
------------------------
``^E T_C`` is a fitted quantity with its own error, and that error is the reason
this package exists.  It is worth knowing how big it is expected to be: the
JHU dVRK registration package that most groups use for exactly this
(``jhu-dvrk/dvrk_camera_registration``) says in its own README that it "seems to
achieve an accuracy closer to 5mm to 10mm cube", and the residual it prints
while doing so is the standard deviation of the Frobenius norm of a composed
homogeneous matrix -- a number with no unit anyone can act on.  A 5-10 mm
registration error, expressed at an 8 cm working distance, is a rotation error
of well under two degrees, and section 2 of the protocol document shows that
such an error contributes a residual that is **exactly affine** in the nominal
position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

#: Arc radius of the SurgicAI needle, metres.
#: ``RL/utils/needle_kinematics_new.py`` -- ``Radius = 0.1018`` in AMBF units,
#: divided by 10 everywhere it is used.
NEEDLE_RADIUS_M = 0.1018 / 10.0

#: The named arc angles, degrees, in SurgicAI's own parameterisation.
NEEDLE_ANGLES_DEG = {"base": 0.0, "bm": 30.0, "mid": 60.0, "tip": 120.0}

#: The angle SurgicAI's Approach policy is trained to grasp at.
#: ``subtask_env.needle_goal_evaluator`` defaults to ``get_bm_pose()``.
#: Note that ``Low_level_env_complete`` uses ``deg_angle=105`` for ``psm_idx=1``
#: -- if this deployment's PSM1 task inherited that, the canonical angle here
#: must change to match, and the calibration must be recollected.  It is a
#: convention, not a measurement, and mixing two of them is section 7's warning.
DEFAULT_GRASP_ANGLE_DEG = NEEDLE_ANGLES_DEG["bm"]


# ---------------------------------------------------------------------------
# needle geometry
# ---------------------------------------------------------------------------
def needle_point_N(angle_deg: float, radius_m: float = NEEDLE_RADIUS_M) -> np.ndarray:
    """A point on the needle arc, in the needle mesh frame, metres."""
    t = np.deg2rad(float(angle_deg))
    return float(radius_m) * np.array([-np.cos(t), np.sin(t), 0.0])


def needle_frame_N(angle_deg: float, radius_m: float = NEEDLE_RADIUS_M) -> np.ndarray:
    """The full 4x4 frame SurgicAI puts at that arc angle, in the mesh frame."""
    t = np.deg2rad(float(angle_deg))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", -t).as_matrix()
    T[:3, 3] = needle_point_N(angle_deg, radius_m)
    return T


@dataclass(frozen=True)
class GraspPointSpec:
    """Which point on the needle the whole experiment is about.

    ``mode="arc"`` (the default and the only one that should be used for a
    calibration) takes the point at ``angle_deg`` on the arc, carried by the
    full FoundationPose orientation.  ``mode="mesh_origin"`` uses the raw
    FoundationPose translation; it exists so the cost of that choice can be
    *measured* on real data rather than argued about, and
    :meth:`orientation_sensitivity_mm` says in advance what it will be.
    """

    mode: str = "arc"
    angle_deg: float = DEFAULT_GRASP_ANGLE_DEG
    radius_m: float = NEEDLE_RADIUS_M

    def __post_init__(self):
        if self.mode not in ("arc", "mesh_origin"):
            raise ValueError(f"grasp point mode must be 'arc' or 'mesh_origin'; got {self.mode!r}")
        if not np.isfinite(self.angle_deg):
            raise ValueError("grasp angle must be finite")
        if not (0.0 < float(self.radius_m) < 0.05):
            raise ValueError("needle radius must be a positive number of metres under 5 cm")

    def point_N(self) -> np.ndarray:
        if self.mode == "mesh_origin":
            return np.zeros(3)
        return needle_point_N(self.angle_deg, self.radius_m)

    def in_camera(self, T_CN) -> np.ndarray:
        """The grasp point in the camera frame, given the full needle pose."""
        T = np.asarray(T_CN, dtype=np.float64).reshape(4, 4)
        return T[:3, :3] @ self.point_N() + T[:3, 3]

    def orientation_sensitivity_mm(self, yaw_envelope_deg: float = 30.0) -> dict:
        """How much of this convention's offset a position-only model cannot see.

        The offset ``R_needle @ q`` is constant only if ``R_needle`` is.  Over a
        needle-yaw envelope of ``+-yaw_envelope_deg`` the non-constant part is
        what no ``f(x, y, z)`` can represent, and it lands in the noise term
        ``epsilon`` of section 4 -- where it will look like irreducible
        randomness and will not be.
        """
        q = self.point_N()
        if np.allclose(q, 0.0):
            return {"rms_mm": 0.0, "max_mm": 0.0, "note": "mesh origin carries no offset"}
        angs = np.linspace(-float(yaw_envelope_deg), float(yaw_envelope_deg), 181)
        pts = np.array(
            [Rotation.from_euler("z", np.deg2rad(a)).as_matrix() @ q for a in angs]
        )
        spread = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
        return {
            "rms_mm": float(spread.mean() * 1000.0),
            "max_mm": float(spread.max() * 1000.0),
            "note": f"|q| = {np.linalg.norm(q)*1000:.2f} mm at {self.angle_deg:.0f} deg",
        }

    def describe(self) -> str:
        if self.mode == "mesh_origin":
            return "the FoundationPose translation (needle mesh origin, the ARC CENTRE)"
        return (
            f"the point {self.angle_deg:.0f} deg along the needle arc "
            f"({np.linalg.norm(self.point_N())*1000:.2f} mm from the mesh origin)"
        )

    def as_dict(self) -> dict:
        return {"mode": self.mode, "angle_deg": self.angle_deg, "radius_m": self.radius_m}

    @classmethod
    def from_dict(cls, d: dict) -> "GraspPointSpec":
        return cls(
            mode=str(d.get("mode", "arc")),
            angle_deg=float(d.get("angle_deg", DEFAULT_GRASP_ANGLE_DEG)),
            radius_m=float(d.get("radius_m", NEEDLE_RADIUS_M)),
        )


# ---------------------------------------------------------------------------
# the fitted camera -> ECM-tip transform
# ---------------------------------------------------------------------------
#: The transform this deployment was handed.  Right camera -> ECM tip.
#: Orthonormal to 7e-11, a rotation of 178.15 degrees about an axis within
#: 6.9 degrees of -z, with a 16.0 mm translation.
DEFAULT_T_EC = np.array(
    [
        [-0.9967665891, 0.0314363494, 0.0739467561, 0.0106338930],
        [-0.0331846353, -0.9991951845, -0.0225336033, 0.0081274151],
        [0.0731788684, -0.0249146391, 0.9970075797, 0.0088570478],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HandEye:
    """``^E T_C``: a point in the camera frame, put into the ECM-tip frame.

    Frozen, checked for orthonormality on construction, and carrying a
    ``source`` string -- because section 15 lists "camera mounting", "ECM-camera
    relationship" and "hand-eye calibration" as changes that invalidate the
    empirical calibration, and the only way to notice one of those is to have
    written down which transform the data was collected under.
    """

    T: np.ndarray = None
    source: str = "supplied with the task description, 2026-09"
    #: the registration's own claimed accuracy, if it is known, in millimetres
    stated_accuracy_mm: Optional[float] = None

    def __post_init__(self):
        T = DEFAULT_T_EC if self.T is None else np.asarray(self.T, dtype=np.float64)
        T = T.reshape(4, 4).copy()
        R = T[:3, :3]
        orth = float(np.abs(R.T @ R - np.eye(3)).max())
        if orth > 1e-6:
            raise ValueError(
                f"hand-eye rotation is not orthonormal (max |R'R - I| = {orth:.2e}); "
                "orthonormalise it before use, and find out why the fit drifted"
            )
        if np.linalg.det(R) < 0:
            raise ValueError("hand-eye rotation has negative determinant (it is a reflection)")
        if not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError("hand-eye bottom row must be [0, 0, 0, 1]")
        object.__setattr__(self, "T", T)

    # -- use ---------------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        return self.T[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.T[:3, 3]

    def point_to_ecm(self, points_C) -> np.ndarray:
        p = np.asarray(points_C, dtype=np.float64).reshape(-1, 3)
        return p @ self.R.T + self.t

    def pose_to_ecm(self, T_CN) -> np.ndarray:
        return self.T @ np.asarray(T_CN, dtype=np.float64).reshape(4, 4)

    def inverse(self) -> "HandEye":
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, 3] = -self.R.T @ self.t
        return HandEye(T, source=f"inverse of: {self.source}")

    # -- diagnostics -------------------------------------------------------
    def rotation_error_to_mm(self, angle_deg: float, range_m: float) -> float:
        """What a rotation error of this size costs at that working distance.

        The number that decides whether this whole exercise is about an affine
        term or something more interesting.
        """
        return float(range_m * np.deg2rad(float(angle_deg)) * 1000.0)

    def describe(self) -> str:
        rot = Rotation.from_matrix(self.R)
        axis = rot.as_rotvec()
        ang = np.degrees(np.linalg.norm(axis))
        unit = axis / max(np.linalg.norm(axis), 1e-12)
        acc = (
            "" if self.stated_accuracy_mm is None
            else f", stated accuracy {self.stated_accuracy_mm:.1f} mm"
        )
        return (
            f"rotation {ang:.2f} deg about ({unit[0]:+.3f}, {unit[1]:+.3f}, "
            f"{unit[2]:+.3f}), translation {np.linalg.norm(self.t)*1000:.2f} mm"
            f"{acc}  [{self.source}]"
        )

    def as_dict(self) -> dict:
        return {
            "T": self.T.tolist(),
            "source": self.source,
            "stated_accuracy_mm": self.stated_accuracy_mm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HandEye":
        return cls(
            T=np.asarray(d["T"], dtype=np.float64),
            source=str(d.get("source", "unrecorded")),
            stated_accuracy_mm=d.get("stated_accuracy_mm"),
        )

    def digest(self) -> str:
        """Short stable fingerprint, so a model can record which transform it
        was fitted against and refuse to run under a different one."""
        import hashlib

        return hashlib.sha256(
            np.ascontiguousarray(np.round(self.T, 9)).tobytes()
        ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# the nominal pipeline
# ---------------------------------------------------------------------------
def nominal_position(T_CN, hand_eye: HandEye, grasp_point: GraspPointSpec) -> np.ndarray:
    """``p_nom`` -- the whole of section 5 steps 2 and 3, in one call.

    ``T_CN`` is the full 4x4 FoundationPose estimate of the needle in the camera
    frame.  The full pose is required, not just the translation: the grasp point
    is a point on the arc, and placing it needs the orientation.
    """
    return hand_eye.point_to_ecm(grasp_point.in_camera(T_CN)).reshape(3)


def average_poses(T_CN_list) -> np.ndarray:
    """Mean of several FoundationPose estimates of a *stationary* needle.

    Section 6: averaging M frames divides the random part of the perception
    error by sqrt(M) and leaves the systematic part -- which is exactly the part
    being learned -- untouched.  Translations average arithmetically; rotations
    average as the chordal (quaternion) mean, which for the sub-degree spreads a
    stationary scene produces is indistinguishable from the geodesic mean and
    cannot fail to converge.

    Raises if the estimates disagree enough to suggest a flip rather than noise;
    see :func:`flip_reasons`.
    """
    Ts = np.asarray(T_CN_list, dtype=np.float64).reshape(-1, 4, 4)
    if len(Ts) == 0:
        raise ValueError("cannot average an empty list of poses")
    reasons = flip_reasons(Ts)
    if reasons:
        raise ValueError(
            "these estimates are not repeated measurements of one pose: "
            + "; ".join(reasons)
        )
    R = Rotation.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = Ts[:, :3, 3].mean(axis=0)
    return T


def convention_digest(hand_eye: "HandEye", grasp_point: GraspPointSpec) -> str:
    """Fingerprint of the two conventions a residual is only meaningful under.

    A dataset records it; a fitted model carries it; :mod:`.resolve` refuses to
    run when it does not match.  That is the machine-checkable half of section
    15 -- the half that catches a changed hand-eye transform or a changed grasp
    point before an arm moves, rather than afterwards from the miss distance.
    """
    import hashlib
    import json

    blob = json.dumps(
        {
            "hand_eye": np.round(hand_eye.T, 9).tolist(),
            "grasp_point": grasp_point.as_dict(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def flip_reasons(T_CN_list, rotation_tol_deg: float = 20.0,
                 translation_tol_mm: float = 10.0) -> list:
    """Why a set of repeats of a stationary needle does not look like noise.

    A learned 6-D pose estimator on a thin, near-symmetric object does not fail
    by drifting: it fails by landing on the wrong branch.  This project has seen
    it before -- ``RL/needle_reset_ranges.py`` records that the pose audit "was
    reliable at 20 degrees but developed near-180 degree failures at 40
    degrees".  One flipped estimate inside a least-squares fit moves the whole
    correction field, because least squares has no way to disbelieve a point.

    The defaults are deliberately loose.  This is not a precision test; it is
    the guard that stops a 180-degree failure being averaged into a calibration
    sample and then modelled as though it were a smooth function of position.
    """
    Ts = np.asarray(T_CN_list, dtype=np.float64).reshape(-1, 4, 4)
    if len(Ts) < 2:
        return []
    out = []
    rots = Rotation.from_matrix(Ts[:, :3, :3])
    ref = rots[0]
    ang = np.degrees((ref.inv() * rots).magnitude())
    if ang.max() > float(rotation_tol_deg):
        out.append(
            f"orientation spread {ang.max():.1f} deg across repeats "
            f"(tolerance {rotation_tol_deg:.0f} deg) -- a pose flip, not noise"
        )
    t = Ts[:, :3, 3]
    spread = np.linalg.norm(t - t.mean(axis=0), axis=1).max() * 1000.0
    if spread > float(translation_tol_mm):
        out.append(
            f"translation spread {spread:.1f} mm across repeats "
            f"(tolerance {translation_tol_mm:.0f} mm)"
        )
    return out
