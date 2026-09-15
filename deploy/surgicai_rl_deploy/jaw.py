"""Jaw command mapping and *observed* grasp evidence.

Read this before using anything in here
---------------------------------------
On a real dVRK there is **no grasp sensor**.  In simulation the SurgicAI
environment calls ``psm.actuators[0].actuate("Needle")`` -- a magnetic
constraint plus a ghost finger sensor -- and ``grasp_status()["needle_grasped"]``
is ground truth.  Neither exists on hardware.  Everything this module produces
is therefore an **observation**, never a verification:

* ``JawEvidence.jaw_blocked`` means "the jaw stopped further open than it does
  with nothing between the fingers".  A needle, a tissue fold, a suture, a
  tendon that has gone slack and a mis-zeroed jaw all produce that signal.
* ``JawEvidence.verified`` is hard-wired to ``False`` and exists so that any
  log line, any JSON record and any downstream reader is forced to see that the
  grasp was *not* confirmed.

If the sequencer is allowed to lift on this evidence alone, that is a
deliberate operator choice (``--grasp-gate evidence``), and it is recorded as
such in the trace.  The default gate is ``manual``: a human looks at the scene
and says go.

Units
-----
dVRK jaw angles are radians on ``<arm>/jaw/servo_jp`` and
``<arm>/jaw/measured_js``.  ``dvrk.psm``'s own helpers use
``open() -> +60 deg`` and ``close() -> -20 deg``: **negative is squeeze**.  The
jaw does not stop at 0; commanding a negative angle is how tendon tension --
and therefore grip force -- is produced.  A command of 0.0 rad closes the
fingers but holds nothing.

The policy's observation uses a *normalised* jaw in 0..1, which maps
``closed_rad -> 0`` and ``open_rad -> 1``.  The grip angle is deliberately
**outside** that range (negative normalised).  That is fine: the grip command
is produced by this deploy package, not by the policy, whose jaw channel stays
frozen unless ``--use-policy-jaw`` is passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

#: dVRK ``psm.jaw.open()`` default.
DEFAULT_OPEN_RAD = float(np.deg2rad(60.0))
#: Fingers touching, no tendon tension.
DEFAULT_CLOSED_RAD = 0.0
#: Squeeze angle used to hold the needle.  ``dvrk.psm.jaw.close()`` uses -20 deg;
#: -15 deg is a slightly gentler default for a thin needle.
DEFAULT_GRIP_RAD = float(np.deg2rad(-15.0))
#: Hard floor.  Commanding past this risks the tool, not the task.
MIN_GRIP_RAD = float(np.deg2rad(-25.0))
#: Hard ceiling on the open command.
MAX_OPEN_RAD = float(np.deg2rad(80.0))


class JawCalibrationError(ValueError):
    """Raised for a jaw calibration that would be unsafe or meaningless."""


@dataclass(frozen=True)
class JawCalibration:
    """Maps between dVRK jaw radians and the policy's normalised 0..1 jaw.

    ``open_rad``
        Angle treated as fully open, i.e. normalised 1.0.
    ``closed_rad``
        Angle treated as closed, i.e. normalised 0.0.  Fingers touch here but
        apply no force.
    ``grip_rad``
        Negative squeeze angle commanded while holding the needle.  Outside the
        normalised range by design.
    ``approach_open_rad``
        Angle the jaw is held at during the approach.  Defaults to a partial
        open so the gripper does not sweep a wide arc through the scene.
    """

    open_rad: float = DEFAULT_OPEN_RAD
    closed_rad: float = DEFAULT_CLOSED_RAD
    grip_rad: float = DEFAULT_GRIP_RAD
    approach_open_rad: float = float(np.deg2rad(40.0))

    def __post_init__(self):
        if not np.isfinite([self.open_rad, self.closed_rad, self.grip_rad,
                            self.approach_open_rad]).all():
            raise JawCalibrationError("jaw calibration must be finite")
        if self.open_rad <= self.closed_rad:
            raise JawCalibrationError(
                f"open_rad ({self.open_rad:.4f}) must exceed closed_rad "
                f"({self.closed_rad:.4f})"
            )
        if self.open_rad > MAX_OPEN_RAD + 1e-9:
            raise JawCalibrationError(
                f"open_rad {np.degrees(self.open_rad):.1f} deg exceeds the "
                f"{np.degrees(MAX_OPEN_RAD):.0f} deg ceiling"
            )
        if self.grip_rad < MIN_GRIP_RAD - 1e-9:
            raise JawCalibrationError(
                f"grip_rad {np.degrees(self.grip_rad):.1f} deg is below the "
                f"{np.degrees(MIN_GRIP_RAD):.0f} deg floor; that is a tool "
                "damage risk, not a firmer grasp"
            )
        if self.grip_rad > self.closed_rad:
            raise JawCalibrationError(
                "grip_rad must be at or below closed_rad: a non-negative grip "
                "command applies no tendon tension and holds nothing"
            )
        if not (self.closed_rad <= self.approach_open_rad <= self.open_rad):
            raise JawCalibrationError(
                "approach_open_rad must lie between closed_rad and open_rad"
            )

    # -- conversions -------------------------------------------------------
    def normalise(self, angle_rad: float) -> float:
        """Radians -> the policy's 0..1 jaw, clamped like the training env."""
        span = self.open_rad - self.closed_rad
        return float(np.clip((float(angle_rad) - self.closed_rad) / span, 0.0, 1.0))

    def normalise_unclamped(self, angle_rad: float) -> float:
        """Radians -> normalised jaw, *without* the 0..1 clamp.

        Use this for diagnostics only.  A squeeze angle maps below 0 here,
        which is exactly why the clamped form is what reaches the network.
        """
        span = self.open_rad - self.closed_rad
        return float((float(angle_rad) - self.closed_rad) / span)

    def to_rad(self, jaw_norm: float) -> float:
        span = self.open_rad - self.closed_rad
        return float(self.closed_rad + float(jaw_norm) * span)

    def describe(self) -> str:
        return (
            f"jaw: open {np.degrees(self.open_rad):.1f} deg, "
            f"approach {np.degrees(self.approach_open_rad):.1f} deg, "
            f"closed {np.degrees(self.closed_rad):.1f} deg, "
            f"grip {np.degrees(self.grip_rad):.1f} deg (negative = squeeze)"
        )

    def as_dict(self) -> dict:
        return {
            "open_deg": float(np.degrees(self.open_rad)),
            "closed_deg": float(np.degrees(self.closed_rad)),
            "grip_deg": float(np.degrees(self.grip_rad)),
            "approach_open_deg": float(np.degrees(self.approach_open_rad)),
        }


@dataclass(frozen=True)
class JawBaseline:
    """What closing on *nothing* looks like on this particular arm.

    Produced by ``tools/calibrate_jaw.py``: command the grip angle with an
    empty gripper, wait for the jaw to settle, and record where it actually
    stops and how much effort it draws.  Without this baseline the evidence
    below has no reference and ``JawEvidence.jaw_blocked`` stays ``None``.
    """

    #: measured angle the empty jaw settles at when commanded to grip_rad
    empty_close_rad: float
    #: |effort| the empty jaw draws at that angle, or None if the arm publishes
    #: no effort field on jaw/measured_js
    empty_close_effort: Optional[float] = None
    #: peak-to-peak spread of the settled measurement, used as the noise floor
    empty_close_rad_noise: float = float(np.deg2rad(0.5))
    empty_close_effort_noise: Optional[float] = None
    #: free-text provenance so a stale baseline is visible in the trace
    source: str = "unspecified"

    @staticmethod
    def from_dict(payload: dict) -> "JawBaseline":
        if "empty_close_rad" not in payload:
            raise JawCalibrationError(
                "jaw baseline file has no 'empty_close_rad'; regenerate it "
                "with tools/calibrate_jaw.py"
            )
        return JawBaseline(
            empty_close_rad=float(payload["empty_close_rad"]),
            empty_close_effort=(
                None if payload.get("empty_close_effort") is None
                else float(payload["empty_close_effort"])
            ),
            empty_close_rad_noise=float(
                payload.get("empty_close_rad_noise", np.deg2rad(0.5))
            ),
            empty_close_effort_noise=(
                None if payload.get("empty_close_effort_noise") is None
                else float(payload["empty_close_effort_noise"])
            ),
            source=str(payload.get("source", "unspecified")),
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class JawEvidence:
    """One cycle of jaw observation.  Not a grasp detector.  See module docs."""

    #: what we asked the jaw to do, radians
    commanded_rad: float
    #: what the jaw reports, radians; None if jaw/measured_js is silent
    measured_rad: Optional[float]
    #: |effort| from jaw/measured_js, or None if the arm publishes none
    effort: Optional[float] = None

    #: measured - commanded.  Positive means the jaw stopped short of the
    #: commanded squeeze, i.e. something is in the way.
    residual_rad: Optional[float] = None
    #: residual in excess of what an empty jaw shows, from the baseline
    residual_excess_rad: Optional[float] = None
    #: effort in excess of the empty-jaw baseline
    effort_excess: Optional[float] = None

    #: True  -> the jaw stopped further open than the empty-jaw baseline
    #: False -> it closed just like an empty jaw
    #: None  -> no baseline, or no jaw feedback: nothing can be said
    jaw_blocked: Optional[bool] = None

    #: Always False.  There is no grasp verification on this hardware.
    verified: bool = False
    #: Human-readable reason the above is what it is.
    note: str = ""

    def as_dict(self) -> dict:
        out = {
            "commanded_deg": float(np.degrees(self.commanded_rad)),
            "measured_deg": (
                None if self.measured_rad is None
                else float(np.degrees(self.measured_rad))
            ),
            "effort": self.effort,
            "residual_deg": (
                None if self.residual_rad is None
                else float(np.degrees(self.residual_rad))
            ),
            "residual_excess_deg": (
                None if self.residual_excess_rad is None
                else float(np.degrees(self.residual_excess_rad))
            ),
            "effort_excess": self.effort_excess,
            "jaw_blocked": self.jaw_blocked,
            "grasp_verified": False,
            "note": self.note,
        }
        return out


def evaluate_jaw_evidence(
    commanded_rad: float,
    measured_rad: Optional[float],
    effort: Optional[float] = None,
    baseline: Optional[JawBaseline] = None,
    *,
    residual_margin_rad: float = float(np.deg2rad(1.0)),
    effort_margin: Optional[float] = None,
) -> JawEvidence:
    """Turn one jaw reading into a :class:`JawEvidence` record.

    ``residual_margin_rad``
        How far past the empty-jaw stop angle the jaw must remain before the
        reading is called ``jaw_blocked``.  The default 1 deg is above typical
        encoder noise but is *not* calibrated for any particular needle: a
        0.5 mm needle wire may hold the jaw open by only a degree or two, and
        on some arms that is indistinguishable from tendon hysteresis.  This is
        the main reason the result is reported and not trusted.
    ``effort_margin``
        Same idea for the effort channel.  Defaults to three times the
        baseline's effort noise when the baseline supplies one.
    """
    evidence = JawEvidence(
        commanded_rad=float(commanded_rad),
        measured_rad=None if measured_rad is None else float(measured_rad),
        effort=None if effort is None else float(abs(effort)),
    )

    if evidence.measured_rad is None:
        evidence.note = (
            "no jaw feedback on jaw/measured_js: nothing observed, nothing claimed"
        )
        return evidence

    evidence.residual_rad = evidence.measured_rad - evidence.commanded_rad

    if baseline is None:
        evidence.note = (
            "no empty-jaw baseline: residual has no reference. Run "
            "tools/calibrate_jaw.py to make this channel meaningful."
        )
        return evidence

    empty_residual = baseline.empty_close_rad - float(commanded_rad)
    evidence.residual_excess_rad = evidence.residual_rad - empty_residual

    margin = max(float(residual_margin_rad), 2.0 * float(baseline.empty_close_rad_noise))
    angle_blocked = evidence.residual_excess_rad > margin

    effort_blocked = None
    if evidence.effort is not None and baseline.empty_close_effort is not None:
        evidence.effort_excess = evidence.effort - abs(baseline.empty_close_effort)
        if effort_margin is None:
            noise = baseline.empty_close_effort_noise
            effort_margin = (
                3.0 * abs(noise) if noise
                else 0.25 * max(abs(baseline.empty_close_effort), 1e-6)
            )
        effort_blocked = evidence.effort_excess > float(effort_margin)

    evidence.jaw_blocked = bool(angle_blocked or bool(effort_blocked))

    parts = [
        f"jaw stopped {np.degrees(evidence.residual_excess_rad):+.2f} deg past the "
        f"empty-jaw stop (margin {np.degrees(margin):.2f} deg)"
    ]
    if effort_blocked is None:
        parts.append("no effort reference")
    else:
        parts.append(
            f"effort {evidence.effort_excess:+.4f} over the empty-jaw draw"
        )
    parts.append("OBSERVED ONLY - the grasp is not verified")
    evidence.note = "; ".join(parts)
    return evidence


@dataclass
class JawEvidenceWindow:
    """Requires the same reading to persist, so one noisy sample cannot decide."""

    required_streak: int = 3
    streak: int = 0
    last: Optional[JawEvidence] = None
    history: list = field(default_factory=list)
    #: keep the trace bounded on a long hold
    max_history: int = 400

    def update(self, evidence: JawEvidence) -> "JawEvidenceWindow":
        self.last = evidence
        if len(self.history) < self.max_history:
            self.history.append(evidence)
        if evidence.jaw_blocked:
            self.streak += 1
        else:
            # None (unknown) also breaks the streak: unknown is not evidence.
            self.streak = 0
        return self

    @property
    def blocked_streak_met(self) -> bool:
        return self.streak >= self.required_streak

    def reset(self):
        self.streak = 0
        self.last = None
        self.history = []

    def summary(self) -> dict:
        blocked = sum(1 for e in self.history if e.jaw_blocked)
        unknown = sum(1 for e in self.history if e.jaw_blocked is None)
        return {
            "samples": len(self.history),
            "blocked_samples": blocked,
            "unknown_samples": unknown,
            "current_streak": self.streak,
            "required_streak": self.required_streak,
            "grasp_verified": False,
            "last": None if self.last is None else self.last.as_dict(),
        }
