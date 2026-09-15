"""Fail-closed precheck, run before a single command is published.

Why this file exists
--------------------
The R6 checkpoint is a *single-goal* policy: 50 demonstrations inside a
+-3 mm / +-15 deg box.  Anything outside that is out of distribution, and the
offline replays in ``docs``/the project findings show the policy orbiting the
goal rather than reaching it.  "Extending the workspace" for an RL policy means
retraining.

The servo does not have that problem -- it is geometry, not a learned map, and
it works anywhere the arm can physically go.  So the workspace question stops
being "is this inside the trained region" and becomes "is this inside the
*reachable and safe* region", which is a set of checks that can be evaluated
up front, on numbers, before the arm moves.  That is what this module does.

Every check returns ``pass``, ``warn`` or ``fail``.  One ``fail`` and the
episode does not start.  A ``warn`` is printed, recorded in the trace, and the
episode proceeds -- with ``--strict`` every warning becomes a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .contract import (
    R6_START_OFFSET_TOOL_MAX,
    R6_START_OFFSET_TOOL_MIN,
    R6_START_ROT_DEG_MAX,
    R6_START_ROT_DEG_MIN,
    SUPPORT_EPS_CM,
)
from .jaw import JawBaseline
from .plan import GraspLiftPlan

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    message: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass
class PrecheckReport:
    checks: list = field(default_factory=list)
    strict: bool = False

    def add(self, name, status, message, **detail):
        self.checks.append(Check(name, status, message, detail))

    @property
    def failures(self):
        bad = [c for c in self.checks if c.status == FAIL]
        if self.strict:
            bad = bad + [c for c in self.checks if c.status == WARN]
        return bad

    @property
    def warnings(self):
        return [c for c in self.checks if c.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        glyph = {PASS: "  ok  ", WARN: " WARN ", FAIL: " FAIL "}
        lines = [f"[{glyph[c.status]}] {c.name}: {c.message}" for c in self.checks]
        lines.append("")
        if self.ok:
            lines.append(
                f"PRECHECK PASS ({len(self.warnings)} warning(s))"
                if self.warnings
                else "PRECHECK PASS"
            )
        else:
            lines.append(f"PRECHECK FAIL ({len(self.failures)} blocking)")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "strict": self.strict,
            "checks": [c.as_dict() for c in self.checks],
        }


def precheck(
    plan: GraspLiftPlan,
    *,
    controller: str = "d2",
    execute: bool = False,
    grasp_gate: str = "manual",
    jaw_baseline: Optional[JawBaseline] = None,
    max_path_radius_cm: float = 8.0,
    limit_low_m=None,
    limit_high_m=None,
    max_step_translation_mm: float = 2.5,
    max_step_rotation_deg: float = 5.0,
    approach_max_steps: int = 200,
    lift_max_steps: int = 120,
    success_trans_cm: float = 1.0,
    success_rot_deg: float = 10.0,
    strict: bool = False,
) -> PrecheckReport:
    report = PrecheckReport(strict=strict)

    # -- 1. the numbers themselves -----------------------------------------
    pts = np.stack([plan.start.p, plan.grasp.p, plan.lifted.p])
    if not np.isfinite(pts).all():
        report.add("inputs", FAIL, "a waypoint is not finite")
        return report
    report.add(
        "inputs",
        PASS,
        f"approach {plan.approach_travel_cm:.2f} cm / "
        f"{plan.approach_rotation_deg:.1f} deg, lift {plan.lift_travel_cm:.2f} cm",
        start_cm=(plan.start.p * 100).tolist(),
        grasp_cm=(plan.grasp.p * 100).tolist(),
        lifted_cm=(plan.lifted.p * 100).tolist(),
    )

    # -- 2. the lift direction is a human decision -------------------------
    if plan.lift_spec.explicit:
        report.add(
            "lift_direction",
            PASS,
            f"operator-confirmed: {plan.lift_spec.describe()}",
            direction=plan.lift_spec.direction(plan.grasp).tolist(),
        )
    else:
        report.add(
            "lift_direction",
            FAIL if execute else WARN,
            "the lift sign was not stated on the command line. Pass --lift-sign "
            "+1 or -1 after checking, in the scene, which way moves the gripper "
            "AWAY from the tissue. A wrong sign drives the needle into the pad.",
            direction=plan.lift_spec.direction(plan.grasp).tolist(),
        )

    # -- 3. does the lift continue into the approach direction? ------------
    approach_vec = plan.grasp.p - plan.start.p
    approach_norm = float(np.linalg.norm(approach_vec))
    lift_dir = plan.lift_spec.direction(plan.grasp)
    if approach_norm > 1e-6:
        cos = float(np.dot(lift_dir, approach_vec / approach_norm))
        if cos > 0.5:
            report.add(
                "lift_vs_approach",
                WARN,
                f"the lift points {np.degrees(np.arccos(np.clip(cos, -1, 1))):.0f} deg "
                "from the approach direction, i.e. it keeps going the way the "
                "gripper came in. If the needle is lying on tissue, that is into "
                "the tissue. Check the sign.",
                cos=cos,
            )
        else:
            report.add(
                "lift_vs_approach",
                PASS,
                f"the lift turns {np.degrees(np.arccos(np.clip(cos, -1, 1))):.0f} deg "
                "away from the approach direction",
                cos=cos,
            )
    else:
        report.add(
            "lift_vs_approach", WARN, "start and grasp coincide; no approach direction"
        )

    # -- 4. reachability, as a radius around the measured start ------------
    radius = plan.path_radius_cm()
    if radius > max_path_radius_cm:
        report.add(
            "path_radius",
            FAIL,
            f"the path reaches {radius:.2f} cm from the measured start pose, past "
            f"the {max_path_radius_cm:.1f} cm limit. Either the goal is wrong, is "
            "in the wrong frame, or this needs a planned motion rather than a "
            "servo. Raise --max-path-radius-cm only if you know the arm covers it.",
            radius_cm=radius,
        )
    else:
        report.add(
            "path_radius",
            PASS,
            f"every waypoint within {radius:.2f} cm of the start "
            f"(limit {max_path_radius_cm:.1f} cm)",
            radius_cm=radius,
        )

    # -- 5. the operator's own hard box ------------------------------------
    if limit_low_m is not None and limit_high_m is not None:
        low = np.asarray(limit_low_m, dtype=np.float64).reshape(3)
        high = np.asarray(limit_high_m, dtype=np.float64).reshape(3)
        if np.any(high <= low):
            report.add("hard_limits", FAIL, "--limit-high must exceed --limit-low")
        else:
            outside = [
                name
                for name, p in (
                    ("start", plan.start.p),
                    ("grasp", plan.grasp.p),
                    ("lifted", plan.lifted.p),
                )
                if np.any(p < low) or np.any(p > high)
            ]
            if outside:
                report.add(
                    "hard_limits",
                    FAIL,
                    f"outside the operator limit box: {', '.join(outside)}",
                    low_cm=(low * 100).tolist(),
                    high_cm=(high * 100).tolist(),
                )
            else:
                report.add(
                    "hard_limits",
                    PASS,
                    "all waypoints inside the operator limit box",
                    low_cm=(low * 100).tolist(),
                    high_cm=(high * 100).tolist(),
                )
    else:
        report.add(
            "hard_limits",
            WARN if execute else PASS,
            "no operator limit box given (--limit-low/--limit-high). The only "
            "positional guard is the padded box around the waypoints.",
        )

    # -- 6. can the segments finish inside their step budgets? -------------
    step_m = max_step_translation_mm / 1000.0
    step_deg = max(max_step_rotation_deg, 1e-6)
    for seg, travel_cm, rot_deg, budget in (
        ("approach", plan.approach_travel_cm, plan.approach_rotation_deg, approach_max_steps),
        ("lift", plan.lift_travel_cm, 0.0, lift_max_steps),
    ):
        # A proportional servo never moves a full step near the goal, so the
        # geometric minimum is doubled to leave convergence headroom.
        need = 2.0 * max(travel_cm / 100.0 / step_m, rot_deg / step_deg)
        if need > budget:
            report.add(
                f"step_budget_{seg}",
                FAIL,
                f"{seg} needs roughly {need:.0f} cycles at the current per-step "
                f"cap but the budget is {budget}. Raise the budget or the caps.",
                estimated_steps=need,
                budget=budget,
            )
        else:
            report.add(
                f"step_budget_{seg}",
                PASS,
                f"{seg} needs roughly {need:.0f} of {budget} cycles",
                estimated_steps=need,
                budget=budget,
            )

    # -- 7. success tolerance vs the lift itself ---------------------------
    if success_trans_cm >= plan.lift_travel_cm:
        report.add(
            "lift_tolerance",
            FAIL,
            f"the {success_trans_cm:.2f} cm success tolerance is at least as "
            f"large as the {plan.lift_travel_cm:.2f} cm lift: the lift would "
            "report success without moving. Use --lift-success-trans-cm.",
        )
    else:
        report.add(
            "lift_tolerance",
            PASS,
            f"success tolerance {success_trans_cm:.2f} cm is well inside the "
            f"{plan.lift_travel_cm:.2f} cm lift",
        )

    # -- 8. the grasp gate --------------------------------------------------
    if grasp_gate == "evidence":
        if jaw_baseline is None:
            report.add(
                "grasp_gate",
                FAIL,
                "--grasp-gate evidence needs an empty-jaw baseline to compare "
                "against. Run tools/calibrate_jaw.py, or use --grasp-gate manual.",
            )
        else:
            report.add(
                "grasp_gate",
                WARN,
                "the lift will be released by jaw evidence alone. That evidence "
                "says the jaw stopped early, which is NOT a confirmed grasp on "
                f"this hardware. Baseline source: {jaw_baseline.source}",
                baseline=jaw_baseline.as_dict(),
            )
    elif grasp_gate == "manual":
        report.add(
            "grasp_gate",
            PASS,
            "the arm will stop with the jaw closed and wait for a human to "
            "release the lift",
        )
    elif grasp_gate == "always":
        report.add(
            "grasp_gate",
            WARN if execute else PASS,
            "--grasp-gate always: the lift runs whether or not anything is in "
            "the jaws. Intended for dry runs and empty-gripper rehearsals.",
        )
    elif grasp_gate == "never":
        report.add(
            "grasp_gate", PASS, "the episode stops after the jaw closes; no lift"
        )
    else:
        report.add("grasp_gate", FAIL, f"unknown grasp gate {grasp_gate!r}")

    if jaw_baseline is None and grasp_gate != "evidence":
        report.add(
            "jaw_baseline",
            WARN,
            "no empty-jaw baseline loaded: jaw readings will be logged raw, with "
            "no reference for what an empty close looks like on this arm",
        )
    elif jaw_baseline is not None:
        report.add(
            "jaw_baseline",
            PASS,
            f"empty-jaw baseline loaded ({jaw_baseline.source})",
            baseline=jaw_baseline.as_dict(),
        )

    # -- 9. where the approach sits relative to the R6 training support ----
    dp_tool_cm = plan.start.R.T @ ((plan.grasp.p - plan.start.p) * 100.0)
    offenders = []
    axes = "xyz"
    for i in range(3):
        if (
            dp_tool_cm[i] < R6_START_OFFSET_TOOL_MIN[i] - SUPPORT_EPS_CM
            or dp_tool_cm[i] > R6_START_OFFSET_TOOL_MAX[i] + SUPPORT_EPS_CM
        ):
            offenders.append(
                f"tool-{axes[i]} {dp_tool_cm[i]:+.2f} cm outside "
                f"[{R6_START_OFFSET_TOOL_MIN[i]:+.2f}, {R6_START_OFFSET_TOOL_MAX[i]:+.2f}]"
            )
    rot_deg = plan.approach_rotation_deg
    if not (R6_START_ROT_DEG_MIN <= rot_deg <= R6_START_ROT_DEG_MAX):
        offenders.append(
            f"start->goal rotation {rot_deg:.1f} deg outside "
            f"[{R6_START_ROT_DEG_MIN:.1f}, {R6_START_ROT_DEG_MAX:.1f}]"
        )

    if controller in ("rl", "residual"):
        if offenders:
            report.add(
                "r6_training_support",
                FAIL if strict else WARN,
                "the approach is outside the R6 demonstration support, where the "
                "policy has been measured to orbit the goal rather than reach it: "
                + "; ".join(offenders),
                offenders=offenders,
                start_offset_tool_cm=dp_tool_cm.tolist(),
            )
        else:
            report.add(
                "r6_training_support",
                PASS,
                "the approach sits inside the R6 demonstration support",
                start_offset_tool_cm=dp_tool_cm.tolist(),
            )
    else:
        report.add(
            "r6_training_support",
            PASS,
            f"controller '{controller}' is geometric, so the R6 trained region "
            "does not bound it"
            + (f" (for reference, the RL support check would flag: {'; '.join(offenders)})"
               if offenders else ""),
            offenders=offenders,
            start_offset_tool_cm=dp_tool_cm.tolist(),
            applies=False,
        )

    # -- 10. the jaw itself -------------------------------------------------
    report.add("jaw_calibration", PASS, plan.jaw.describe(), **plan.jaw.as_dict())

    return report
