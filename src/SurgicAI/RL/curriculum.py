"""A randomisation curriculum for widening a subtask's trained support.

Why this is shaped the way it is
--------------------------------
Three findings drive the design, two from this repository's own measurements
and one from the sim2real literature.

1. **For the approach leg, the trained support is not the binding constraint.**
   A goal-conditioned policy's support is expressed in the tool-frame offset
   from start to goal and the geodesic rotation between them, and both are
   invariant under any rigid transform.  ``deploy/tools/workspace_spec.py``
   measures the consequence: over the +-3 mm / +-30 deg needle envelope this
   repository assumes, staging the arm puts **400/400** samples in support,
   against **0/400** unstaged, with a 3.7 cm median staging move.  Retraining
   for a wider *support* would not change what the deployment can do.

   So this curriculum is not for widening the support.  It is for the three
   things that retraining genuinely buys:

   * robustness to the arm not being exactly where it was told (the demos all
     start from a single pose -- 1 distinct start across 50 episodes)
   * a wider *needle* envelope, once perception can estimate one
   * a Place policy that works, since the released one reproduces 12/25 at its
     own published tolerance

2. **Widen one dimension at a time.**  Xie et al.'s analysis of randomisation
   effects on sim2real (arXiv:2206.06282) reports that all-at-once
   randomisation converged to a substantially lower return than the
   unrandomised baseline, and that sequential strategies "seem to lead to a
   more consistent real-world performance".  Their practical range-finding
   heuristic is adopted directly below: increase a range until performance
   degrades by more than 10%, then stop.

3. **The perception envelope is the real cap.**  ``needle_reset_ranges.py`` in
   this repository documents its +-30 deg yaw bound as a pose-audit limit --
   reliable at 20 deg, near-180-degree failures at 40 -- not a policy limit.
   Training over a needle yaw that perception cannot estimate produces a policy
   that is correct about a goal nobody can supply.  The curriculum refuses to
   exceed a stated perception envelope unless explicitly overridden.

Nothing here imports AMBF, so it is unit-testable without a simulator.

Usage
-----
    curriculum = approach_curriculum()
    env = gym.make("approach", **curriculum.env_kwargs(), **training_env_kwargs())
    ...
    verdict = curriculum.record(success_rate)     # after each eval block
    if verdict.changed:
        env = gym.make("approach", **curriculum.env_kwargs(), ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from needle_reset_ranges import (
    ASSUMED_REAL_NEEDLE_RANGE,
    CURRICULUM_END_NEEDLE_RANGE,
    CURRICULUM_START_NEEDLE_RANGE,
)

# --- the verified training contract --------------------------------------
# RL/RL_training_online.py and RL/Model_evaluation.py both hard-code these.
# The 0.5 mm / 2 deg in RL/Env_info is the scale the *demonstrations* were
# collected at, not the scale the policy acts at; driving the released
# Approach checkpoint at 0.5 mm halves its success rate (7/20 against 19/20).
TRAIN_TRANS_STEP_M = 1.0e-3
TRAIN_ANGLE_STEP_RAD = float(np.deg2rad(3.0))
TRAIN_JAW_STEP = 0.05
TRAIN_MAX_EPISODE_STEPS = 300

#: the tolerance every env class carries, and the one the published success
#: rates were measured at
CERTIFIED_THRESHOLD = (0.5, float(np.deg2rad(30.0)))
#: the tighter pair in RL/Env_info, which is what a real grasp needs
TIGHT_THRESHOLD = (0.3, float(np.deg2rad(10.0)))


def training_env_kwargs(threshold=None, max_episode_steps=None) -> dict:
    """Step size and budget matching what the released checkpoints used."""
    trans, angle = threshold or CERTIFIED_THRESHOLD
    return {
        "step_size": np.array(
            [TRAIN_TRANS_STEP_M] * 3 + [TRAIN_ANGLE_STEP_RAD] * 3 + [TRAIN_JAW_STEP],
            dtype=np.float32,
        ),
        "threshold": np.array([trans, angle], dtype=np.float32),
        "max_episode_step": int(max_episode_steps or TRAIN_MAX_EPISODE_STEPS),
    }


# --- the curriculum -------------------------------------------------------
@dataclass(frozen=True)
class Dimension:
    """One axis of randomisation, widened on its own."""

    #: the env kwarg this belongs to, e.g. "psm_reset_random_range"
    kwarg: str
    #: index within that array
    index: int
    name: str
    start: float
    end: float
    unit: str = ""
    #: hard ceiling imposed by something outside the policy (e.g. perception).
    #: None means the only ceiling is ``end``.
    envelope: Optional[float] = None

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"{self.name}: end {self.end} is below start {self.start}")
        if self.envelope is not None and self.end > self.envelope + 1e-12:
            raise ValueError(
                f"{self.name}: end {self.end}{self.unit} exceeds the stated "
                f"envelope {self.envelope}{self.unit}. Widen the envelope first, "
                "and say why it is safe to, or lower end."
            )

    def value_at(self, fraction: float) -> float:
        f = float(np.clip(fraction, 0.0, 1.0))
        return float(self.start + (self.end - self.start) * f)

    def describe(self, fraction: float) -> str:
        return f"{self.name} {self.value_at(fraction):.4g}{self.unit}"


@dataclass
class Verdict:
    action: str  # "promote" | "hold" | "regress" | "done"
    reason: str
    changed: bool
    stage: int
    level: int


@dataclass
class Curriculum:
    """Widen one dimension at a time, and step back when it stops working.

    ``levels`` steps take a dimension from ``start`` to ``end``.  A dimension is
    only widened once the policy is succeeding at the current width; if
    widening costs more than ``regress_margin`` of success relative to the best
    seen on that dimension, the curriculum steps back and freezes it, which is
    Xie et al.'s "increase until performance degrades by more than 10%, then
    stop" turned into a rule.
    """

    dimensions: list
    levels: int = 4
    promote_at: float = 0.80
    regress_margin: float = 0.10
    #: evaluations to see at a level before it may be promoted
    patience: int = 2

    stage: int = 0          # which dimension is being widened
    level: int = 0          # 0..levels, position within that dimension
    frozen: dict = field(default_factory=dict)   # stage -> level it froze at
    history: list = field(default_factory=list)
    _best: float = 0.0
    _seen: int = 0

    def __post_init__(self):
        if not self.dimensions:
            raise ValueError("a curriculum needs at least one dimension")
        if self.levels < 1:
            raise ValueError("levels must be at least 1")
        if not 0.0 < self.promote_at <= 1.0:
            raise ValueError("promote_at must be in (0, 1]")

    # -- state -------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.stage >= len(self.dimensions)

    def fraction(self, stage: int) -> float:
        """How far along dimension ``stage`` the curriculum currently sits."""
        if stage < self.stage:
            return self.frozen.get(stage, self.levels) / self.levels
        if stage > self.stage:
            return 0.0
        return self.level / self.levels

    def env_kwargs(self) -> dict:
        """Randomisation ranges for the current position, as env kwargs."""
        out: dict = {}
        for i, dim in enumerate(self.dimensions):
            array = out.setdefault(dim.kwarg, {})
            array[dim.index] = dim.value_at(self.fraction(i))
        # turn the sparse maps into dense arrays, leaving unlisted entries at
        # whatever the caller's defaults are by returning only what we set
        return {
            kwarg: _dense(values) for kwarg, values in out.items()
        }

    def describe(self) -> str:
        if self.done:
            return "curriculum complete"
        dim = self.dimensions[self.stage]
        return (
            f"stage {self.stage + 1}/{len(self.dimensions)} "
            f"level {self.level}/{self.levels}: {dim.describe(self.fraction(self.stage))}"
        )

    # -- the rule ----------------------------------------------------------
    def record(self, success_rate: float) -> Verdict:
        """Feed one evaluation block's success rate; get the next move."""
        rate = float(success_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"success rate must be in [0, 1]; got {rate}")
        if self.done:
            return Verdict("done", "curriculum complete", False, self.stage, self.level)

        self.history.append({"stage": self.stage, "level": self.level, "rate": rate})
        self._seen += 1
        dim = self.dimensions[self.stage]

        if self.level == 0:
            # the baseline for this dimension: whatever it manages unwidened
            self._best = max(self._best, rate)

        if rate < self._best - self.regress_margin:
            frozen_at = max(self.level - 1, 0)
            self.frozen[self.stage] = frozen_at
            verdict = Verdict(
                "regress",
                f"{dim.name} cost {100*(self._best - rate):.0f} points of success "
                f"(best {100*self._best:.0f}%, now {100*rate:.0f}%). Freezing it at "
                f"{dim.describe(frozen_at / self.levels)} and moving on.",
                True, self.stage, frozen_at,
            )
            self._advance_stage()
            return verdict

        if rate < self.promote_at:
            return Verdict(
                "hold",
                f"{100*rate:.0f}% is below the {100*self.promote_at:.0f}% needed to "
                f"widen {dim.name}; training on at this width",
                False, self.stage, self.level,
            )

        if self._seen < self.patience:
            return Verdict(
                "hold",
                f"{100*rate:.0f}% is enough, but waiting for {self.patience} "
                "consistent evaluations before widening",
                False, self.stage, self.level,
            )

        self._best = max(self._best, rate)
        if self.level >= self.levels:
            self.frozen[self.stage] = self.levels
            verdict = Verdict(
                "promote",
                f"{dim.name} reached {dim.describe(1.0)} at {100*rate:.0f}%; "
                "moving to the next dimension",
                True, self.stage, self.levels,
            )
            self._advance_stage()
            return verdict

        self.level += 1
        self._seen = 0
        return Verdict(
            "promote",
            f"{100*rate:.0f}% at {dim.describe((self.level - 1) / self.levels)}; "
            f"widening to {dim.describe(self.fraction(self.stage))}",
            True, self.stage, self.level,
        )

    def _advance_stage(self):
        self.stage += 1
        self.level = 0
        self._seen = 0
        self._best = 0.0

    def summary(self) -> dict:
        return {
            "dimensions": [
                {
                    "name": d.name,
                    "kwarg": d.kwarg,
                    "index": d.index,
                    "start": d.start,
                    "end": d.end,
                    "reached": d.value_at(self.fraction(i)),
                    "frozen_at_level": self.frozen.get(i),
                    "unit": d.unit,
                }
                for i, d in enumerate(self.dimensions)
            ],
            "stage": self.stage,
            "level": self.level,
            "done": self.done,
            "evaluations": len(self.history),
        }


def _dense(values: dict) -> np.ndarray:
    size = max(values) + 1
    out = np.zeros(size, dtype=np.float32)
    for index, value in values.items():
        out[index] = value
    return out


# --- the concrete plans ---------------------------------------------------
def approach_curriculum(
    *,
    psm_trans_end_m: float = 0.010,
    psm_rot_end_rad: float = float(np.deg2rad(40.0)),
    needle_yaw_envelope_rad: Optional[float] = None,
    levels: int = 4,
) -> Curriculum:
    """The order to widen things in for the Approach subtask.

    Start-pose randomisation comes first and needle placement second, because
    the first is free -- nothing outside the simulator has to change for the
    policy to see a start pose it was not put at -- while the second is capped
    by what perception can estimate.  The demonstrations embedded in the
    released checkpoint contain exactly **one** distinct start pose across all
    fifty episodes, so this is the dimension with the most to gain.
    """
    envelope = (
        float(ASSUMED_REAL_NEEDLE_RANGE[2])
        if needle_yaw_envelope_rad is None else float(needle_yaw_envelope_rad)
    )
    needle_yaw_end = min(float(CURRICULUM_END_NEEDLE_RANGE[2]), envelope)

    return Curriculum(
        dimensions=[
            Dimension("psm_reset_random_range", 0, "psm reset x", 0.0,
                      psm_trans_end_m, " m"),
            Dimension("psm_reset_random_range", 1, "psm reset y", 0.0,
                      psm_trans_end_m, " m"),
            Dimension("psm_reset_random_range", 2, "psm reset z", 0.0,
                      psm_trans_end_m, " m"),
            Dimension("psm_reset_random_range", 3, "psm reset roll", 0.0,
                      psm_rot_end_rad, " rad"),
            Dimension("psm_reset_random_range", 4, "psm reset pitch", 0.0,
                      psm_rot_end_rad, " rad"),
            Dimension("psm_reset_random_range", 5, "psm reset yaw", 0.0,
                      psm_rot_end_rad, " rad"),
            Dimension("needle_random_range", 0, "needle x",
                      float(CURRICULUM_START_NEEDLE_RANGE[0]),
                      float(CURRICULUM_END_NEEDLE_RANGE[0]), " m"),
            Dimension("needle_random_range", 1, "needle y",
                      float(CURRICULUM_START_NEEDLE_RANGE[1]),
                      float(CURRICULUM_END_NEEDLE_RANGE[1]), " m"),
            Dimension("needle_random_range", 2, "needle yaw",
                      float(CURRICULUM_START_NEEDLE_RANGE[2]),
                      needle_yaw_end, " rad", envelope=envelope),
        ],
        levels=levels,
    )
