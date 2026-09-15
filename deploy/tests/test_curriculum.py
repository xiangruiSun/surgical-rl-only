"""The randomisation curriculum. Pure Python, so no AMBF is needed."""

import sys
from pathlib import Path

import numpy as np
import pytest

SIM_RL = Path(__file__).resolve().parents[2] / "src" / "SurgicAI" / "RL"
sys.path.insert(0, str(SIM_RL))

from curriculum import (  # noqa: E402
    CERTIFIED_THRESHOLD,
    TRAIN_MAX_EPISODE_STEPS,
    Curriculum,
    Dimension,
    approach_curriculum,
    training_env_kwargs,
)


# ======================================================================
# the training contract the curriculum trains against
# ======================================================================
def test_training_kwargs_match_the_training_scripts():
    """1.0 mm / 3 deg / 300 steps, not Env_info's demonstration scale."""
    kw = training_env_kwargs()
    np.testing.assert_allclose(kw["step_size"][:3], 1.0e-3, rtol=1e-6)
    np.testing.assert_allclose(np.degrees(kw["step_size"][3:6]), 3.0, rtol=1e-6)
    assert kw["step_size"][6] == pytest.approx(0.05)
    assert kw["max_episode_step"] == TRAIN_MAX_EPISODE_STEPS == 300


def test_the_default_threshold_is_the_certified_one():
    kw = training_env_kwargs()
    assert kw["threshold"][0] == pytest.approx(CERTIFIED_THRESHOLD[0])
    assert np.degrees(kw["threshold"][1]) == pytest.approx(30.0)


def test_a_tighter_threshold_can_be_asked_for():
    kw = training_env_kwargs(threshold=(0.1, np.deg2rad(5.0)))
    assert kw["threshold"][0] == pytest.approx(0.1)
    assert np.degrees(kw["threshold"][1]) == pytest.approx(5.0)


# ======================================================================
# one dimension at a time
# ======================================================================
def simple(levels=2, **kw):
    return Curriculum(
        dimensions=[
            Dimension("r", 0, "first", 0.0, 1.0),
            Dimension("r", 1, "second", 0.0, 2.0),
        ],
        levels=levels, patience=1, **kw,
    )


def test_only_the_current_dimension_is_widened():
    c = simple()
    c.record(1.0)  # promote to level 1 of dimension 0
    kw = c.env_kwargs()
    assert kw["r"][0] == pytest.approx(0.5)
    assert kw["r"][1] == pytest.approx(0.0), "the second dimension must not move yet"


def test_a_dimension_completes_before_the_next_starts():
    c = simple()
    assert c.stage == 0
    c.record(1.0)   # level 1
    assert (c.stage, c.level) == (0, 1)
    c.record(1.0)   # level 2 == levels
    assert (c.stage, c.level) == (0, 2)
    v = c.record(1.0)  # dimension done -> next stage
    assert v.action == "promote" and c.stage == 1 and c.level == 0
    kw = c.env_kwargs()
    assert kw["r"][0] == pytest.approx(1.0), "the finished dimension stays at its end"
    assert kw["r"][1] == pytest.approx(0.0)


def test_success_below_the_bar_holds_instead_of_widening():
    c = simple()
    v = c.record(0.5)
    assert v.action == "hold" and not v.changed
    assert c.level == 0


def test_patience_delays_promotion():
    c = Curriculum(
        dimensions=[Dimension("r", 0, "only", 0.0, 1.0)], levels=2, patience=3
    )
    assert c.record(1.0).action == "hold"
    assert c.record(1.0).action == "hold"
    assert c.record(1.0).action == "promote"


# ======================================================================
# the 10% rule
# ======================================================================
def test_a_costly_widening_freezes_the_dimension_and_moves_on():
    """Xie et al.: widen until performance degrades by more than 10%, then stop."""
    c = simple(levels=4)
    c.record(0.95)              # baseline established at level 0
    assert c.level == 1
    v = c.record(0.70)          # 25 points worse than the best
    assert v.action == "regress"
    assert v.changed
    assert c.frozen[0] == 0, "it should fall back to the last width that worked"
    assert c.stage == 1, "and move on to the next dimension"
    assert c.env_kwargs()["r"][0] == pytest.approx(0.0)


def test_a_small_dip_is_tolerated():
    c = simple(levels=4)
    c.record(0.95)
    v = c.record(0.90)          # 5 points, inside the margin
    assert v.action in ("promote", "hold")
    assert 0 not in c.frozen


def test_the_margin_is_configurable():
    c = simple(levels=4, regress_margin=0.02)
    c.record(0.95)
    assert c.record(0.90).action == "regress"


def test_the_frozen_width_is_kept_for_the_rest_of_the_run():
    c = simple(levels=4)
    c.record(1.0)               # level 1  (0.25)
    c.record(1.0)               # level 2  (0.50)
    c.record(0.2)               # collapse -> freeze at level 1
    assert c.frozen[0] == 1
    assert c.env_kwargs()["r"][0] == pytest.approx(0.25)
    # and it stays there while the next dimension is worked on
    c.record(1.0)
    assert c.env_kwargs()["r"][0] == pytest.approx(0.25)


def test_the_curriculum_finishes():
    c = Curriculum(dimensions=[Dimension("r", 0, "only", 0.0, 1.0)], levels=1,
                   patience=1)
    c.record(1.0)               # to level 1
    c.record(1.0)               # dimension complete
    assert c.done
    v = c.record(1.0)
    assert v.action == "done" and not v.changed


# ======================================================================
# validation
# ======================================================================
def test_a_dimension_cannot_exceed_a_stated_envelope():
    with pytest.raises(ValueError, match="envelope"):
        Dimension("r", 0, "needle yaw", 0.0, np.deg2rad(60.0), " rad",
                  envelope=np.deg2rad(30.0))


def test_a_dimension_cannot_run_backwards():
    with pytest.raises(ValueError, match="below start"):
        Dimension("r", 0, "bad", 1.0, 0.0)


@pytest.mark.parametrize("bad", [{"levels": 0}, {"promote_at": 0.0},
                                 {"promote_at": 1.5}])
def test_bad_curricula_are_refused(bad):
    with pytest.raises(ValueError):
        Curriculum(dimensions=[Dimension("r", 0, "x", 0.0, 1.0)], **bad)


def test_an_empty_curriculum_is_refused():
    with pytest.raises(ValueError, match="at least one dimension"):
        Curriculum(dimensions=[])


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_an_impossible_success_rate_is_refused(rate):
    with pytest.raises(ValueError, match="success rate"):
        simple().record(rate)


# ======================================================================
# the concrete Approach plan
# ======================================================================
def test_the_approach_plan_starts_with_the_start_pose():
    """The demos contain one distinct start pose; that is the free win."""
    c = approach_curriculum()
    assert c.dimensions[0].kwarg == "psm_reset_random_range"
    assert [d.kwarg for d in c.dimensions[:6]] == ["psm_reset_random_range"] * 6
    assert [d.kwarg for d in c.dimensions[6:]] == ["needle_random_range"] * 3


def test_the_approach_plan_begins_at_no_start_randomisation():
    c = approach_curriculum()
    kw = c.env_kwargs()
    np.testing.assert_allclose(kw["psm_reset_random_range"], 0.0, atol=1e-12)


def test_needle_yaw_is_capped_by_the_perception_envelope():
    """needle_reset_ranges.py documents +-30 deg as a pose-audit limit."""
    c = approach_curriculum()
    yaw = c.dimensions[-1]
    assert yaw.name == "needle yaw"
    assert yaw.envelope is not None
    assert yaw.end <= yaw.envelope + 1e-12
    assert np.degrees(yaw.envelope) == pytest.approx(30.0, abs=1e-6)


def test_a_wider_envelope_can_be_asked_for_explicitly():
    c = approach_curriculum(needle_yaw_envelope_rad=np.deg2rad(45.0))
    yaw = c.dimensions[-1]
    assert np.degrees(yaw.end) == pytest.approx(45.0, abs=1e-6)


def test_the_plan_runs_to_completion_under_a_perfect_learner():
    c = approach_curriculum(levels=2)
    for _ in range(200):
        if c.record(1.0).action == "done":
            break
    assert c.done
    kw = c.env_kwargs()
    # every dimension ended at its end value
    for i, dim in enumerate(c.dimensions):
        assert kw[dim.kwarg][dim.index] == pytest.approx(dim.end)


def test_the_summary_records_where_each_dimension_stopped():
    c = approach_curriculum(levels=2)
    c.record(1.0)
    c.record(0.1)          # collapse on the first dimension
    s = c.summary()
    assert s["dimensions"][0]["frozen_at_level"] == 0
    assert s["stage"] == 1
    assert s["evaluations"] == 2
