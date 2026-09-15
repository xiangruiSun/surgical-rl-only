"""Loading the RL checkpoint, for the two ways it goes wrong before torch."""

from pathlib import Path

import pytest

from surgicai_rl_deploy.policy import ApproachPolicy


def test_a_missing_checkpoint_is_reported_before_importing_torch(tmp_path):
    """A wrong path used to surface as ModuleNotFoundError for
    stable_baselines3, which sends you installing gigabytes to fix a typo."""
    with pytest.raises(FileNotFoundError) as exc:
        ApproachPolicy.load(tmp_path / "nope.zip")
    assert "checkpoint not found" in str(exc.value)


def test_a_relative_path_says_what_it_resolved_against(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        ApproachPolicy.load("models/rl/missing.zip")
    message = str(exc.value)
    assert "resolved against" in message
    assert "../models/rl/" in message


def test_an_absolute_path_does_not_get_the_relative_hint(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        ApproachPolicy.load(tmp_path / "abs.zip")
    assert "resolved against" not in str(exc.value)
