"""The command-line hook: a needle observation becoming ``--grasp-pos``.

Everything downstream of this -- the plan, the feasibility precheck, the safety
caps, the trace -- sees an ordinary grasp position and knows nothing about
perception.  That is the contract these tests defend.
"""

import argparse
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from surgicai_rl_deploy.calib import cli as calib_cli
from surgicai_rl_deploy.calib.bernstein import BernsteinBasis
from surgicai_rl_deploy.calib.fit import fit_dataset
from surgicai_rl_deploy.calib.simulate import SyntheticWorld, make_dataset


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp-pos", nargs=3, type=float, default=None)
    ap.add_argument("--grasp-quat", nargs=4, type=float, default=None)
    ap.add_argument("--goal-orientation", default=None)
    ap.add_argument("--strict", action="store_true")
    calib_cli.add_arguments(ap)
    return ap


@pytest.fixture
def world_and_model(tmp_path):
    world = SyntheticWorld()
    ds = make_dataset(world, 40, seed=0, n_grasps=3).usable()
    model = fit_dataset(
        ds, BernsteinBasis(1, "total"),
        metadata={"validated": True, "cv_rmse_mm": 0.5, "cv_baseline_mm": 3.9},
    )
    return world, ds, model.save(tmp_path / "cal.json")


def needle_file(world, tmp_path, seed=42, frames=5, name="needle.json"):
    rng = np.random.default_rng(seed)
    T = world.place_needle(rng)
    path = tmp_path / name
    path.write_text(
        json.dumps([{"T": world.estimate(T, rng).tolist()} for _ in range(frames)])
    )
    return T, path


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_inline_seven_numbers(tmp_path):
    poses = calib_cli.load_needle_pose("0.01,0.02,0.08,0,0,0,1")
    assert len(poses) == 1
    assert np.allclose(poses[0][:3, 3], [0.01, 0.02, 0.08])
    assert np.allclose(poses[0][:3, :3], np.eye(3))


def test_inline_rejects_the_wrong_count():
    with pytest.raises(ValueError, match="seven numbers"):
        calib_cli.load_needle_pose("0.01,0.02,0.08")


def test_a_bare_4x4_is_not_mistaken_for_four_records(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 0.08]
    p = tmp_path / "one.json"
    p.write_text(json.dumps(T.tolist()))
    poses = calib_cli.load_needle_pose(str(p))
    assert len(poses) == 1 and np.allclose(poses[0], T)


def test_position_and_quaternion_records(tmp_path):
    p = tmp_path / "pq.json"
    p.write_text(json.dumps([
        {"position": [0.0, 0.0, 0.08], "quaternion": [0, 0, 0, 1]},
        {"position": [0.001, 0.0, 0.08], "quaternion": [0, 0, 0, 1]},
    ]))
    assert len(calib_cli.load_needle_pose(str(p))) == 2


def test_an_empty_file_is_refused(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("[]")
    with pytest.raises(ValueError, match="no needle poses"):
        calib_cli.load_needle_pose(str(p))


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def test_fills_in_grasp_pos_from_a_needle_pose(tmp_path, world_and_model):
    world, ds, model_path = world_and_model
    T_true, needle = needle_file(world, tmp_path)
    args = parser().parse_args(
        ["--needle-pose", str(needle), "--grasp-calibration", str(model_path)]
    )
    assert args.grasp_pos is None
    target, messages = calib_cli.apply_to_args(args)
    assert target is not None and target.report.ok
    assert args.grasp_pos is not None and len(args.grasp_pos) == 3
    assert np.allclose(args.grasp_pos, target.p_target_m)
    assert any("correction" in m for m in messages)


def test_the_filled_in_position_is_close_to_the_true_grasp(tmp_path, world_and_model):
    world, ds, model_path = world_and_model
    T_true, needle = needle_file(world, tmp_path, seed=77)
    args = parser().parse_args(
        ["--needle-pose", str(needle), "--grasp-calibration", str(model_path)]
    )
    calib_cli.apply_to_args(args)
    truth = world.true_grasp_pose(T_true, ds.grasp_point, np.random.default_rng(0))[:3, 3]
    assert np.linalg.norm(np.array(args.grasp_pos) - truth) * 1000.0 < 1.5


def test_a_typed_grasp_pos_wins(tmp_path, world_and_model):
    world, _, model_path = world_and_model
    _, needle = needle_file(world, tmp_path)
    args = parser().parse_args([
        "--grasp-pos", "0.1", "0.2", "0.3",
        "--needle-pose", str(needle), "--grasp-calibration", str(model_path),
    ])
    _, messages = calib_cli.apply_to_args(args)
    assert args.grasp_pos == [0.1, 0.2, 0.3]
    assert any("typed pose wins" in m for m in messages)


def test_no_needle_pose_is_not_an_error():
    args = parser().parse_args(["--grasp-pos", "0.1", "0.2", "0.3"])
    target, messages = calib_cli.apply_to_args(args)
    assert target is None and messages == []
    assert args.grasp_pos == [0.1, 0.2, 0.3]


def test_a_refused_needle_leaves_grasp_pos_empty(tmp_path, world_and_model):
    """A refusal must not quietly hand a number to the rest of the pipeline."""
    world, _, model_path = world_and_model
    rng = np.random.default_rng(1)
    T = world.place_needle(rng)
    T[0, 3] += 0.08                                   # far outside the box
    p = tmp_path / "far.json"
    p.write_text(json.dumps([{"T": world.estimate(T, rng).tolist()}]))
    args = parser().parse_args(
        ["--needle-pose", str(p), "--grasp-calibration", str(model_path)]
    )
    target, _ = calib_cli.apply_to_args(args)
    assert not target.report.ok
    assert args.grasp_pos is None


# ---------------------------------------------------------------------------
# the uncalibrated path
# ---------------------------------------------------------------------------
def test_refuses_a_needle_pose_with_no_calibration(tmp_path, world_and_model):
    world, _, _ = world_and_model
    _, needle = needle_file(world, tmp_path)
    args = parser().parse_args(["--needle-pose", str(needle)])
    with pytest.raises(ValueError, match="no empirical correction"):
        calib_cli.apply_to_args(args)


def test_uncalibrated_is_allowed_deliberately(tmp_path, world_and_model):
    world, _, _ = world_and_model
    _, needle = needle_file(world, tmp_path)
    args = parser().parse_args(
        ["--needle-pose", str(needle), "--allow-uncalibrated-needle"]
    )
    target, _ = calib_cli.apply_to_args(args)
    assert np.abs(target.correction_mm).max() == 0.0
    assert np.allclose(target.p_target_m, target.p_nominal_m)


def test_needle_point_override_moves_the_target(tmp_path, world_and_model):
    """Overriding the convention must change the answer by the lever arm --
    and, because it also changes the convention digest, must be refused."""
    world, _, model_path = world_and_model
    _, needle = needle_file(world, tmp_path)
    base = parser().parse_args(
        ["--needle-pose", str(needle), "--grasp-calibration", str(model_path)]
    )
    calib_cli.apply_to_args(base)

    override = parser().parse_args([
        "--needle-pose", str(needle), "--grasp-calibration", str(model_path),
        "--needle-point", "mesh_origin",
    ])
    target, _ = calib_cli.apply_to_args(override)
    assert not target.report.ok                       # the digest no longer matches
    moved = np.linalg.norm(target.p_nominal_m - np.array(base.grasp_pos)
                           + target.correction_m)
    assert moved > 0.005


def test_a_supplied_grasp_quat_is_checked_against_the_taught_wrist(
    tmp_path, world_and_model
):
    world, _, model_path = world_and_model
    _, needle = needle_file(world, tmp_path)
    turned = (Rotation.from_rotvec(world.gripper_rotvec)
              * Rotation.from_rotvec([0, 0, np.deg2rad(45)])).as_quat()
    args = parser().parse_args(
        ["--needle-pose", str(needle), "--grasp-calibration", str(model_path),
         "--grasp-quat", *[str(v) for v in turned]]
    )
    target, _ = calib_cli.apply_to_args(args)
    check = [c for c in target.report.checks if c.name == "calibration.orientation"][0]
    assert check.status == "warn"


def test_max_correction_ceiling_is_honoured(tmp_path, world_and_model):
    world, _, model_path = world_and_model
    _, needle = needle_file(world, tmp_path)
    args = parser().parse_args([
        "--needle-pose", str(needle), "--grasp-calibration", str(model_path),
        "--max-correction-mm", "0.05",
    ])
    target, _ = calib_cli.apply_to_args(args)
    assert not target.report.ok
    assert any(c.name == "calibration.bound" and c.status == "fail"
               for c in target.report.checks)
