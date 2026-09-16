"""Frozen observation/action contract of the SurgicAI Approach checkpoints.

Every constant here was read back out of
``models/rl/r6_unified_single_goal_yaw15_seed1_final.zip`` or out of the
training sources (``RL/utils/gym_manager.py``, ``RL/subtask_env.py``,
``RL/utils/utils.py``).  Do not "clean up" the units: the observation mixes
centimetres (positions) with radians (orientation) with a normalised jaw, and
the action step size is in metres/radians.  That asymmetry is the training
contract, not a bug.

Contract summary
----------------
obs["achieved_goal"] : (7,)  [x_cm, y_cm, z_cm, roll, pitch, yaw, jaw_norm]
obs["desired_goal"]  : (7,)  same layout
obs["observation"]   : (21,) concat(achieved, desired, desired - achieved)
action               : (7,)  in [-1, 1], applied as
                             cmd_raw = measured_raw + action * STEP_SIZE_RAW
                             with translation in metres and rotation in radians.
"""

from __future__ import annotations

import numpy as np

# --- action scaling -------------------------------------------------------
# The package-wide fallback, used only when no checkpoint contract applies.
#
# This was 1.5 mm for most of this project's life, taken from
# RL/utils/utils.py :: default_step_size in THIS repository. It is wrong for
# every checkpoint measured so far. Upstream trains and evaluates at 1.0 mm /
# 3 deg (RL/RL_training_online.py, RL/Model_evaluation.py), and replaying each
# checkpoint from its own demonstrations says the same:
#
#     checkpoint   0.5 mm/2deg   1.0 mm/3deg   1.5 mm/3deg   2.0 mm/2deg
#     upstream       7/20 35%     19/20 95%         --            --
#     R6            24/50 48%     46/50 92%     36/50 72%     10/50 20%
#
# So 1.5 mm was 50% over scale on R6, which is what made a working policy
# track for a few steps and then orbit the goal. Reproduce with
# tools/replay_demos.py --compare.
STEP_SIZE_RAW = np.array(
    [1.0e-3, 1.0e-3, 1.0e-3, np.deg2rad(3.0), np.deg2rad(3.0), np.deg2rad(3.0), 0.05],
    dtype=np.float32,
)

#: what this package used to apply, kept so the regression can be measured
LEGACY_STEP_SIZE_RAW = np.array(
    [1.5e-3, 1.5e-3, 1.5e-3, np.deg2rad(3.0), np.deg2rad(3.0), np.deg2rad(3.0), 0.05],
    dtype=np.float32,
)

# --- observation scaling --------------------------------------------------
# RL/utils/gym_manager.py :: normalize_observation
GOAL_SCALE = np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
OBS_SCALE = np.concatenate([GOAL_SCALE, GOAL_SCALE, GOAL_SCALE]).astype(np.float32)

# --- episode / success ----------------------------------------------------
# R6 single-goal evaluation contract: 1 cm translation, 10 deg rotation.
SUCCESS_TRANS_CM = 1.0
SUCCESS_ROT_RAD = float(np.deg2rad(10.0))
MAX_EPISODE_STEPS = 200

# --- command safety envelope (RL/subtask_env.py) --------------------------
# Expressed in the *training* (PSM base) frame, metres.  Kept here for
# reference; real-robot clamping is done in the ECM frame relative to the
# start pose, see loop.SafetyLimits.
TRAIN_WORKSPACE_LOW_M = np.array([-0.10, -0.10, -0.25], dtype=np.float32)
TRAIN_WORKSPACE_HIGH_M = np.array([0.10, 0.10, 0.05], dtype=np.float32)

# --- the goal the R6 checkpoint was actually trained on -------------------
# Mean of the 50 frozen single goals in the checkpoint's embedded demo set,
# in the training (PSM base) frame.  Units: metres + radians (xyz-Euler).
R6_TRAINED_GOAL_VEC7 = np.array(
    [-0.03017, 0.02265, -0.11890, -3.100, 0.902, 2.397, 0.0], dtype=np.float64
)

# Per-axis support of those 50 goals, in cm/rad, for the in-distribution report.
R6_GOAL_MIN = np.array([-3.297, 1.953, -12.229, -3.485, 0.852, 1.889, 0.0])
R6_GOAL_MAX = np.array([-2.716, 2.543, -11.441, -2.763, 0.927, 2.843, 0.0])

# Support of the *start* pose relative to the goal, measured over the 50
# demonstration episodes.  "tool" = expressed in the gripper frame at the
# episode start, i.e. R_start^T @ (p_goal - p_start).  Units cm.
R6_START_OFFSET_TOOL_MIN = np.array([-1.354, 0.948, 0.699])
R6_START_OFFSET_TOOL_MAX = np.array([3.211, 3.948, 4.415])
R6_START_OFFSET_TOOL_MEAN = np.array([0.960, 2.445, 2.661])

# Tolerance for testing membership of that box.  Its edges are the min and max
# of 50 samples, not a physical boundary, so a pose sitting exactly on a face
# must not be called out-of-support by floating-point noise.  Every place that
# tests the support imports this, so the answers cannot disagree.
SUPPORT_EPS_CM = 1.0e-9
# Geodesic rotation from start orientation to goal orientation, degrees.
R6_START_ROT_DEG_MIN = 25.7
R6_START_ROT_DEG_MAX = 100.2
R6_START_ROT_DEG_MEDIAN = 65.0

# Start offset expressed in the *goal* frame, R_goal^T @ (p_start - p_goal), cm.
R6_START_OFFSET_GOAL_MIN = np.array([-3.724, -2.637, -2.713])
R6_START_OFFSET_GOAL_MAX = np.array([-2.366, -0.091, -1.547])
R6_START_OFFSET_GOAL_MEAN = np.array([-3.142, -1.291, -2.177])

# Rotation the gripper performs from start to goal, as a rotation vector, in
# the training frame.  Mean over the 50 demonstration episodes (median
# magnitude 65 deg).  Used by goal_orientation="trained_relative".
R6_START_TO_GOAL_ROTVEC_MEAN = np.array([-0.5631, -0.8186, -0.0832])

# The single demonstration episode whose travel (3.29 cm) is closest to a
# 3 cm approach; handy as a concrete reference pair.
R6_REPRESENTATIVE_START_VEC7 = np.array(
    [-0.033072, 0.016730, -0.084081, -2.9896, 0.1754, 1.5404, 0.6012]
)
R6_REPRESENTATIVE_GOAL_VEC7 = np.array(
    [-0.027417, 0.021870, -0.116112, -3.4686, 0.8872, 1.9073, 0.0]
)

# The M3 checkpoint's embedded mean goal (RL/Approach_env.py).
M3_CHECKPOINT_GOAL_MEAN_RAW = np.array(
    [-0.032555993, 0.011748599, -0.11873012, -3.775371, 0.493273, 1.3117456, 0.0],
    dtype=np.float64,
)

# The Approach environment drives PSM2 in simulation (Approach_env.py sets
# self.psm_idx = 2).  Deploying on PSM1 is a deliberate substitution.
TRAINED_ARM = "PSM2"


# ==========================================================================
# Per-subtask contracts
# ==========================================================================
# One global STEP_SIZE_RAW was always a bug waiting to happen: SurgicAI ships a
# different action scale, success tolerance and episode budget for every
# subtask, in RL/Env_info/<task>_noise_env_info, and a checkpoint is only valid
# against its own.  Each block below was decoded twice -- once from that
# Env_info pickle, once from the demonstrations embedded in the released
# checkpoint -- and the two agree.
#
# Decode any checkpoint yourself with tools/profile_checkpoint.py.

from dataclasses import dataclass  # noqa: E402  (kept beside the contracts)
from typing import Optional  # noqa: E402


@dataclass(frozen=True)
class SubtaskContract:
    """Everything that must match between a checkpoint and the code driving it."""

    name: str
    #: cmd_raw = state_raw + action * step_size; metres / radians / normalised jaw.
    #:
    #: THIS IS NOT THE SCALE ``recover_step_size.py`` FINDS.  SurgicAI ships two
    #: different action scales and it is easy to take the wrong one:
    #:
    #:   * ``RL/Env_info/<task>_noise_env_info`` -- 0.5 mm / 2 deg for Approach.
    #:     This is the scale the *demonstrations embedded in the checkpoint* were
    #:     collected at, and it is what a least-squares fit against those
    #:     demonstrations recovers, exactly, to numerical zero.
    #:   * ``RL/RL_training_online.py`` and ``RL/Model_evaluation.py`` -- both
    #:     hard-code ``trans_step = 1.0e-3``, ``angle_step = deg2rad(3)``,
    #:     ``max_episode_steps = 300``.  This is the scale the *policy* was
    #:     trained and evaluated at, and therefore the only one it is valid to
    #:     act with.
    #:
    #: Replaying the upstream Approach checkpoint from its own demonstration
    #: starts, in the training frame, against a perfect arm:
    #:
    #:     0.5 mm / 2 deg (the demonstration scale)    7/20   35%
    #:     1.0 mm / 3 deg (the training scale)        19/20   95%   <- published 96 +- 6
    #:
    #: The demonstration scale reproduces the stored transitions perfectly and
    #: still runs the policy at half speed.  Fitting the demonstrations answers
    #: a question about the demonstrations, not about the policy.
    step_size: np.ndarray
    #: the scale the embedded demonstrations integrate at -- recoverable from
    #: them, and useful only for checking that the demonstrations parse
    demo_step_size: np.ndarray
    #: ||p_achieved - p_desired||, centimetres.
    #:
    #: This is the tolerance the policy was *certified* at, i.e. the one its
    #: published success rate was measured with -- the default written into the
    #: env class itself (``threshold = [0.5, np.deg2rad(30)]``, identical in
    #: Approach_env, Place_env, Insert_env, Regrasp_env and Pullout_env).
    #: ``Model_evaluation.py`` takes the threshold from the command line and the
    #: results files do not record what was passed, so the class default is the
    #: only tolerance actually written down anywhere in the task code.
    #:
    #: Replaying 25 demonstration episodes through this loop at that tolerance
    #: gives Approach 25/25 and Place 22/25, against published 96 +- 6 and
    #: 97 +- 9.  At Env_info's tighter 10 degrees the same runs give 23/25 and
    #: 12/25 -- which is the honest way to read what these policies are: Place
    #: is certified to *thirty degrees* of orientation error and is not a
    #: precision orientation controller, whatever "angle the needle correctly"
    #: needs it to be.
    success_trans_cm: float
    #: ||rpy_achieved - rpy_desired||, radians.  NOTE: SurgicAI's criteria() uses
    #: the Euclidean norm of the RPY *difference vector*, not the geodesic angle
    #: between the two rotations.  They are different numbers.  Reproducing a
    #: training success rate means using this one; judging whether the tool is
    #: physically pointing the right way means using the geodesic.  Both are
    #: reported; see LoopConfig.rot_metric.
    success_rot_rad: float
    #: the tighter pair shipped in ``RL/Env_info/<task>_noise_env_info``, which
    #: the demonstrations were collected against.  Reaching *this* is what a
    #: real grasp or placement needs; reaching ``success_*`` above is only what
    #: the policy was ever shown to do.
    env_info_trans_cm: float
    env_info_rot_rad: float
    max_steps: int
    #: mean of the frozen goals in the embedded demonstrations, raw units
    trained_goal_vec7: np.ndarray
    #: per-axis support of those goals, centimetres
    goal_min_cm: np.ndarray
    goal_max_cm: np.ndarray
    #: support of the start pose relative to the goal, in the gripper frame at
    #: the episode start: R_start^T @ (p_goal - p_start), centimetres
    start_offset_tool_min: np.ndarray
    start_offset_tool_max: np.ndarray
    start_offset_tool_mean: np.ndarray
    #: geodesic start -> goal rotation over the demonstrations, degrees
    start_rot_deg_min: float
    start_rot_deg_max: float
    start_rot_deg_median: float
    #: mean start -> goal rotation as a rotation vector, training frame
    start_to_goal_rotvec_mean: np.ndarray
    #: normalised jaw the demonstrations began and ended at
    demo_start_jaw: float
    demo_goal_jaw: float
    #: median demonstration episode length, control cycles
    demo_episode_steps: int
    #: median straight-line travel, centimetres
    demo_travel_cm: float
    #: True once the step size has been reproduced from the checkpoint's own
    #: demonstrations to numerical zero.  False means it was read off a config
    #: file and nothing has checked it -- run tools/recover_step_size.py.
    step_size_verified: bool = False
    note: str = ""

    def in_support(self, offset_tool_cm, rot_deg) -> list:
        """Return the reasons this start geometry sits outside the support."""
        offset = np.asarray(offset_tool_cm, dtype=np.float64).reshape(3)
        axes = "xyz"
        reasons = [
            f"tool-{axes[i]} {offset[i]:+.2f} cm outside "
            f"[{self.start_offset_tool_min[i]:+.2f}, {self.start_offset_tool_max[i]:+.2f}]"
            for i in range(3)
            if offset[i] < self.start_offset_tool_min[i] - SUPPORT_EPS_CM
            or offset[i] > self.start_offset_tool_max[i] + SUPPORT_EPS_CM
        ]
        if not (self.start_rot_deg_min <= float(rot_deg) <= self.start_rot_deg_max):
            reasons.append(
                f"start->goal rotation {float(rot_deg):.1f} deg outside "
                f"[{self.start_rot_deg_min:.1f}, {self.start_rot_deg_max:.1f}]"
            )
        return reasons

    def describe(self) -> str:
        return (
            f"{self.name}: step {self.step_size[0]*1000:.2f} mm / "
            f"{np.degrees(self.step_size[3]):.1f} deg"
            f"{'' if self.step_size_verified else ' [UNVERIFIED]'}, "
            f"success {self.success_trans_cm:.2f} cm / "
            f"{np.degrees(self.success_rot_rad):.1f} deg, "
            f"<= {self.max_steps} steps"
        )


def _step(trans_mm, deg, jaw):
    return np.array(
        [trans_mm * 1e-3] * 3 + [np.deg2rad(deg)] * 3 + [jaw], dtype=np.float64
    )


#: Upstream ``RL/Evaluation_model/Approach/TD3_HER_BC/final_model.zip``.
#: Reported 96% +- 6% in simulation.  Genuinely multi-goal: the 50 embedded
#: demonstration goals span 0.8 x 3.5 x 2.6 cm, where R6 collapsed to one goal.
APPROACH_UPSTREAM = SubtaskContract(
    name="Approach (upstream TD3_HER_BC)",
    step_size=_step(1.0, 3.0, 0.05),
    demo_step_size=_step(0.5, 2.0, 0.05),
    success_trans_cm=0.5,
    success_rot_rad=float(np.deg2rad(30.0)),
    env_info_trans_cm=0.3,
    env_info_rot_rad=float(np.deg2rad(10.0)),
    max_steps=300,
    trained_goal_vec7=np.array(
        [-0.032556, 0.011749, -0.118730, -3.775371, 0.493273, 1.311746, 0.0]
    ),
    goal_min_cm=np.array([-3.685, -0.518, -12.991]),
    goal_max_cm=np.array([-2.904, 3.006, -10.414]),
    start_offset_tool_min=np.array([-1.541, 0.602, 2.442]),
    start_offset_tool_max=np.array([1.618, 3.433, 4.085]),
    start_offset_tool_mean=np.array([0.053, 1.990, 3.312]),
    start_rot_deg_min=53.2,
    start_rot_deg_max=97.4,
    start_rot_deg_median=67.4,
    start_to_goal_rotvec_mean=np.array([-0.8233, -0.5191, 0.7030]),
    demo_start_jaw=0.80,
    demo_goal_jaw=0.00,
    demo_episode_steps=121,
    demo_travel_cm=3.87,
    step_size_verified=True,
    note=(
        "1.0 mm / 3 deg from RL_training_online.py; reproduces 19/20 of its own "
        "demonstration episodes through this loop (published 96% +- 6%)"
    ),
)

#: Upstream ``RL/Evaluation_model/Place/TD3_HER_BC/final_model.zip``.
#: Reported 97% +- 9%.  Carries the needle from the lift pose to the suturing
#: entry point.  Its goal support is tight (4.5 x 2.1 x 2.4 mm) because the
#: entry point barely moves; what varies is the wrist, by 94-136 degrees.
PLACE_UPSTREAM = SubtaskContract(
    name="Place (upstream TD3_HER_BC)",
    step_size=_step(1.0, 3.0, 0.05),
    demo_step_size=_step(0.5, 2.0, 0.05),
    success_trans_cm=0.5,
    success_rot_rad=float(np.deg2rad(30.0)),
    env_info_trans_cm=0.5,
    env_info_rot_rad=float(np.deg2rad(10.0)),
    max_steps=300,
    trained_goal_vec7=np.array(
        [-0.061201, 0.007671, -0.118702, -3.990033, -0.088337, 2.825493, 0.0]
    ),
    goal_min_cm=np.array([-6.383, 0.699, -12.006]),
    goal_max_cm=np.array([-5.936, 0.909, -11.770]),
    start_offset_tool_min=np.array([-2.962, -3.133, -1.402]),
    start_offset_tool_max=np.array([1.791, -0.037, -1.167]),
    start_offset_tool_mean=np.array([-1.032, -2.131, -1.283]),
    start_rot_deg_min=94.0,
    start_rot_deg_max=136.2,
    start_rot_deg_median=107.4,
    start_to_goal_rotvec_mean=np.array([-0.4680, 1.5642, -0.9695]),
    demo_start_jaw=0.00,
    demo_goal_jaw=0.00,
    demo_episode_steps=132,
    demo_travel_cm=3.00,
    step_size_verified=True,
    note=(
        "the jaw never moves during Place -- the needle is held throughout, so "
        "the jaw action channel is identically zero and its scale is "
        "unrecoverable from the demonstrations (0.05 assumed, from Env_info)"
    ),
)

#: The locally revised checkpoint this project has been running.  Its step size
#: has NEVER been reproduced from its demonstrations; 1.5 mm / 3 deg was read
#: out of the training sources.  The upstream checkpoints both use 0.5 mm / 2
#: deg, so treat this as suspect until tools/recover_step_size.py says otherwise.
APPROACH_R6 = SubtaskContract(
    name="Approach (R6 single-goal revision)",
    step_size=_step(1.0, 3.0, 0.05),
    demo_step_size=_step(1.5, 3.0, 0.05),
    success_trans_cm=1.0,
    env_info_trans_cm=0.5,
    env_info_rot_rad=float(np.deg2rad(10.0)),
    success_rot_rad=float(np.deg2rad(10.0)),
    max_steps=200,
    trained_goal_vec7=np.asarray(R6_TRAINED_GOAL_VEC7, dtype=np.float64),
    goal_min_cm=R6_GOAL_MIN[:3].copy(),
    goal_max_cm=R6_GOAL_MAX[:3].copy(),
    start_offset_tool_min=R6_START_OFFSET_TOOL_MIN.copy(),
    start_offset_tool_max=R6_START_OFFSET_TOOL_MAX.copy(),
    start_offset_tool_mean=R6_START_OFFSET_TOOL_MEAN.copy(),
    start_rot_deg_min=R6_START_ROT_DEG_MIN,
    start_rot_deg_max=R6_START_ROT_DEG_MAX,
    start_rot_deg_median=R6_START_ROT_DEG_MEDIAN,
    start_to_goal_rotvec_mean=R6_START_TO_GOAL_ROTVEC_MEAN.copy(),
    demo_start_jaw=0.76,
    demo_goal_jaw=0.00,
    demo_episode_steps=120,
    demo_travel_cm=4.12,
    step_size_verified=True,
    note=(
        "single-goal local revision. It inherited upstream's 1.0 mm / 3 deg "
        "after all: replayed from its own 50 demonstration starts against a "
        "perfect arm, 1.0 mm/3 deg gives 46/50 (92%) where the 1.5 mm/3 deg "
        "this package applied gives 36/50 (72%), 0.5 mm/2 deg gives 24/50 and "
        "2.0 mm/2 deg gives 10/50. Every R6 result recorded in this project "
        "before 2026-09-16 was taken 50% over scale. Its own tolerance "
        "(1.0 cm / 10 deg, 200 steps) beats the upstream pair on this "
        "checkpoint: 46/50 against 34/50."
    ),
)

CONTRACTS = {
    "approach_upstream": APPROACH_UPSTREAM,
    "place_upstream": PLACE_UPSTREAM,
    "approach_r6": APPROACH_R6,
}

#: SHA256 -> contract key.  A checkpoint whose digest is not here has no known
#: contract, and running it means supplying the step size and tolerances by
#: hand.  Digests of the upstream files as committed in surgical-robotics-ai/
#: SurgicAI under RL/Evaluation_model/<task>/TD3_HER_BC/final_model.zip.
CHECKPOINT_CONTRACTS = {
    "06fc7813f93feef5efa16d06e43c33f9750b4ef78d7ff921563205aaf4d2e71c": "approach_upstream",
    "4fc615f074158780744de28d189eddd7adc2bbdbbb8639f77582cbc2007eaff0": "place_upstream",
    "6286a88c21f04abfbc4b0747a87a67bc2c5dcba17f710692c6b5138f7776e525": "approach_r6",
}

#: The other released upstream checkpoints, by digest.  Not deployed by this
#: package -- Insert drives the needle into tissue at a 0.2 mm step size and
#: 0.1 cm tolerance, which is a separate safety conversation -- but recorded so
#: an unknown file can at least be named.
OTHER_UPSTREAM_CHECKPOINTS = {
    "3167dc35c50b75cf1b06f56f43556b27680f559ebf2e1c56dd5b755284ce4b31": "Insert TD3_HER_BC",
    "5a2eaa04b832d6e1423f07d535d766818c90dc3b946cb1df3d18b6140db5fecf": "Pullout TD3_HER_BC",
    "4eac0c1ac2510791f0147b6e32f3ef9b12712044d27a4c5fb33aa271c4a91a64": "Regrasp TD3_HER_BC",
}


def contract_for_digest(digest: str) -> Optional[SubtaskContract]:
    key = CHECKPOINT_CONTRACTS.get(str(digest))
    return CONTRACTS[key] if key else None
