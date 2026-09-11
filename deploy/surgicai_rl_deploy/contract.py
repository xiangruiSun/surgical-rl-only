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
# RL/utils/utils.py :: default_step_size(trans_step=1.5e-3, angle_step_deg=3.0,
# jaw_step=0.05); identical to collect_measured_approach_demos.STEP_SIZE_RAW.
STEP_SIZE_RAW = np.array(
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
