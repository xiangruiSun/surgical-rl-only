import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from PyKDL import Frame, Rotation, Vector, dot
import numpy as np
import json
import math
import importlib
from typing import Type


def resolve_src_env(task_name: str):
    """
    Resolve the SRC Gymnasium environment class for a given task name.

    This avoids CWD-dependent imports by loading environments as `RL.<Task>_env`.
    """
    # Keep this mapping logic centralized so scripts don't depend on CWD/PYTHONPATH quirks.
    # Environments live as modules like `RL/Approach_env.py` with classes `SRC_approach`.
    task = str(task_name)
    module_name = f"RL.{task.capitalize()}_env"
    class_name = f"SRC_{task.lower()}"
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def default_step_size(
    *,
    trans_step: float = 1.5e-3,
    angle_step_rad: float | None = None,
    angle_step_deg: float = 3.0,
    jaw_step: float = 0.05,
) -> np.ndarray:
    """
    Standard 7D action scaling used across training/eval scripts.
    Returns: np.array([dx, dy, dz, droll, dpitch, dyaw, djaw], dtype=float32)
    """
    if angle_step_rad is None:
        angle_step_rad = np.deg2rad(float(angle_step_deg))
    return np.array(
        [trans_step, trans_step, trans_step, angle_step_rad, angle_step_rad, angle_step_rad, jaw_step],
        dtype=np.float32,
    )


def threshold_from_args(trans_error: float, angle_error_deg: float) -> np.ndarray:
    """Helper to build (trans, angle) thresholds in consistent units."""
    return np.array([float(trans_error), np.deg2rad(float(angle_error_deg))], dtype=np.float32)


def experiment_variant(
    *,
    variant: str | None = None,
    stepDR: bool | None = None,
    randomized: bool | None = None,
    randomization_params: str | None = None,
) -> str:
    """
    Canonical experiment naming used for directory layout.

    Priority:
    1) explicit `variant` if provided
    2) `randomized=True` OR `randomization_params` indicates all-on → "randomization"
    3) `stepDR=True` → "stepDR"
    4) otherwise → "base_env"
    """
    # `variant` is intended to be the only knob users need for directory naming.
    # Keep it stable because it becomes part of experiment paths and filenames.
    if variant:
        return str(variant)

    if randomized:
        return "randomization"

    if randomization_params is not None:
        # historical convention: "1,1,1,1,1" means full world randomization
        if str(randomization_params).strip() == "1,1,1,1,1":
            return "randomization"

    if stepDR:
        return "stepDR"

    return "base_env"

def frame_to_vector(frame):
    """
    Convert a PyKDL.Frame to a 6D vector [x, y, z, roll, pitch, yaw]
    """
    if frame is None:
        return np.zeros(6, dtype=np.float32)
    x = frame.p[0]
    y = frame.p[1]
    z = frame.p[2]
    
    # GetRPY returns (roll, pitch, yaw) - be explicit
    rpy = frame.M.GetRPY()
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]
    
    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)

def wrap_rpy(rpy):
    """Wrap roll/pitch/yaw to (-pi, pi]."""
    angles = np.asarray(rpy, dtype=np.float64).reshape(3)
    return ((angles + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def rotation_geodesic_rad(vec_a, vec_b):
    """SO(3) geodesic angle between two 6D/7D pose vectors (uses RPY only)."""
    Ta = vector_to_frame(vec_a)
    Tb = vector_to_frame(vec_b)
    qa = np.array(Ta.M.GetQuaternion(), dtype=np.float64)
    qb = np.array(Tb.M.GetQuaternion(), dtype=np.float64)
    qa /= np.linalg.norm(qa) + 1e-12
    qb /= np.linalg.norm(qb) + 1e-12
    dot = abs(float(np.dot(qa, qb)))
    dot = np.clip(dot, -1.0, 1.0)
    return float(2.0 * np.arccos(dot))


def axis_angle_between_vec7(vec_a, vec_b, axis="z", sign_invariant=True):
    """Angle between a body axis after applying each pose's rotation."""
    Ta = vector_to_frame(vec_a)
    Tb = vector_to_frame(vec_b)
    local_axis = {
        "x": Vector(1, 0, 0),
        "y": Vector(0, 1, 0),
        "z": Vector(0, 0, 1),
    }[axis]
    aa = np.array((Ta.M * local_axis), dtype=np.float64)
    bb = np.array((Tb.M * local_axis), dtype=np.float64)
    aa /= np.linalg.norm(aa) + 1e-12
    bb /= np.linalg.norm(bb) + 1e-12
    dot = float(np.dot(aa, bb))
    if sign_invariant:
        dot = abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.arccos(dot))


def vector_to_frame(vec):
    """
    Convert a 6D vector [x, y, z, roll, pitch, yaw] to a PyKDL.Frame.
    Extra entries (e.g. jaw) are ignored.
    """
    if vec is None:
        return Frame()
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size < 6:
        raise ValueError(f"vector_to_frame expects at least 6 values, got {v.size}")
    return Frame(
        Rotation.RPY(float(v[3]), float(v[4]), float(v[5])),
        Vector(float(v[0]), float(v[1]), float(v[2]))
    )

def convert_mat_to_frame(mat):
    frame = Frame(Rotation.RPY(0, 0, 0), Vector(0, 0, 0))
    for i in range(3):
        for j in range(3):
            frame[(i, j)] = mat[i, j]

    for i in range(3):
        frame.p[i] = mat[i, 3]

    return frame

def convert_mat_to_vector(mat):
    frame = convert_mat_to_frame(mat)
    return frame_to_vector(frame)


def gripper_T_c(scene_manager, psm_idx):
    """Measured gripper pose expressed in camera frame."""
    psm = scene_manager.psm_list[psm_idx - 1]
    T_g_b = convert_mat_to_frame(psm.measured_cp())
    T_g_w = psm.get_T_b_w() * T_g_b
    T_w_c = scene_manager.ecm.get_T_w_c()
    return T_w_c * T_g_w


def goal_vec7_c_from_goal_vec7_b(goal_vec7_b, psm_idx, scene_manager):
    """Convert desired goal from PSM base frame to camera frame."""
    psm = scene_manager.psm_list[psm_idx - 1]
    T_goal_b = vector_to_frame(goal_vec7_b)
    T_goal_w = psm.get_T_b_w() * T_goal_b
    T_w_c = scene_manager.ecm.get_T_w_c()
    goal_vec6_c = frame_to_vector(T_w_c * T_goal_w)
    return np.append(goal_vec6_c, float(np.asarray(goal_vec7_b)[6]))


def goal_vec7_b_from_goal_vec7_c(goal_vec7_c, psm_idx, scene_manager):
    """Convert desired goal from camera frame to PSM base frame (for commanding)."""
    psm = scene_manager.psm_list[psm_idx - 1]
    T_goal_c = vector_to_frame(goal_vec7_c)
    T_c_w = scene_manager.ecm.get_T_c_w()
    T_goal_w = T_c_w * T_goal_c
    T_goal_b = psm.get_T_w_b() * T_goal_w
    goal_vec6_b = frame_to_vector(T_goal_b)
    return np.append(goal_vec6_b, float(np.asarray(goal_vec7_c)[6]))

def get_angle(vec_a, vec_b, up_vector=None):
    vec_a.Normalize()
    vec_b.Normalize()
    cross_ab = vec_a * vec_b
    vdot = dot(vec_a, vec_b)
    # print('VDOT', vdot, vec_a, vec_b)
    # Check if the vectors are in the same direction
    if 1.0 - vdot < 0.000001:
        angle = 0.0
        # Or in the opposite direction
    elif 1.0 + vdot < 0.000001:
        angle = np.pi
    else:
        angle = math.acos(vdot)

    if up_vector is not None:
        same_dir = np.sign(dot(cross_ab, up_vector))
        if same_dir < 0.0:
            angle = -angle

    return angle

def load_json_dvrk(file_path:str)->dict:
    '''
    Load json files from dVRK repository
    :param file_path: json file path
    :return: a dictionary with loaded json file content
    '''
    with open(file_path) as f:
        data = f.read()
        data = re.sub("//.*?\n", "", data)
        data = re.sub("/\\*.*?\\*/", "", data)
        obj = data[data.find('{'): data.rfind('}') + 1]
        jsonObj = json.loads(obj)
    return jsonObj