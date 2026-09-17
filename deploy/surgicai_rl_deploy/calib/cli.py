"""Command-line wiring: turn a needle observation into ``--grasp-pos``.

The deployment's entry points have always taken the grasp pose as three numbers
on the command line -- a human read them off the scene and typed them in.  This
module adds the other way in::

    --needle-pose  <file or nine numbers>    where FoundationPose says the needle is
    --grasp-calibration <model.json>         the fitted correction

and resolves the pair into exactly the same three numbers, through section 14's
pipeline, with the section 15 refusals attached.

Why it is a separate module
---------------------------
Two reasons, both about keeping the failure modes visible.  The resolution
prints what it did -- camera point, nominal point, correction, target -- so a
dry run shows the four numbers a miss would have to be explained by, rather than
one.  And the same function backs the offline rehearsal, so the numbers a live
run will use can be produced on a laptop first, which is the pattern the rest of
this package already follows.

``--grasp-pos`` still works, and still wins if both are given, because a human
who has just typed in a measured pose means it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .models import DEFAULT_MAX_CORRECTION_MM
from .resolve import GraspResolver, GraspTarget


def add_arguments(ap) -> None:
    """Attach the needle-observation options to an ``ArgumentParser``."""
    group = ap.add_argument_group(
        "needle observation (an alternative to typing --grasp-pos by hand)"
    )
    group.add_argument(
        "--needle-pose", default=None,
        help="FoundationPose's estimate of the needle in the CAMERA frame. "
             "Either a path to a JSON file -- a 4x4 'T', a {position, "
             "quaternion} object, or a list of either, which are averaged as "
             "repeated frames of one stationary needle -- or seven numbers "
             "'x y z qx qy qz qw' separated by commas.",
    )
    group.add_argument(
        "--grasp-calibration", default=None,
        help="a model from tools/fit_grasp_calibration.py. Without it the "
             "needle pose is carried through the hand-eye transform with NO "
             "empirical correction, which is section 11's first baseline and is "
             "worth several millimetres of miss.",
    )
    group.add_argument(
        "--needle-point", choices=["arc", "mesh_origin"], default=None,
        help="which point on the needle to aim at. Defaults to whatever the "
             "calibration was fitted with; 'mesh_origin' is the raw "
             "FoundationPose translation, which is the CENTRE OF THE ARC and "
             "10.18 mm from the wire.",
    )
    group.add_argument("--needle-grasp-angle", type=float, default=None,
                       help="arc angle of the grasp point, degrees; SurgicAI's "
                            "own target is 30")
    group.add_argument("--hand-eye", default=None,
                       help="JSON file holding the 4x4 ^E T_C; defaults to the "
                            "transform this package ships")
    group.add_argument("--max-correction-mm", type=float,
                       default=DEFAULT_MAX_CORRECTION_MM,
                       help="refuse a calibration that can command more than "
                            "this much correction inside its own box")
    group.add_argument("--allow-uncalibrated-needle", action="store_true",
                       help="accept --needle-pose with no calibration, or a "
                            "calibration that was never cross-validated")


# ---------------------------------------------------------------------------
def load_needle_pose(spec: str) -> list:
    """Parse ``--needle-pose`` into a list of 4x4 camera-frame poses."""
    text = str(spec).strip()
    if "," in text and not Path(text).exists():
        nums = [float(v) for v in text.replace(" ", ",").split(",") if v]
        if len(nums) != 7:
            raise ValueError(
                "inline --needle-pose takes seven numbers, 'x y z qx qy qz qw'; "
                f"got {len(nums)}"
            )
        return [_from_pos_quat(nums[:3], nums[3:])]

    raw = json.loads(Path(text).read_text())
    records = raw if isinstance(raw, list) else [raw]
    # A bare 4x4 nested list is itself a list, so disambiguate on shape.
    if (
        isinstance(raw, list)
        and len(raw) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in raw)
    ):
        records = [raw]
    out = []
    for rec in records:
        if isinstance(rec, list):
            out.append(np.asarray(rec, dtype=np.float64).reshape(4, 4))
        elif "T" in rec:
            out.append(np.asarray(rec["T"], dtype=np.float64).reshape(4, 4))
        else:
            out.append(_from_pos_quat(rec["position"], rec["quaternion"]))
    if not out:
        raise ValueError(f"{text} holds no needle poses")
    return out


def _from_pos_quat(pos, quat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def build_resolver(args) -> GraspResolver:
    from .models import ResidualModel
    from .perception import GraspPointSpec, HandEye
    from .models import Workspace, zero_model

    hand_eye = HandEye(
        np.asarray(json.loads(Path(args.hand_eye).read_text()), dtype=np.float64)
        if getattr(args, "hand_eye", None) else None,
        source=getattr(args, "hand_eye", None) or "package default",
    )

    if getattr(args, "grasp_calibration", None):
        resolver = GraspResolver.from_files(
            args.grasp_calibration,
            hand_eye=hand_eye if getattr(args, "hand_eye", None) else None,
            max_correction_mm=args.max_correction_mm,
            require_validated=not args.allow_uncalibrated_needle,
        )
    else:
        if not args.allow_uncalibrated_needle:
            raise ValueError(
                "--needle-pose without --grasp-calibration carries the "
                "FoundationPose estimate through the hand-eye transform with no "
                "empirical correction at all. On this project's own numbers "
                "that is a several-millimetre miss, which is more than a needle "
                "grasp has to spare. Fit one with "
                "tools/fit_grasp_calibration.py, or pass "
                "--allow-uncalibrated-needle to do it deliberately."
            )
        # A zero model over an unbounded box: the uncorrected baseline, run
        # through exactly the same code path as a real one.
        resolver = GraspResolver(
            model=zero_model(Workspace([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])),
            hand_eye=hand_eye,
            require_validated=False,
        )

    if getattr(args, "needle_point", None) or getattr(args, "needle_grasp_angle", None):
        from .perception import DEFAULT_GRASP_ANGLE_DEG

        resolver.grasp_point = GraspPointSpec(
            mode=args.needle_point or resolver.grasp_point.mode,
            angle_deg=(
                args.needle_grasp_angle
                if args.needle_grasp_angle is not None
                else resolver.grasp_point.angle_deg
            ),
        )
    return resolver


def resolve_grasp_position(args, strict: bool = False) -> Tuple[Optional[GraspTarget], list]:
    """Turn ``--needle-pose`` into a grasp position, or explain why not.

    Returns ``(target, messages)``.  ``target`` is ``None`` when no needle pose
    was given at all, which is the ordinary "the operator typed --grasp-pos"
    case and not an error.
    """
    messages = []
    if not getattr(args, "needle_pose", None):
        return None, messages

    if getattr(args, "grasp_pos", None) is not None:
        messages.append(
            "both --grasp-pos and --needle-pose were given; the typed pose wins "
            "and the needle observation is only reported"
        )

    resolver = build_resolver(args)
    frames = load_needle_pose(args.needle_pose)

    orientation = None
    if getattr(args, "grasp_quat", None) is not None:
        orientation = Rotation.from_quat(np.asarray(args.grasp_quat, dtype=np.float64))

    target = resolver.resolve(frames, orientation=orientation, strict=strict)
    messages.append(target.describe())
    return target, messages


def apply_to_args(args, strict: bool = False) -> Tuple[Optional[GraspTarget], list]:
    """Resolve, and write the result into ``args.grasp_pos`` if it was empty.

    Called by the entry points just after parsing, so that everything
    downstream -- the plan, the feasibility precheck, the trace -- sees an
    ordinary grasp position and needs to know nothing about perception.
    """
    target, messages = resolve_grasp_position(args, strict=strict)
    if target is None:
        return None, messages
    if getattr(args, "grasp_pos", None) is None:
        if not target.report.ok:
            return target, messages
        args.grasp_pos = target.p_target_m.tolist()
        if getattr(args, "grasp_quat", None) is None and \
                getattr(args, "goal_orientation", None) is None:
            messages.append(
                "no --grasp-quat was given, so the gripper holds its current "
                "orientation. The calibration corrects POSITION only and was "
                "taught at one wrist angle (section 8); if that angle is not "
                "the one the arm is in, pass it with --grasp-quat."
            )
    return target, messages
