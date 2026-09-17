"""Task-specific empirical calibration for vision-guided needle grasping.

The two-stage framework of the protocol document::

    FoundationPose + hand-eye transform        coarse geometric localisation
    + Bernstein residual calibration           task-specific fine correction
    -----------------------------------------------------------------------
    = the measured_cp target expected to grasp the needle

Read :mod:`.bernstein` first: it contains the one algebraic fact -- that the
constant-offset and affine baselines are the degree-0 and degree-1 members of
the Bernstein family -- which turns the protocol's separate model comparisons
into a single choice of degree.

Then run ``tools/residual_structure.py`` before collecting anything.  It derives
what each error term in section 4 contributes as a function of position, from
the dVRK's own DH parameters, and the answer changes what is worth collecting.

Nothing in this package imports ROS, torch or a checkpoint.  It is numpy and
scipy, and the whole of it runs on a laptop.
"""

from .bernstein import BernsteinBasis, ladder, n_coefficients
from .dataset import CalibrationDataset, Placement
from .fit import fit, fit_dataset, smoothing_grid
from .models import (
    OutsideCalibratedRegion,
    ResidualModel,
    Workspace,
    zero_model,
)
from .perception import (
    DEFAULT_GRASP_ANGLE_DEG,
    DEFAULT_T_EC,
    NEEDLE_RADIUS_M,
    GraspPointSpec,
    HandEye,
    average_poses,
    convention_digest,
    flip_reasons,
    needle_frame_N,
    needle_point_N,
    nominal_position,
)
from .resolve import GraspResolver, GraspTarget
from .validate import (
    CVResult,
    ErrorSummary,
    cross_validate,
    paired_improvement,
    placements_needed,
    report,
    select_model,
    spatial_holdout,
    summarise,
    uncorrected,
    variance_budget,
)

__all__ = [
    "BernsteinBasis",
    "CVResult",
    "CalibrationDataset",
    "DEFAULT_GRASP_ANGLE_DEG",
    "DEFAULT_T_EC",
    "ErrorSummary",
    "GraspPointSpec",
    "GraspResolver",
    "GraspTarget",
    "HandEye",
    "NEEDLE_RADIUS_M",
    "OutsideCalibratedRegion",
    "Placement",
    "ResidualModel",
    "Workspace",
    "average_poses",
    "convention_digest",
    "cross_validate",
    "fit",
    "fit_dataset",
    "flip_reasons",
    "ladder",
    "n_coefficients",
    "needle_frame_N",
    "needle_point_N",
    "nominal_position",
    "paired_improvement",
    "placements_needed",
    "report",
    "select_model",
    "smoothing_grid",
    "spatial_holdout",
    "summarise",
    "uncorrected",
    "variance_budget",
    "zero_model",
]
