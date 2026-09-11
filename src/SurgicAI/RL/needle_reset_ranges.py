"""Single source of truth for Approach needle-reset distributions.

There is no measured real-robot needle-placement distribution yet.  The
deployment/demo target below is therefore a conservative engineering
assumption, not a measurement.  Its yaw bound stays at 30 degrees because the
Stage-E perception audit was reliable at 20 degrees but developed near-180
degree failures at 40 degrees.
"""

from __future__ import annotations

import numpy as np


ASSUMED_REAL_X_MM = 3.0
ASSUMED_REAL_Y_MM = 3.0
ASSUMED_REAL_RZ_DEG = 30.0

# Demonstration collection and curriculum level 0 deliberately share support.
ASSUMED_REAL_NEEDLE_RANGE = np.array(
    [
        ASSUMED_REAL_X_MM * 1.0e-3,
        ASSUMED_REAL_Y_MM * 1.0e-3,
        np.deg2rad(ASSUMED_REAL_RZ_DEG),
    ],
    dtype=np.float32,
)
CURRICULUM_START_NEEDLE_RANGE = ASSUMED_REAL_NEEDLE_RANGE.copy()

# Simulation-only orientation expansion.  XY stays inside the assumed real
# support because wider contact-constrained commands can hit the production
# 5 mm resolved-pose transport guard.  The 45-degree yaw edge is outside the
# validated uncontrolled real-camera envelope and requires a controlled safe
# initialization view before any real deployment.
CURRICULUM_END_X_MM = 3.0
CURRICULUM_END_Y_MM = 3.0
CURRICULUM_END_RZ_DEG = 45.0
CURRICULUM_END_NEEDLE_RANGE = np.array(
    [
        CURRICULUM_END_X_MM * 1.0e-3,
        CURRICULUM_END_Y_MM * 1.0e-3,
        np.deg2rad(CURRICULUM_END_RZ_DEG),
    ],
    dtype=np.float32,
)
