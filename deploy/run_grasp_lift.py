#!/usr/bin/env python3
"""Entry point for the real-robot approach + grasp + lift run.

    source /opt/ros/humble/setup.bash

    # 1. dry run - publishes nothing, prints the precheck and the whole sequence
    python3 run_grasp_lift.py --grasp-pos -0.050726357 0.015332369 0.049514053

    # 2. live, manual gate: the arm closes the jaw, stops, and waits for you
    python3 run_grasp_lift.py --grasp-pos -0.050726357 0.015332369 0.049514053 \
        --controller d2 --interface move_cp --rate 2 \
        --lift-sign -1 --jaw-baseline jaw_baseline.json \
        --trace live.jsonl --execute

Dry run by default.  ``--lift-sign`` is required for a live run.  Keep a hand
on the e-stop: the guards in this package are software.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from surgicai_rl_deploy.grasp_lift_node import main

if __name__ == "__main__":
    sys.exit(main())
