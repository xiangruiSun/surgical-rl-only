#!/usr/bin/env python3
"""Entry point for the real-robot approach run.

    source /opt/ros/humble/setup.bash
    python3 run_approach.py --goal-pos -0.050726357 0.015332369 0.049514053 \
        --model models/rl/r6_unified_single_goal_yaw15_seed1_final.zip

Dry run by default. Add --execute to publish commands.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from surgicai_rl_deploy.ros_node import main

if __name__ == "__main__":
    sys.exit(main())
