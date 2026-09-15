#!/usr/bin/env python3
"""Entry point for the full run: stage, approach, grasp, lift, place.

Three poses describe the task, all in the frame ``measured_cp`` reports
(``ECM`` on lcsr-dvrk-15), all **tool** poses -- where the gripper goes, not
where the needle goes:

    --grasp-pos / --grasp-quat     where the gripper must be to take the needle
    --suture-pos / --suture-quat   where the gripper must be for the needle to
                                   sit correctly angled at the entry point

The start pose is read from the arm; you do not pass it.

    source /opt/ros/humble/setup.bash

    # 1. rehearse it with no robot at all
    python3 tools/offline_grasp_lift.py \
        --start-pos  <x y z> --start-quat <qx qy qz qw> \
        --grasp-pos  <x y z> --grasp-quat <qx qy qz qw> \
        --suture-pos <x y z> --suture-quat <qx qy qz qw> --suture-confirmed \
        --lift-sign -1 --grasp-gate always --jaw-stops-at-deg -5

    # 2. dry run on the real arm - publishes nothing, walks the whole sequence
    python3 run_pipeline.py \
        --grasp-pos  <x y z> --grasp-quat <qx qy qz qw> \
        --suture-pos <x y z> --suture-quat <qx qy qz qw>

    # 3. live, manual gate: the arm closes the jaw, stops, and waits for you
    python3 run_pipeline.py \
        --grasp-pos  <x y z> --grasp-quat <qx qy qz qw> \
        --suture-pos <x y z> --suture-quat <qx qy qz qw> --suture-confirmed \
        --controller d2 --interface move_cp --rate 2 \
        --lift-sign -1 --jaw-baseline jaw_baseline.json \
        --trace live.jsonl --execute

To put the RL policy on the approach leg, add ``--controller rl --model <zip>
--stage``.  ``--stage`` servos the arm into the checkpoint's own demonstrated
support first; without it the policy starts wherever the arm happens to be,
which for this task has always been outside it.

To measure the SurgicAI Place policy on the transport leg without letting it
drive, add ``--shadow-model <Place checkpoint>``.  It is stepped on the poses
the arm actually visits and its commands are logged, never published.

Dry run by default.  ``--lift-sign`` and ``--suture-confirmed`` are both
required for a live run.  Keep a hand on the e-stop: the guards in this package
are software.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from surgicai_rl_deploy.grasp_lift_node import main

if __name__ == "__main__":
    sys.exit(main())
