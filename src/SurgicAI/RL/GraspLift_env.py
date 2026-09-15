"""Simulation counterpart of the real grasp-and-lift deployment.

The point of this file is **one state machine, two worlds**.  The phase logic,
the jaw ramp, the settle test, the slip monitor and the safety clamps all live
in ``deploy/surgicai_rl_deploy/sequence.py`` and are imported here unchanged.
AMBF supplies the arm; the deploy package supplies the behaviour.  If the
sequence works here and fails on hardware, the difference is the robot, not the
logic -- which is the only way a sim2real claim means anything.

What is genuinely different in simulation
-----------------------------------------
AMBF *can* confirm a grasp.  ``PSM.grasp_status()["needle_grasped"]`` is true
only when the jaw command crossed the actuation threshold **and** a finger
ghost sensor reported a Needle body, and the needle is then attached by a
magnetic constraint.  That is ground truth, and no such signal exists on a real
dVRK.

Rather than special-case it, the ground truth is fed in through the sequencer's
ordinary operator-confirmation hook: in simulation "the operator" is the ghost
sensor.  The same ``grasp_gate='manual'`` path that waits for a human on
hardware waits for the simulator here.

Units
-----
The simulation's jaw is the environment's normalised 0..1 command, not dVRK
radians.  :class:`SIM_JAW` maps it onto the same
:class:`~surgicai_rl_deploy.jaw.JawCalibration` interface: ``closed`` is the
0.05 actuation threshold from ``PSM.grasp_actuation_jaw_angle``, and ``grip``
is 0.0, i.e. commanded past that threshold.  Cartesian poses are metres and
radians in the PSM base frame, exactly as in the deploy package.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from RL.Approach_env import SRC_approach
from RL.utils.utils import convert_mat_to_frame, frame_to_vector


def _import_deploy():
    """Import the deployment package, which is the source of truth.

    Set ``SURGICAI_DEPLOY_ROOT`` if this tree is not laid out as
    ``<repo>/src/SurgicAI/RL`` next to ``<repo>/deploy``.
    """
    root = os.environ.get("SURGICAI_DEPLOY_ROOT")
    if root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "..", "..", "..", "deploy"))
    if not os.path.isdir(root):
        raise ImportError(
            f"deployment package not found at {root!r}. Set SURGICAI_DEPLOY_ROOT "
            "to the repository's deploy/ directory."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    import surgicai_rl_deploy  # noqa: F401

    return root


_import_deploy()

from surgicai_rl_deploy.controllers import D2Controller  # noqa: E402
from surgicai_rl_deploy.frames import Pose  # noqa: E402
from surgicai_rl_deploy.jaw import JawCalibration  # noqa: E402
from surgicai_rl_deploy.loop import SafetyLimits  # noqa: E402
from surgicai_rl_deploy.plan import LiftSpec, build_plan  # noqa: E402
from surgicai_rl_deploy.sequence import (  # noqa: E402
    ArmState,
    GraspLiftSequencer,
    SequenceConfig,
    PHASE_DONE,
)

#: The simulation's jaw command is normalised 0..1, and ``PSM.run_grasp_logic``
#: actuates below ``grasp_actuation_jaw_angle = 0.05``.  Commanding 0.0 is
#: therefore the simulation's equivalent of a squeeze.
SIM_JAW = JawCalibration(
    open_rad=1.0,
    closed_rad=0.05,
    grip_rad=0.0,
    approach_open_rad=0.8,
)


class SRC_grasp_lift(SRC_approach):
    """Approach, close on the needle, confirm with the ghost sensor, lift.

    The approach half is inherited from :class:`SRC_approach` unchanged, so the
    goal evaluation, the reset validity gate and the observation contract are
    exactly the ones the released checkpoints were trained against.
    """

    def __init__(
        self,
        *args,
        lift_distance_m: float = 0.015,
        lift_axis: str = "z",
        lift_sign: int = 1,
        lift_frame: str = "robot",
        lift_source: str = "frame_axis",
        grasp_confirm_timeout_steps: int = 200,
        **kwargs,
    ):
        # The sequence needs the finger-sensor confirmation path, not the
        # historical "attach on pose success" shortcut.
        kwargs.setdefault("require_grasp_confirmation", True)
        kwargs.setdefault("attach_on_pose_success", False)
        kwargs.setdefault("measured_success_reward", True)
        super().__init__(*args, **kwargs)

        if lift_source not in ("frame_axis", "needle_evaluator"):
            raise ValueError(
                "lift_source must be 'frame_axis' (identical to the real "
                "deployment) or 'needle_evaluator' (the simulator's own "
                "needle-frame offset)"
            )
        self.lift_source = lift_source
        self.lift_spec = LiftSpec(
            axis=lift_axis,
            sign=int(lift_sign),
            distance_m=float(lift_distance_m),
            frame=lift_frame,
            explicit=True,  # in simulation nothing is at risk
        )
        self.grasp_confirm_timeout_steps = int(grasp_confirm_timeout_steps)
        self.sequencer = None
        self.last_sequence_summary = None

    # ------------------------------------------------------------------
    # state plumbing
    # ------------------------------------------------------------------
    def _psm(self):
        return self.scene_manager.psm_list[self.psm_idx - 1]

    def measured_vec7(self):
        """Measured [x_m, y_m, z_m, r, p, y, jaw_norm] in the PSM base frame."""
        psm = self._psm()
        measured_mat = psm.measured_cp()
        if measured_mat is None:
            return None
        vec6 = frame_to_vector(convert_mat_to_frame(measured_mat))
        return np.append(vec6, float(psm.get_jaw_angle())).astype(np.float64)

    def arm_state(self):
        vec7 = self.measured_vec7()
        if vec7 is None:
            return None
        return ArmState(
            pose=Pose.from_vec7(
                np.concatenate([vec7[:6], [SIM_JAW.normalise(vec7[6])]])
            ),
            # The simulated jaw reaches its command: there is nothing physical
            # stopping it, so the evidence channel will correctly report "not
            # blocked" even while the needle is attached.  That mismatch is the
            # honest picture of what the real hardware channel can and cannot
            # see, and it is why simulation uses the ground-truth gate below.
            jaw_rad=float(vec7[6]),
            jaw_effort=None,
        )

    def needle_grasped(self) -> bool:
        status = self._psm().grasp_status()
        self.last_grasp_status = status
        return bool(status and status.get("needle_grasped", False))

    def _ground_truth_confirm(self):
        """The simulator standing in for the human at the manual gate."""
        return True if self.needle_grasped() else None

    # ------------------------------------------------------------------
    # the lift goal
    # ------------------------------------------------------------------
    def lift_goal_vec7(self, grasp_vec7):
        """Where the gripper ends up after the lift, in the PSM base frame."""
        grasp_pose = Pose.from_vec7(
            np.concatenate([np.asarray(grasp_vec7)[:6], [0.0]])
        )
        if self.lift_source == "needle_evaluator":
            # The simulator's own construction: the same grasp transform,
            # offset further along the needle frame's z.
            deeper = self.scene_manager.needle_goal_evaluator(
                lift_height=self.lift_height + float(self.lift_spec.distance_m),
                psm_idx=self.psm_idx,
                deg_angle=self.grasp_angle,
            )
            return self.apply_goal_offset(deeper)
        lifted = grasp_pose.p + self.lift_spec.displacement(grasp_pose)
        return np.concatenate([lifted, np.asarray(grasp_vec7)[3:6], [0.0]])

    # ------------------------------------------------------------------
    # the episode
    # ------------------------------------------------------------------
    def build_sequencer(self, approach_controller=None, config=None, limits=None):
        state = self.arm_state()
        if state is None:
            raise RuntimeError("measured_cp is not available yet; reset() first")

        grasp_vec7 = np.asarray(self.goal_obs, dtype=np.float64)
        plan = build_plan(
            state.pose,
            grasp_vec7[:3],
            goal_orientation="explicit",
            goal_quat_xyzw=tuple(
                Pose.from_vec7(np.concatenate([grasp_vec7[:6], [0.0]])).quat_xyzw()
            ),
            lift=self.lift_spec,
            jaw=SIM_JAW,
        )
        if self.lift_source == "needle_evaluator":
            lifted = self.lift_goal_vec7(grasp_vec7)
            plan.lifted = Pose.from_vec7(
                np.concatenate([lifted[:6], [SIM_JAW.normalise(SIM_JAW.grip_rad)]])
            )

        cfg = config or SequenceConfig(
            frame_mode="identity",
            grasp_gate="manual",  # the ghost sensor answers the gate
            # The env moves the jaw at most ``step_size[6]`` per action, so the
            # sequencer's ramp is matched to it rather than being clipped and
            # silently lagging a cycle behind its own plan.
            jaw_ramp_rad=float(np.asarray(self.step_size, dtype=np.float64)[6]),
            operator_timeout_steps=self.grasp_confirm_timeout_steps,
            approach_success_trans_cm=float(self.threshold_trans),
            approach_success_rot_deg=float(np.degrees(self.threshold_angle)),
        )
        self.sequencer = GraspLiftSequencer(
            plan,
            approach_controller or D2Controller(staged=True),
            cfg,
            limits or SafetyLimits(),
            jaw_baseline=None,  # simulation has ground truth; evidence is a diagnostic
            confirm_callback=self._ground_truth_confirm,
        )
        self.sequencer.begin(state)
        return self.sequencer

    def action_from_command(self, command, measured_vec7):
        """Turn the sequencer's absolute command into the env's [-1,1]^7 action."""
        target = np.concatenate(
            [command.pose.p, command.pose.to_vec7()[3:6], [float(command.jaw_rad)]]
        )
        delta = target - np.asarray(measured_vec7, dtype=np.float64)
        delta[3:6] = (delta[3:6] + np.pi) % (2.0 * np.pi) - np.pi
        return np.clip(delta / np.asarray(self.step_size, dtype=np.float64), -1.0, 1.0)

    def run_grasp_lift(self, approach_controller=None, max_cycles=800, verbose=True):
        """Run one full approach -> close -> confirm -> lift episode.

        Returns the sequencer summary, with the simulator's ground-truth grasp
        state recorded alongside the jaw evidence so the two can be compared.
        """
        sequencer = self.build_sequencer(approach_controller)
        trace = []
        phase = None

        for _ in range(max_cycles):
            state = self.arm_state()
            if state is None:
                self.scene_manager.step()
                continue

            step = sequencer.step(state)
            record = step.as_dict()
            record["needle_grasped_ground_truth"] = self.needle_grasped()
            trace.append(record)

            if verbose and (step.phase != phase or step.events):
                phase = step.phase
                print(
                    f"[grasp_lift] {step.index:4d} {step.phase:<14s} "
                    f"err {step.trans_err_cm:6.2f} cm / {step.rot_err_deg:6.2f} deg "
                    f"grasped={record['needle_grasped_ground_truth']}"
                )
                for event in step.events:
                    print(f"[grasp_lift]      * {event}")

            if step.done:
                break

            measured_vec7 = self.measured_vec7()
            action = self.action_from_command(step.command, measured_vec7)
            super(SRC_approach, self).step(action)

        summary = sequencer.summary()
        summary["needle_grasped_ground_truth"] = self.needle_grasped()
        summary["lift_source"] = self.lift_source
        summary["trace"] = trace
        # In simulation, and only in simulation, the grasp really is verified.
        summary["grasp_verified"] = bool(summary["needle_grasped_ground_truth"])
        summary["grasp_verification_note"] = (
            "Verified by the AMBF finger ghost sensor. No equivalent signal "
            "exists on real dVRK hardware; the real run reports "
            "grasp_verified=False by construction."
        )
        self.last_sequence_summary = summary
        return summary

    def succeeded(self) -> bool:
        """Pose reached, needle actually attached, and the lift completed."""
        summary = self.last_sequence_summary
        if summary is None:
            return False
        return bool(
            summary["phase"] == PHASE_DONE
            and summary["reason"] == "success"
            and summary["needle_grasped_ground_truth"]
        )
