import time
import numpy as np
from PyKDL import Frame, Rotation, Vector
from RL.utils.utils import convert_mat_to_frame
from RL.utils.psm_arm import PSM
from RL.utils.ecm_arm import ECM
from RL.utils.scene import Scene
from RL.utils.needle import Needle
from RL.utils.world_manager import WorldManager


class SceneManager:
    """Manages simulation initialization, PSM arms, and needle handling"""
    
    def __init__(self, env, ral_instance):
        self.env = env
        self.psm_list = []
        self.psm_goal_list = []
        self.jaw_angle_list = []
        
        # step size DR manage
        self.stepDR = self.env.stepDR
        self.pct = 1.0  # 100% variation for step size domain randomization
        
        # Initialize world manager
        self.world_manager = WorldManager(
            ral_instance,
            synchronous=bool(getattr(env, "synchronous_physics", False)),
            steps_per_action=int(getattr(env, "physics_steps_per_action", 10)),
            barrier_timeout_s=float(getattr(env, "physics_barrier_timeout_s", 2.0)),
        )
        self.world_manager.reset()
        time.sleep(0.5)
        self.scene = Scene(ral_instance)
        
        # Initialize arms
        self.psm1 = PSM(ral_instance, 'psm1')
        self.psm2 = PSM(ral_instance, 'psm2')
        self.psm_list = [self.psm1, self.psm2]
        for psm in self.psm_list:
            if not psm.wait_for_state(timeout=5.0):
                raise RuntimeError(
                    f"Timed out waiting for {psm.name} baselink state before base placement"
                )
        self.ecm = ECM(ral_instance, 'cameraL')
        self.base_camera_pose = self.ecm.measured_cp()  # Store initial camera pose for resets
        #self.camera_view_reset(True)

        # Initialize needle
        self.needle = Needle(ral_instance)
        if not self.needle.wait_for_state(timeout=5.0):
            raise RuntimeError(
                "Timed out waiting for the needle state before rest-pose calibration"
            )
        # One-time natural-rest calibration.  The simulator starts with the
        # needle already resting on the pad, so this measured frame supplies
        # both the placement origin and the non-flipped SO(3) reference.
        needle_rest_pose = self.needle.get_pose()
        self.needle_init_pos = Vector(
            needle_rest_pose.p.x(),
            needle_rest_pose.p.y(),
            needle_rest_pose.p.z(),
        )
        self.needle_rest_quaternion = tuple(
            float(v) for v in needle_rest_pose.M.GetQuaternion()
        )
        self.needle_rest_rotation = Rotation.Quaternion(
            *self.needle_rest_quaternion
        )
        
        # Move PSM baselinks
        self.psm1.move_base(Vector(0.14, 0.34, 0.8))
        self.psm2.move_base(Vector(-0.08, 0.34, 0.8))

        # Set initial positions
        self.init_psm1 = np.array([ 0.04629208,0.00752399,-0.08173992,-3.598019,-0.05762508,1.2738742,0.8],dtype=np.float32)
        self.init_psm2 = np.array([-0.03721037,  0.01213105, -0.08036895, -2.7039163, 0.07693613, 2.0361109, 0.8],dtype=np.float32)

        self.psm_goal_list = [self.init_psm1.copy(), self.init_psm2.copy()]
        
        # Move to initial positions
        self.psm_step(self.init_psm1, 1)
        time.sleep(0.5)
        self.psm_step(self.init_psm2, 2)
        time.sleep(0.5)
        
    def step(self):
        """Perform a simulation step"""

        jaw1_angle = self.psm_goal_list[0][-1]
        jaw2_angle = self.psm_goal_list[1][-1]
        self.jaw_angle_list = [jaw1_angle, jaw2_angle]
        
        if self.world_manager.synchronous:
            self.psm_step(self.psm_goal_list[self.env.psm_idx-1], self.env.psm_idx)
            self.world_manager.update()
        else:
            self.world_manager.update()
            self.psm_step(self.psm_goal_list[self.env.psm_idx-1], self.env.psm_idx)
    
    def env_reset(self):
        """Reset the simulation state"""
        self.needle.release()
        time.sleep(0.5)
        self.psm1.deactuate()
        self.psm2.deactuate()
        self.psm_goal_list[0] = np.copy(self.init_psm1)
        self.psm_goal_list[1] = np.copy(self.init_psm2)
        psm2_offset = getattr(self.env, "psm2_reset_offset", None)
        if psm2_offset is not None:
            self.psm_goal_list[1] = (
                np.asarray(self.psm_goal_list[1], dtype=np.float32)
                + np.asarray(psm2_offset, dtype=np.float32)
            )
            self.psm_goal_list[1][-1] = float(np.clip(self.psm_goal_list[1][-1], 0.0, 1.0))
            print("Applied PSM2 reset offset:", np.asarray(psm2_offset, dtype=np.float32))
        self.psm_step(self.psm_goal_list[0], 1)
        self.psm_step(self.psm_goal_list[1], 2)
        if self.world_manager.synchronous:
            self.world_manager.update()
        else:
            time.sleep(0.5)
        if getattr(self.env, "randomize_psm_reset", True):
            self.randomize_psm_pos()

        self.world_manager.reset()
        self.camera_view_reset(True)

        # Re-apply PSM commanded goals after world reset.
        # world_manager.reset() may reset AMBF object/joint states, while
        # psm_goal_list still stores the randomized commanded goals.
        # Keep reset observation semantics as commanded psm_goal_list, but
        # make the simulator measured pose catch up to it before policy starts.
        self.psm_step(self.psm_goal_list[0], 1)
        self.psm_step(self.psm_goal_list[1], 2)
        if self.world_manager.synchronous:
            self.world_manager.update()

        for settle_i in range(20):
            self.psm_step(self.psm_goal_list[0], 1)
            self.psm_step(self.psm_goal_list[1], 2)
            self.world_manager.update()
            if not self.world_manager.synchronous:
                time.sleep(0.02)

        try:
            measured = self.psm2.measured_cp()
            if measured is not None:
                from RL.utils.utils import convert_mat_to_frame, frame_to_vector
                measured_vec6 = frame_to_vector(convert_mat_to_frame(measured))
                measured_vec7 = np.append(measured_vec6, float(self.psm2.get_jaw_angle())).astype(np.float32)
                commanded_vec7 = np.array(self.psm_goal_list[1], dtype=np.float32)

                def _wrap_to_pi(x):
                    return (x + np.pi) % (2 * np.pi) - np.pi

                print(
                    "ENV_RESET_REAPPLY_DIAG:",
                    "commanded_psm2=", commanded_vec7,
                    "measured_psm2=", measured_vec7,
                    "trans_delta_cm=", np.linalg.norm((commanded_vec7[:3] - measured_vec7[:3]) * 100.0),
                    "rpy_delta_deg=", np.degrees(_wrap_to_pi(commanded_vec7[3:6] - measured_vec7[3:6])),
                    "jaw_delta=", commanded_vec7[6] - measured_vec7[6],
                )
            else:
                print("ENV_RESET_REAPPLY_DIAG: measured_cp is None")
        except Exception as exc:
            print("ENV_RESET_REAPPLY_DIAG error:", exc)

        if self.stepDR:
            self.step_size_update()
            print("Randomized step size:", self.env.step_size)
        time.sleep(1.0)

    def randomize_psm_pos(self):
        psm_random_range = np.asarray(
            getattr(
                self.env,
                "psm_reset_random_range",
                np.array([0.005, 0.005, 0.005, 0.5, 0.5, 0.5, 0.3], dtype=np.float32),
            ),
            dtype=np.float32,
        )
        # add random offsets in [-range, +range] for each DOF
        noise0 = np.random.uniform(-psm_random_range, psm_random_range).astype(np.float32)
        noise1 = np.random.uniform(-psm_random_range, psm_random_range).astype(np.float32)
        self.psm_goal_list[0] = np.array(self.psm_goal_list[0], dtype=np.float32) + noise0
        self.psm_goal_list[1] = np.array(self.psm_goal_list[1], dtype=np.float32) + noise1
        # clamp jaw angles to a safe range (0.0 - 1.0)
        self.psm_goal_list[0][-1] = float(np.clip(self.psm_goal_list[0][-1], 0.0, 1.0))
        self.psm_goal_list[1][-1] = float(np.clip(self.psm_goal_list[1][-1], 0.0, 1.0))

        print("Randomized PSM2 noise:", noise1)

        self.psm_step(self.psm_goal_list[0], 1)
        self.psm_step(self.psm_goal_list[1], 2) 

    
    def step_size_update(self):
        """Update step size based on current randomization settings"""
        if not self.stepDR:
            return
        
        base = np.asarray(self.env.base_step_size)
        factors = np.random.uniform(1.0 - self.pct, 1.0 + self.pct, size=base.shape).astype(base.dtype)
        self.env.step_size = base * factors


    def camera_view_reset(self, reset_noise=False):
        """Reset camera view"""
        camera_pose = self.base_camera_pose
        rotation = camera_pose.M
        base_vec = camera_pose.p

        if reset_noise:
            vector = base_vec + Vector(
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02)
            )
        else:
            vector = base_vec
        
        ecm_pos_origin = Frame(rotation, vector)
        self.ecm.servo_cp(ecm_pos_origin)
    
    def psm_step(self, obs, psm_idx):
        """Move PSM to specified position"""
        X = obs[0]
        Y = obs[1]
        Z = obs[2]
        Roll = obs[3]
        Pitch = obs[4]
        Yaw = obs[5]
        Jaw_angle = obs[6]

        T_goal = Frame(Rotation.RPY(Roll, Pitch, Yaw), Vector(X, Y, Z))
        self.psm_list[psm_idx-1].servo_cp(T_goal)
        self.psm_list[psm_idx-1].set_jaw_angle(Jaw_angle)
    
    def psm_step_move(self, obs, psm_idx, execute_time=0.5):
        """Smoothly move PSM to position"""
        X = obs[0]
        Y = obs[1]
        Z = obs[2]
        Roll = obs[3]
        Pitch = obs[4]
        Yaw = obs[5]
        Jaw_angle = obs[6]

        T_goal = Frame(Rotation.RPY(Roll, Pitch, Yaw), Vector(X, Y, Z))
        self.psm_list[psm_idx-1].move_cp(T_goal, execute_time)
        self.psm_list[psm_idx-1].set_jaw_angle(Jaw_angle)
            
    def needle_randomization(self):
        """
        Initialize needle at random positions in the world
        """

        self.needle.release()  # Ensure needle is released before randomizing

        random_range = self.env.random_range
        # The needle is a free dynamic body (may rest on the suturing pad), so the
        # default 0.1 mm / 1 s set_pose convergence check is too strict after a large
        # rz rotation. Loosen the tolerance, give it more time, and retry with a fresh
        # sample instead of crashing the whole reset on a single transient miss. The
        # goal is computed from the needle's measured pose afterwards, so a ~1 mm
        # settling tolerance does not corrupt the desired_goal.
        max_attempts = int(getattr(self.env, "needle_reset_max_attempts", 5))
        set_pose_timeout = float(getattr(self.env, "needle_set_pose_timeout", 3.0))
        set_pose_tol = float(getattr(self.env, "needle_set_pose_tol", 1.0e-3))
        resolved_pose_tol = float(
            getattr(self.env, "needle_resolved_pose_tol", 5.0e-3)
        )

        # Draw exactly one target per episode. Retries must retry delivery of
        # that target; drawing a new target on every transport failure consumes
        # extra RNG state and makes the reset distribution timing-dependent.
        origin_p = Vector(
            self.needle_init_pos.x(),
            self.needle_init_pos.y(),
            self.needle_init_pos.z(),
        )
        requested_offset = getattr(self.env, "needle_reset_offset", None)
        if requested_offset is None:
            random_x = np.random.uniform(-random_range[0], random_range[0])
            random_y = np.random.uniform(-random_range[1], random_range[1])
            random_rz = np.random.uniform(-random_range[2], random_range[2])
        else:
            requested_offset = np.asarray(requested_offset, dtype=np.float32)
            if requested_offset.shape != (3,):
                raise ValueError("needle_reset_offset must be [x_m, y_m, rz_rad]")
            if np.any(np.abs(requested_offset) > random_range + 1.0e-9):
                raise ValueError("needle_reset_offset exceeds needle random_range")
            random_x, random_y, random_rz = requested_offset.tolist()
            print(
                "NEEDLE_RESET_TARGET_OFFSET:",
                {"x_m": random_x, "y_m": random_y, "rz_rad": random_rz},
            )
        origin_p[0] += random_x
        origin_p[1] += random_y
        # Rotate within the pad plane relative to the needle's measured natural
        # rest attitude.  A pure world Rz command discards that attitude and
        # injects a large contact incompatibility when the command is released.
        new_rot = self.needle_rest_rotation * Rotation.RotZ(float(random_rz))
        needle_pos_new = Frame(new_rot, origin_p)

        # Record the intended placement so a downstream reset-validity gate can
        # compare the settled needle pose against what randomization asked for
        # (R_rest * Rz(target_rz) gives the expected non-flipped orientation).
        self.last_needle_target_rz = float(random_rz)
        self.last_needle_target_xy = (float(origin_p[0]), float(origin_p[1]))
        self.last_needle_target_frame = needle_pos_new

        last_delta_mm = None
        for attempt in range(max_attempts):
            if self.needle.set_pose(
                needle_pos_new,
                timeout=set_pose_timeout,
                position_tolerance=set_pose_tol,
                step_callback=(
                    (lambda: self.world_manager.update(requested_steps=1))
                    if self.world_manager.synchronous
                    else None
                ),
                # Keep the Cartesian command active.  Approach.reset() performs
                # a quasi-static hold, releases once, then verifies free-body
                # stability before accepting the placement.
                release=False,
            ):
                return

            reached = self.needle.get_pos()
            last_delta_mm = float(np.linalg.norm(np.array([
                reached.x() - origin_p[0],
                reached.y() - origin_p[1],
                reached.z() - origin_p[2],
            ])) * 1000.0)
            print(
                "NEEDLE_RANDOMIZATION_RETRY:",
                "attempt=", attempt + 1, "/", max_attempts,
                "tol_mm=", round(set_pose_tol * 1000.0, 4),
                "timeout_s=", set_pose_timeout,
                "final_delta_mm=", round(last_delta_mm, 4),
            )
            # A free needle can settle a few millimetres onto the pad after the
            # position command is released.  The episode goal is sampled from
            # this measured, contact-resolved pose, so accepting a bounded
            # residual is physically consistent.  Keep a hard guard so a lost
            # or stale simulator command cannot silently pass reset.
            if last_delta_mm <= resolved_pose_tol * 1000.0:
                resolved = self.needle.get_pose()
                print(
                    "NEEDLE_RANDOMIZATION_RESOLVED_POSE_ACCEPTED:",
                    "target_xyz_m=", [
                        float(origin_p[0]), float(origin_p[1]), float(origin_p[2])
                    ],
                    "resolved_xyz_m=", [
                        float(resolved.p.x()),
                        float(resolved.p.y()),
                        float(resolved.p.z()),
                    ],
                    "residual_mm=", round(last_delta_mm, 4),
                    "acceptance_limit_mm=", round(resolved_pose_tol * 1000.0, 4),
                    "reason=contact_resolved_pose_outside_transport_tolerance",
                )
                return

        raise RuntimeError(
            "Needle pose randomization did not reach the commanded simulator pose "
            f"after {max_attempts} attempts (last position delta "
            f"{last_delta_mm:.3f} mm, transport tolerance "
            f"{set_pose_tol * 1000.0:.3f} mm, resolved-pose acceptance limit "
            f"{resolved_pose_tol * 1000.0:.3f} mm)"
        )
        
    def entry_goal_evaluator(self,deg=120,dev_trans=[0,0,0],dev_Yangle = 0.0,idx=2,noise=False):
        rotation_noise = Rotation.RotY(np.deg2rad(dev_Yangle))
        translation_noise = Vector(0, 0, 0)
        noise_in_entry = Frame(rotation_noise, translation_noise)

        entry_in_world = self.scene.entry1_measured_cp()
        entry_in_world.p[1] -= 0.005
        entry_in_world.p[2] += 0.006
        noise_in_world = entry_in_world*noise_in_entry
        entry_in_base = self.psm_list[idx-1].get_T_w_b()*noise_in_world # entry with angle deviation

        rotation_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        rotation_tip_in_entry = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_tip_in_entry = Vector(dev_trans[0],dev_trans[1],dev_trans[2])
        tip_in_entry = Frame(rotation_tip_in_entry,trans_tip_in_entry)


        tip_in_world = self.needle.get_pose_angle(deg)
        gripper_in_world = self.psm_list[idx-1].get_T_b_w() * convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        gripper_in_tip = tip_in_world.Inverse() * gripper_in_world

        gripper_in_base = entry_in_base * tip_in_entry * gripper_in_tip
        array_insert = self.Frame2Vec(gripper_in_base)
        array_insert = np.append(array_insert,0.0)
        if noise:
            ranges = np.array([0.001, 0.001, 0.001, np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), 0])
            random_noise = np.random.uniform(-ranges, ranges)
            array_insert += random_noise
        return array_insert
    
    def insert_goal_evaluator(self,deg=120,dev=[0,0,0],idx=2):
        exit_in_world = self.scene.exit1_measured_cp()
        exit_in_base = self.psm_list[idx-1].get_T_w_b()*exit_in_world

        # entry_pos.Inverse() to obtain the inverse transformation matrix
        # rotation_matrix = np.array([[0,-1,0],[0,0,1],[-1,0,0]]).astype(np.float32)
        rotation_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]]).astype(np.float32)
        rotation_front_in_exit = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_front_in_exit = Vector(dev[0],dev[1],dev[2])
        front_in_exit = Frame(rotation_front_in_exit,trans_front_in_exit)

        front_in_world = self.needle.get_pose_angle(deg)
        gripper_in_world = self.psm_list[idx-1].get_T_b_w() * convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        gripper_in_front = front_in_world.Inverse() * gripper_in_world

        gripper_in_base = exit_in_base * front_in_exit * gripper_in_front
        array_insert = self.Frame2Vec(gripper_in_base)
        array_insert = np.append(array_insert,0.0)
        return array_insert
    
    def handover_goal_evaluator(self,deg=110,dev=[0,0,0],idx=1):
        exit_in_world = self.scene.exit1_measured_cp()
        rotation_decrease_y = Rotation.RotY(-np.deg2rad(50))
        new_rotation = exit_in_world.M * rotation_decrease_y
        handover_in_world = Frame(new_rotation, Vector(exit_in_world.p[0] + 0.03, exit_in_world.p[1], exit_in_world.p[2] + 0.03))
        exit_in_base = self.psm_list[idx-1].get_T_w_b()*handover_in_world

        rotation_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]]).astype(np.float32)
        rotation_front_in_exit = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_front_in_exit = Vector(dev[0],dev[1],dev[2])
        front_in_exit = Frame(rotation_front_in_exit,trans_front_in_exit)

        front_in_world = self.needle.get_pose_angle(deg)
        gripper_in_world = self.psm_list[idx-1].get_T_b_w() * convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        gripper_in_front = front_in_world.Inverse() * gripper_in_world

        gripper_in_base = exit_in_base * front_in_exit * gripper_in_front
        array_handover = self.Frame2Vec(gripper_in_base)
        array_handover = np.append(array_handover, 0.0)
        return array_handover
    
    # Overridden entry goal evaluator for place subtask
    def place_entry_goal_evaluator(self,idx = 2):
        self.entry_w = self.scene.entry1_measured_cp()
        entry_pos = self.psm_list[idx-1].get_T_w_b()*self.entry_w
        rotation_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        rotation_entry = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        T_tip_base = self.needle.get_tip_pose()
        T_gripper_base = self.psm_list[idx-1].get_T_b_w() * self.psm_list[idx-1].measured_cp()
        T_gripper_tip = T_tip_base.Inverse() * T_gripper_base

        T_insert = entry_pos
        T_insert.M *= rotation_entry
        T_insert = T_insert * T_gripper_tip
        array_insert = self.Frame2Vec(T_insert)
        array_insert = np.append(array_insert,0.0)
        return array_insert
    
    def needle_random_grasping_evaluator(self,lift_height):
        self.random_degree = np.random.uniform(12, 15)
        self.grasping_pos = self.needle.get_random_grasp_point()
        needle_rot = self.grasping_pos.M
        needle_trans_lift = Vector(self.grasping_pos.p.x(),self.grasping_pos.p.y(),self.grasping_pos.p.z()+lift_height)
        needle_goal_lift = Frame(needle_rot, needle_trans_lift)

        T_calibrate = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_calibrate[:3, :3]

        rotation_calibrate = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        needle_goal_lift.M = needle_goal_lift.M * rotation_calibrate # To be tested
        print("needle_goal_lift:", needle_goal_lift)
        print("twb:", self.psm2.get_T_w_b())
        psm_goal_lift = self.psm2.get_T_w_b()*needle_goal_lift
        print("psm_goal_lift:", psm_goal_lift)

        T_goal = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_goal[:3, :3]

        rotation = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        psm_goal_lift.M = psm_goal_lift.M*rotation
        array_goal_base = self.Frame2Vec(psm_goal_lift)
        array_goal_base = np.append(array_goal_base,0.0)
        print("Needle grasping goal in base frame:", array_goal_base)
        return array_goal_base
    
    def needle_goal_evaluator(self,lift_height=0.007, psm_idx=2, deg_angle = None):
        '''
        Evaluate the target goal for needle grasping in Robot frame.
        '''

        if deg_angle is None:
            grasp_in_World = self.needle.get_bm_pose()

        else:
            grasp_in_World = self.needle.get_pose_angle(deg_angle)

        lift_in_grasp_rot = Rotation(1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1)    
        lift_in_grasp_trans = Vector(0,0,lift_height)
        lift_in_grasp = Frame(lift_in_grasp_rot,lift_in_grasp_trans)

        if psm_idx == 2:
            gripper_in_lift_rot = Rotation(0, -1, 0,
                                            -1, 0, 0,
                                            0, 0, -1)
        else:
            gripper_in_lift_rot = Rotation(0, 1, 0,
                                            1, 0, 0,
                                            0, 0, -1)           

        gripper_in_lift_trans = Vector(0.0, 0.0, 0.0)
        gripper_in_lift = Frame(gripper_in_lift_rot, gripper_in_lift_trans)

        gripper_in_world = grasp_in_World * lift_in_grasp * gripper_in_lift
        T_w_b = self.psm_list[psm_idx-1].get_T_w_b()
        gripper_in_base = T_w_b * gripper_in_world
        

        array_goal_base = self.Frame2Vec(gripper_in_base)
        array_goal_base = np.append(array_goal_base, 0.0)
        if getattr(self.env, "goal_transform_diag", False):
            base_roundtrip = self.psm_list[psm_idx-1].get_T_b_w() * T_w_b
            print(
                "GOAL_TRANSFORM_DIAG:",
                "needle_world=", self.Frame2Vec(self.needle.get_pose()),
                "grasp_world=", self.Frame2Vec(grasp_in_World),
                "base_roundtrip=", self.Frame2Vec(base_roundtrip),
                "goal_base=", array_goal_base,
            )
        return array_goal_base
    
    def needle_multigoal_evaluator(self, lift_height=0.007, psm_idx=2, start_degree=5, end_degree=30, num_points=25):
        """
        Evaluate the multiple allowed goal grasping points.
        """
        interpolated_transforms = self.needle.get_interpolated_transforms(start_degree, end_degree, num_points)
        goals = []

        for transform in interpolated_transforms:
            grasp_in_World = transform

            lift_in_grasp_rot = Rotation(1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1)
            lift_in_grasp_trans = Vector(0, 0, lift_height)
            lift_in_grasp = Frame(lift_in_grasp_rot, lift_in_grasp_trans)

            if psm_idx == 2:
                gripper_in_lift_rot = Rotation(0, -1, 0,
                                               -1, 0, 0,
                                               0, 0, -1)
            else:
                gripper_in_lift_rot = Rotation(0, 1, 0,
                                               1, 0, 0,
                                               0, 0, -1)

            gripper_in_lift_trans = Vector(0.0, 0.0, 0.0)
            gripper_in_lift = Frame(gripper_in_lift_rot, gripper_in_lift_trans)

            gripper_in_world = grasp_in_World * lift_in_grasp * gripper_in_lift
            gripper_in_base = self.psm_list[psm_idx - 1].get_T_w_b() * gripper_in_world

            array_goal_base = self.Frame2Vec(gripper_in_base)
            array_goal_base = np.append(array_goal_base, 0.0)
            goals.append(array_goal_base)

        return goals


    def Frame2Vec(self,goal_frame,bound = True):
        """
        Convert Frame variables into vector forms.
        """
        X_goal = goal_frame.p.x()
        Y_goal = goal_frame.p.y()
        Z_goal = goal_frame.p.z()
        rot_goal = goal_frame.M
        roll_goal, pitch_goal, yaw_goal = rot_goal.GetRPY()
        if bound:
            if roll_goal <= np.deg2rad(-360):
                roll_goal += 2 * np.pi
            elif roll_goal > np.deg2rad(0):
                roll_goal -= 2 * np.pi
        array_goal = np.array([X_goal, Y_goal, Z_goal, roll_goal, pitch_goal, yaw_goal], dtype=np.float32)
        return array_goal
    
    def approach_and_grasp(self):
        # Approach and grasp the needle
        self.needle_obs = self.needle_random_grasping_evaluator(0.0007)
        self.needle_obs = np.append(self.needle_obs,0.8)
        self.psm_step_move(self.needle_obs,2)
        time.sleep(10.6)
        self.needle_obs[-1] = 0.0
        self.psm_step(self.needle_obs,2)
        time.sleep(0.5)
        self.psm2.actuate("Needle")
        self.needle.release()  # Ensure needle is grasped by setting forces to zero
    
    def place_at_entry(self):
        # Place the needle at the entry
        self.entry_obs = self.entry_goal_evaluator(idx=2,dev_trans=[0,0,0.001],noise=False) # Close noise in this case
        self.adjusted_entry_obs = np.copy(self.entry_obs)
        self.adjusted_entry_obs[1] = self.adjusted_entry_obs[1] + 0.003
        self.adjusted_entry_obs[2] = self.adjusted_entry_obs[2] + 0.003
        self.psm_step_move(self.adjusted_entry_obs,2,execute_time=1.2)
        time.sleep(1.4) 
        self.psm_step_move(self.entry_obs,2,execute_time=1)
        time.sleep(1.8)

    def insert_needle(self):
        # Insert the needle
        self.insert_obs = self.insert_goal_evaluator(90,[0.002,0,0])
        self.psm_step_move(self.insert_obs,2,execute_time=0.7)
        self.psm_goal_list[1] = np.copy(self.insert_obs)
        time.sleep(1)

    def regrasp_needle(self):        
        # Regrasp the needle
        self.regrasp_obs = self.needle_goal_evaluator(deg_angle=105,lift_height=0.005,psm_idx=1)
        self.regrasp_obs[-1] = 0.8
        self.psm_step_move(self.regrasp_obs,1,execute_time=0.8)
        time.sleep(1)
        self.regrasp_obs[-1] = 0.0
        self.psm_step(self.regrasp_obs,1)
        time.sleep(0.4)
        self.psm1.actuate("Needle")
        self.needle.release()  # Ensure needle is grasped by setting forces to zero
        self.psm_goal_list[1][-1] = 0.8
        self.psm_step(self.psm_goal_list[1],2)
        time.sleep(0.3)
            
