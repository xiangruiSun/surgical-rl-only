import time
import numpy as np
from threading import Thread, Lock, RLock
from ambf_msgs.msg import RigidBodyCmd, RigidBodyState, ActuatorCmd, GhostObjectState
from geometry_msgs.msg import Pose
from PyKDL import Vector, Frame, Rotation
from RL.utils.kinematics.kinematics import PSMKinematicSolver
from RL.utils.kinematics.DH import enforce_limits
from RL.utils.utils import convert_mat_to_frame
from RL.utils.interpolation import Interpolation

_psm_global_compute_lock = RLock()

class PSM:
    def __init__(self, ral_instance, name, env = "ambf/env"):
        self.name = name
        self.env = env
        self.ral_instance = ral_instance       
        
        self._kd = PSMKinematicSolver()
        self.interpolater = Interpolation()        
        self._thread_lock = Lock()
        self._cmd_lock = Lock()
        self._actuator_cmd_lock = Lock()
        self._state_lock = Lock()
        self._grasp_lock = Lock()
        
        self._T_b_w = None 
        self._T_w_b = None 
        self._measured_jp = None
        self._measured_jv = None
        self._measured_jaw_angle = None
        self.orientation = None
        self._cmd = None
        self._actuator_cmd = None
        
        self.jaw_angle = 0.5
        self.grasp_actuation_jaw_angle = 0.05
        self.graspable_objs_prefix = ["Needle", "Thread", "Puzzle"]
        self.grasped = [False]*len(self.graspable_objs_prefix) 
        self.grasped_obj_name = None
        self.left_finger_sensed_objs = []
        self.right_finger_sensed_objs = []
        
        self._setup_ros_interface()
        
        cmd_thread = Thread(target=self.send_cmds, daemon=True)
        cmd_thread.start()

    def _setup_ros_interface(self):
        topic = f'{self.env}/{self.name}'
        self.psm_pub = self.ral_instance.publisher(f'/{topic}/baselink/Command', RigidBodyCmd, queue_size=1)
        self.psm_sub = self.ral_instance.subscriber(f'/{topic}/baselink/State', RigidBodyState, self._state_callback)
        self.actuator_pub = self.ral_instance.publisher(f'/{self.env}/ghosts/{self.name}/Actuator0/Command', ActuatorCmd, queue_size=1)
        self.left_finger_ghost = self.ral_instance.subscriber(f'/{self.env}/ghosts/{self.name}/left_finger_ghost/State', GhostObjectState, self._left_finger_callback)
        self.right_finger_ghost = self.ral_instance.subscriber(f'/{self.env}/ghosts/{self.name}/right_finger_ghost/State', GhostObjectState, self._right_finger_callback)

    def _state_callback(self, msg):
        v = Vector(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        quat = msg.pose.orientation
        r = Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)
        with self._state_lock:
            self._T_b_w = Frame(r, v)
            self._T_w_b = self._T_b_w.Inverse()
            if len(msg.joint_positions) >= 6:
                self._measured_jp = list(msg.joint_positions)[:6]
            if len(msg.joint_velocities) >= 6:
                self._measured_jv = list(msg.joint_velocities)[:6]
            if len(msg.joint_positions) >= 7:
                self._measured_jaw_angle = float(msg.joint_positions[6])
                self.jaw_angle = self._measured_jaw_angle
            self.orientation = msg.pose.orientation
        with self._cmd_lock:
            needs_initial_command = self._cmd is None
        if needs_initial_command and self._measured_jp is not None:
            # Seed commands from simulator feedback. Publishing a jaw-only
            # command before this point would implicitly command arm joints 0-5
            # to zero and corrupt the reset pose.
            self.servo_jp(self._measured_jp)
            self.set_jaw_angle(self.jaw_angle)
        
    def _left_finger_callback(self, msg):
        with self._state_lock:
            self.left_finger_sensed_objs = [v.data for v in msg.sensed_objects]
    
    def _right_finger_callback(self, msg):
        with self._state_lock:
            self.right_finger_sensed_objs = [v.data for v in msg.sensed_objects]

    def get_T_b_w(self):
        with self._state_lock:
            if self._T_b_w is None:
                return Frame()
            return self._T_b_w

    def get_T_w_b(self):
        with self._state_lock:
            if self._T_w_b is None:
                return Frame()
            return self._T_w_b

    def wait_for_state(self, timeout=5.0):
        """Wait until the simulator has supplied the baselink pose."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._state_lock:
                if self._T_b_w is not None and self.orientation is not None:
                    return True
            time.sleep(0.01)
        return False

    def servo_cp(self, T_t_b):
        with _psm_global_compute_lock, self._state_lock:
            if type(T_t_b) in [np.matrix, np.ndarray]:
                T_t_b = convert_mat_to_frame(T_t_b)
            self.ik_solution = self._kd.compute_IK(T_t_b)
            self._ik_solution = enforce_limits(self.ik_solution, self._kd.JOINT_LIMITS_LOWER, self._kd.JOINT_LIMITS_UPPER)
            self.servo_jp(self._ik_solution)
        
    def move_cp(self, T_t_b, execute_time=0.5, control_rate=120):
        with _psm_global_compute_lock, self._state_lock:
            if type(T_t_b) in [np.matrix, np.ndarray]:
                T_t_b = convert_mat_to_frame(T_t_b)
            ik_solution = self._kd.compute_IK(T_t_b)
            self._ik_solution = enforce_limits(ik_solution, self.get_lower_limits(), self.get_upper_limits())
            self.move_jp(self._ik_solution, execute_time, control_rate)
    def servo_jp(self, jp_vec):
        with self._cmd_lock:
            msg = self._cmd if self._cmd is not None else RigidBodyCmd()
            # Ensure message has at least 8 joints (6 arm + 2 jaw)
            if len(msg.joint_cmds) < 8:
                msg.joint_cmds = list(msg.joint_cmds) + [0.0] * (8 - len(msg.joint_cmds))
            if len(msg.joint_cmds_types) < 8:
                msg.joint_cmds_types = list(msg.joint_cmds_types) + [RigidBodyCmd.TYPE_POSITION] * (8 - len(msg.joint_cmds_types))
            # Set arm joints 0-5, preserve jaw angle at 6-7
            for i in range(6):
                msg.joint_cmds[i] = float(jp_vec[i])
                msg.joint_cmds_types[i] = RigidBodyCmd.TYPE_POSITION
            self._cmd = msg
        
    def move_jp(self, jp_cmd, execute_time=0.5, control_rate=120):

        jp_cur = np.array(self._measured_jp[:6], dtype=np.float32)
        jp_cmd = np.array(jp_cmd[:6], dtype=np.float32)
        jv_cur = np.array(self._measured_jv[:6], dtype=np.float32)

        zero = np.zeros(6, dtype=np.float32)
        self.interpolater.compute_interpolation_params(jp_cur, jp_cmd, jv_cur, zero, zero, zero, 0, execute_time)
        trajectory_execute_thread = Thread(target=self._execute_trajectory, args=(self.interpolater, execute_time, control_rate,))
        self._force_exit_thread = True
        trajectory_execute_thread.start()
    
    def _execute_trajectory(self, trajectory_gen, execute_time, rate):
        self._thread_lock.acquire()
        self._force_exit_thread = False
        init_time = self.ral_instance.to_sec(self.ral_instance.now())
        rate_ctrl = self.ral_instance.create_rate(rate)
        while not self.ral_instance.is_shutdown() and not self._force_exit_thread:
            cur_time = self.ral_instance.to_sec(self.ral_instance.now()) - init_time
            if cur_time > execute_time:
                break
            val = trajectory_gen.get_interpolated_x(np.array(cur_time, dtype=np.float32))
            self.servo_jp(val)
            rate_ctrl.sleep()
        self._thread_lock.release()
        
    def measured_jp(self):
        """Get all measured joint positions from the current RigidBodyState message"""
        with self._state_lock:
            return self._measured_jp
    
    def measured_jv(self):
        """Get all measured joint velocities from the current RigidBodyState message"""
        with self._state_lock:
            return self._measured_jv
    
    def measured_cp(self):
        with self._state_lock:
            jp = None if self._measured_jp is None else list(self._measured_jp)
        if jp is None:
            return None
        jp.append(0.0)
        return self._kd.compute_FK(jp[:7], 7)
        
    def set_jaw_angle(self, jaw_angle):
        """Set the jaw angle (always include latest pose)"""
        self.jaw_angle = jaw_angle
        with self._cmd_lock:
            msg = self._cmd if self._cmd is not None else RigidBodyCmd()
            # Ensure message has at least 8 joints
            if len(msg.joint_cmds) < 8:
                msg.joint_cmds = list(msg.joint_cmds) + [0.0] * (8 - len(msg.joint_cmds))
            if len(msg.joint_cmds_types) < 8:
                msg.joint_cmds_types = list(msg.joint_cmds_types) + [RigidBodyCmd.TYPE_POSITION] * (8 - len(msg.joint_cmds_types))
            # Only update jaw joints 6-7, preserve arm joints 0-5
            msg.joint_cmds[6] = float(jaw_angle)
            msg.joint_cmds[7] = float(jaw_angle)
            msg.joint_cmds_types[6] = RigidBodyCmd.TYPE_POSITION
            msg.joint_cmds_types[7] = RigidBodyCmd.TYPE_POSITION
            self._cmd = msg
        self.run_grasp_logic(jaw_angle)

    def get_jaw_angle(self):
        with self._state_lock:
            return self.jaw_angle

    def measured_jaw_angle(self):
        """Return jaw feedback from State, never the locally written command cache."""
        with self._state_lock:
            return self._measured_jaw_angle
    
    def run_grasp_logic(self, jaw_angle):
        with self._state_lock:
            sensed_object_names = list(
                self.left_finger_sensed_objs + self.right_finger_sensed_objs
            )
        with self._grasp_lock:
            self._run_grasp_logic_locked(jaw_angle, sensed_object_names)

    def _run_grasp_logic_locked(self, jaw_angle, sensed_object_names):
        if jaw_angle < self.grasp_actuation_jaw_angle:
            if not self.grasped[0]:
                for gon in self.graspable_objs_prefix:
                    matches = [son for son in sensed_object_names if gon in son]
                    if matches:
                        self.grasped_obj_name = matches[0]
                        self.actuate(self.grasped_obj_name)
                        self.grasped[0] = True

        else:
            self.deactuate()
            self.grasped_obj_name = None
            if self.grasped[0] is True:
                print('Releasing Grasped Object')
            self.grasped[0] = False

    def grasp_status(self):
        """Return a thread-safe snapshot used for grasp completion decisions.

        ``needle_grasped`` can only become true through ``run_grasp_logic``:
        the jaw must cross the actuation threshold while a finger ghost sensor
        reports a Needle object.  It is therefore stronger than jaw closure
        alone and does not rely on the evaluator unconditionally attaching the
        needle.
        """
        with self._state_lock:
            left = list(self.left_finger_sensed_objs)
            right = list(self.right_finger_sensed_objs)
        with self._grasp_lock:
            grasped = bool(self.grasped[0])
            object_name = self.grasped_obj_name
        return {
            "grasped": grasped,
            "grasped_object": object_name,
            "needle_grasped": bool(
                grasped and object_name is not None and "Needle" in object_name
            ),
            "left_finger_objects": left,
            "right_finger_objects": right,
            "left_needle_contact": any("Needle" in name for name in left),
            "right_needle_contact": any("Needle" in name for name in right),
        }

    def actuate(self, name):
        with self._actuator_cmd_lock:
            cmd = self._actuator_cmd if self._actuator_cmd is not None else ActuatorCmd()
            cmd.body_name.data = name
            cmd.actuate = True
            self._actuator_cmd = cmd

    def deactuate(self):
        with self._actuator_cmd_lock:
            cmd = self._actuator_cmd if self._actuator_cmd is not None else ActuatorCmd()
            cmd.actuate = False
            cmd.use_sensor_data = False
            cmd.sensor_identifier.data = ""
            self._actuator_cmd = cmd
        
    def send_cmds(self):
        while True:
            with self._cmd_lock:
                if self._cmd is not None:
                    # Ask AMBF to publish joint state back on the /State topic.
                    # Without these flags RigidBodyState.joint_positions stays empty,
                    # so measured_cp() can never read the arm joints (measured frozen).
                    self._cmd.publish_joint_positions = True
                    self._cmd.publish_joint_names = True
                    self._cmd.publish_children_names = True
                    self.psm_pub.publish(self._cmd)
            with self._actuator_cmd_lock:
                if self._actuator_cmd is not None:
                    self.actuator_pub.publish(self._actuator_cmd)
            time.sleep(0.05)

    def move_base(self, pos_vector):
        with self._state_lock:
            orientation = self.orientation
        if orientation is None:
            raise RuntimeError(
                f"{self.name} baselink state is unavailable; refusing to overwrite its mounting orientation"
            )
        pose_cmd = Pose()
        pose_cmd.position.x = pos_vector[0]
        pose_cmd.position.y = pos_vector[1]
        pose_cmd.position.z = pos_vector[2]
        pose_cmd.orientation = orientation
        with self._cmd_lock:
            self._cmd = self._cmd if self._cmd is not None else RigidBodyCmd()
            self._cmd.cartesian_cmd_type = RigidBodyCmd.TYPE_POSITION
            self._cmd.pose = pose_cmd
        
    def get_lower_limits(self):
        return self._kd.JOINT_LIMITS_LOWER
    
    def get_upper_limits(self):
        return self._kd.JOINT_LIMITS_UPPER
