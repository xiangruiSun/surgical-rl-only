from PyKDL import Frame, Rotation, Vector, Twist
import numpy as np
import time
from threading import Lock
from ros_abstraction_layer import ral
from ambf_msgs.msg import RigidBodyState


def pose_msg_to_frame(msg):
    """Convert ROS Pose message to PyKDL Frame"""
    if msg is None:
        return Frame()
    
    p = Vector(msg.position.x,
               msg.position.y,
               msg.position.z)

    R = Rotation.Quaternion(msg.orientation.x,
                            msg.orientation.y,
                            msg.orientation.z,
                            msg.orientation.w)

    return Frame(R, p)


class Scene:
    def __init__(self, ral_instance):
        """
        Initialize Scene with RAL instance to track scene objects.
        
        :param ral_instance: RAL instance for ROS communication
        """
        self.ral = ral_instance
        self._state_lock = Lock()
        
        # Object poses (stored as PyKDL Frames)
        self._needle_pose = Frame()
        self._entry1_pose = Frame()
        self._entry2_pose = Frame()
        self._entry3_pose = Frame()
        self._entry4_pose = Frame()
        self._exit1_pose = Frame()
        self._exit2_pose = Frame()
        self._exit3_pose = Frame()
        self._exit4_pose = Frame()
        
        # Setup ROS subscribers for scene objects
        self._setup_ros_interface()
        time.sleep(0.5)

    def _setup_ros_interface(self):
        """Setup ROS subscribers for all scene objects"""
        # Needle
        self.ral.subscriber('/ambf/env/phantom/Needle/State', RigidBodyState, 
                           self._needle_cb, queue_size=1)
        # Entry points
        self.ral.subscriber('/ambf/env/phantom/Entry1/State', RigidBodyState, 
                           self._entry1_cb, queue_size=1)
        self.ral.subscriber('/ambf/env/phantom/Entry2/State', RigidBodyState, 
                           self._entry2_cb, queue_size=1)
        self.ral.subscriber('/ambf/env/phantom/Entry3/State', RigidBodyState, 
                           self._entry3_cb, queue_size=1)
        self.ral.subscriber('/ambf/env/phantom/Entry4/State', RigidBodyState, 
                           self._entry4_cb, queue_size=1)
        # Exit points
        self.ral.subscriber('/ambf/env/phantom/Exit1/State', RigidBodyState, 
                           self._exit1_cb, queue_size=1)
        self.ral.subscriber('/ambf/env/phantom/Exit2/State', RigidBodyState, 
                           self._exit2_cb, queue_size=1)
        self.ral.subscriber('/ambf/env/phantom/Exit3/State', RigidBodyState, 
                           self._exit3_cb, queue_size=1)
        self.ral.subscriber('/ambf/env/phantom/Exit4/State', RigidBodyState, 
                           self._exit4_cb, queue_size=1)

    # Callback functions
    def _needle_cb(self, msg):
        with self._state_lock:
            self._needle_pose = pose_msg_to_frame(msg.pose)

    def _entry1_cb(self, msg):
        with self._state_lock:
            self._entry1_pose = pose_msg_to_frame(msg.pose)

    def _entry2_cb(self, msg):
        with self._state_lock:
            self._entry2_pose = pose_msg_to_frame(msg.pose)

    def _entry3_cb(self, msg):
        with self._state_lock:
            self._entry3_pose = pose_msg_to_frame(msg.pose)

    def _entry4_cb(self, msg):
        with self._state_lock:
            self._entry4_pose = pose_msg_to_frame(msg.pose)

    def _exit1_cb(self, msg):
        with self._state_lock:
            self._exit1_pose = pose_msg_to_frame(msg.pose)

    def _exit2_cb(self, msg):
        with self._state_lock:
            self._exit2_pose = pose_msg_to_frame(msg.pose)

    def _exit3_cb(self, msg):
        with self._state_lock:
            self._exit3_pose = pose_msg_to_frame(msg.pose)

    def _exit4_cb(self, msg):
        with self._state_lock:
            self._exit4_pose = pose_msg_to_frame(msg.pose)

    # Get pose methods (thread-safe)
    def needle_measured_cp(self):
        with self._state_lock:
            return self._needle_pose

    def entry1_measured_cp(self):
        with self._state_lock:
            return self._entry1_pose

    def entry2_measured_cp(self):
        with self._state_lock:
            return self._entry2_pose

    def entry3_measured_cp(self):
        with self._state_lock:
            return self._entry3_pose

    def entry4_measured_cp(self):
        with self._state_lock:
            return self._entry4_pose

    def exit1_measured_cp(self):
        with self._state_lock:
            return self._exit1_pose

    def exit2_measured_cp(self):
        with self._state_lock:
            return self._exit2_pose

    def exit3_measured_cp(self):
        with self._state_lock:
            return self._exit3_pose

    def exit4_measured_cp(self):
        with self._state_lock:
            return self._exit4_pose