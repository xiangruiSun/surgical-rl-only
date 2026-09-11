import numpy as np
from gymnasium import spaces

class GymManager:
    """Manages observation space, action space, and reward calculations"""
    
    def __init__(self, env, reward_type="sparse", threshold=[0.5, np.deg2rad(30)]):
        self.env = env
        self.reward_type = reward_type
        
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Initialize observation and action spaces"""
        self.env.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            shape=(7,), dtype=np.float32
        )
        
        self.env.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=np.array([-np.inf] * 21, dtype=np.float32),
                high=np.array([np.inf] * 21, dtype=np.float32),
                shape=(21,), dtype=np.float32),
            "achieved_goal": spaces.Box(
                low=np.array([-np.inf]*7, dtype=np.float32),
                high=np.array([np.inf]*7, dtype=np.float32),
                shape=(7,), dtype=np.float32),
            "desired_goal": spaces.Box(
                low=np.array([-np.inf]*7, dtype=np.float32),
                high=np.array([np.inf]*7, dtype=np.float32),
                shape=(7,), dtype=np.float32)              
        })
        
    def update_observation(self, current):
        """
        Build a GoalEnv-style observation dict and compute reward/termination.

        The env is expected to provide:
        - `goal_obs`: desired goal vector (len 7)
        - `compute_reward(achieved, desired, info=None)`
        - `criteria()`: returns True when success condition is met
        - `timestep`, `max_timestep`
        """
        goal_obs = self.env.goal_obs
        current = np.array(current, dtype=np.float32)
        goal_obs = np.array(goal_obs, dtype=np.float32)

        if getattr(self.env, "align_obs_rpy_branch", False):
            # Snap the measured achieved RPY onto the SAME 2*pi branch as the
            # desired goal. PyKDL GetRPY returns measured roll/yaw in [-pi, pi],
            # which can sit a full branch away from the command-side desired_goal
            # (e.g. measured roll +2.0 vs desired roll -3.97). Wrapping only the
            # delta leaves the absolute achieved-RPY block out of distribution;
            # aligning the achieved RPY fixes the current block AND the delta so
            # both match the command-integrated training observations.
            current = current.copy()
            current[3:6] = current[3:6] + 2 * np.pi * np.round(
                (goal_obs[3:6] - current[3:6]) / (2 * np.pi)
            )
            delta = goal_obs - current
        elif getattr(self.env, "wrap_obs_rpy_delta", False):
            # Lighter variant: wrap only the RPY delta block (indices 3:6) onto
            # [-pi, pi]; leaves the absolute achieved-RPY block untouched.
            delta = goal_obs - current
            delta = delta.copy()
            delta[3:6] = (delta[3:6] + np.pi) % (2 * np.pi) - np.pi
        else:
            delta = goal_obs - current

        obs_array = np.concatenate((current, goal_obs, delta), dtype=np.float32)
        obs_dict = {
            "observation": obs_array,
            "achieved_goal": current,
            "desired_goal": goal_obs
        }
        
        self.env.obs = self.normalize_observation(obs_dict)
        reward_achieved = self.env.reward_achieved_goal(self.env.obs["achieved_goal"])
        self.reward = self.env.compute_reward(reward_achieved, self.env.obs["desired_goal"])
        self.terminate = self.env.criteria()
        self.truncate = self.env.timestep >= self.env.max_timestep
        self.env.info = {"is_success": self.terminate}

        return self.env.obs, self.reward, self.terminate, self.truncate, self.env.info

    def normalize_observation(self, observation_dict):
        """Scale observations - translation into 'cm', orientation into 'rad'."""
        observation = np.asarray(observation_dict["observation"], dtype=np.float32)
        achieved_goal = np.asarray(observation_dict["achieved_goal"], dtype=np.float32)
        desired_goal = np.asarray(observation_dict["desired_goal"], dtype=np.float32)

        multiplier = np.ones(21, dtype=np.float32)
        indices_to_multiply_100 = [0, 1, 2, 7, 8, 9, 14, 15, 16]
        multiplier[indices_to_multiply_100] = 100

        multiplier2 = np.array([100, 100, 100, 1, 1, 1, 1], dtype=np.float32)
        return {
            "observation": np.asarray(observation * multiplier, dtype=np.float32),
            "achieved_goal": np.asarray(achieved_goal * multiplier2, dtype=np.float32),
            "desired_goal": np.asarray(desired_goal * multiplier2, dtype=np.float32),
        }
