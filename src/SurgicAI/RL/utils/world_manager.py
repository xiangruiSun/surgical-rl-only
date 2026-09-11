import time
from threading import Lock, Thread
from std_msgs.msg import Empty
from ambf_msgs.msg import WorldCmd, WorldState

class WorldManager:
    """Manages world state and reset commands"""
    
    def __init__(self, ral_instance, synchronous=False, steps_per_action=10, barrier_timeout_s=2.0):
        """
        Initialize WorldManager with RAL instance.
        
        :param ral_instance: RAL instance for ROS communication
        """
        self.ral = ral_instance
        self._world_cmd = WorldCmd()
        self.synchronous = bool(synchronous)
        self.steps_per_action = int(steps_per_action)
        self.barrier_timeout_s = float(barrier_timeout_s)
        if self.steps_per_action <= 0:
            raise ValueError("steps_per_action must be positive")
        self._state_lock = Lock()
        self._cmd_lock = Lock()
        self._sim_step = None
        self._state_sequence = 0
        self.last_step_barrier = None
        self._reset_pub = self.ral.publisher(
            '/ambf/env/World/Command/Reset',
            Empty,
            queue_size=1
        )
        self._world_cmd_pub = self.ral.publisher(
            '/ambf/env/World/Command',
            WorldCmd,
            queue_size=1
        )
        self._world_state_sub = self.ral.subscriber(
            '/ambf/env/World/State',
            WorldState,
            self._world_state_callback,
            queue_size=1,
        )
        self._world_cmd.enable_step_throttling = self.synchronous
        self._world_cmd.n_skip_steps = self.steps_per_action
        time.sleep(0.5)  # Allow some time for publishers to initialize
        if self.synchronous:
            self._publish_world_cmd()
            self._wait_for_world_state()
            # Allow the simulator to consume the throttle command and stop at
            # the first deterministic boundary.
            time.sleep(0.2)
            Thread(target=self._sync_keepalive_loop, daemon=True).start()

    def _publish_world_cmd(self):
        with self._cmd_lock:
            self._world_cmd_pub.publish(self._world_cmd)

    def _sync_keepalive_loop(self):
        # WorldRosCom's watchdog otherwise calls reset_cmd(), silently disables
        # throttling, and lets physics free-run. Republishing the same clock bit
        # acknowledges the watchdog without releasing another step batch.
        while True:
            self._publish_world_cmd()
            # Stay comfortably inside the AMBF command watchdog window. The
            # previous 100 ms cadence occasionally expired at the boundary and
            # released 2-3 uncommanded physics steps.
            time.sleep(0.02)

    def _world_state_callback(self, msg):
        with self._state_lock:
            self._sim_step = int(msg.sim_step)
            self._state_sequence += 1

    def _get_sim_step(self):
        with self._state_lock:
            return self._sim_step

    def _get_world_state_sample(self):
        with self._state_lock:
            return self._sim_step, self._state_sequence

    def _wait_for_stopped_step(self):
        """Wait for two published WorldState samples at the same sim step.

        A single WorldState sample can arrive while an AMBF skip batch is still
        running. Releasing the next batch from that stale sample can overlap the
        current batch and produce a short advance (for example 4 instead of 10
        steps). Two equal, successive samples prove that the throttle has stopped.
        """
        deadline = time.monotonic() + self.barrier_timeout_s
        previous_step = None
        previous_sequence = -1
        while time.monotonic() < deadline:
            step, sequence = self._get_world_state_sample()
            if step is not None and sequence != previous_sequence:
                if previous_step == step:
                    return step
                previous_step = step
                previous_sequence = sequence
            time.sleep(0.001)
        raise RuntimeError(
            "Timed out waiting for two stable /ambf/env/World/State samples: "
            f"step={previous_step}, sequence={previous_sequence}"
        )

    def _wait_for_world_state(self):
        deadline = time.monotonic() + self.barrier_timeout_s
        while time.monotonic() < deadline:
            step = self._get_sim_step()
            if step is not None:
                return step
            time.sleep(0.001)
        raise RuntimeError("Timed out waiting for /ambf/env/World/State")
    
    def reset(self):
        """Reset the world state"""
        reset_msg = Empty()
        self._reset_pub.publish(reset_msg)
        if self.synchronous:
            self.update()
        else:
            time.sleep(0.5)
        
    def update(self, requested_steps=None):
        requested_steps = (
            self.steps_per_action if requested_steps is None else int(requested_steps)
        )
        if requested_steps <= 0:
            raise ValueError("requested_steps must be positive")
        before = self._wait_for_stopped_step() if self.synchronous else self._get_sim_step()
        with self._cmd_lock:
            self._world_cmd.n_skip_steps = requested_steps
            self._world_cmd.step_clock = not self._world_cmd.step_clock
            self._world_cmd_pub.publish(self._world_cmd)
        if not self.synchronous:
            self.last_step_barrier = None
            return
        target = before + requested_steps
        start = time.monotonic()
        deadline = start + self.barrier_timeout_s
        after = before
        while time.monotonic() < deadline:
            current = self._get_sim_step()
            if current is not None:
                after = current
                if current >= target:
                    self.last_step_barrier = {
                        "before": int(before),
                        "target": int(target),
                        "after": int(current),
                        "requested_steps": int(requested_steps),
                        "observed_steps": int(current - before),
                        "wait_s": float(time.monotonic() - start),
                        "timed_out": False,
                    }
                    return
            time.sleep(0.001)

        # Switching AMBF's n_skip_steps between reset pulses (1) and policy
        # pulses (10) can expose a non-zero internal skip counter. In that case
        # AMBF stops early (for example after 6/10 steps). Confirm that it has
        # stopped, then release exactly the missing remainder; never free-run
        # or silently accept a short batch.
        recovery = None
        try:
            stable_after = self._wait_for_stopped_step()
        except RuntimeError:
            stable_after = after
        if stable_after >= target:
            self.last_step_barrier = {
                "before": int(before),
                "target": int(target),
                "after": int(stable_after),
                "requested_steps": int(requested_steps),
                "observed_steps": int(stable_after - before),
                "wait_s": float(time.monotonic() - start),
                "timed_out": False,
                "recovery": {"late_world_state": True},
            }
            if stable_after != target:
                raise RuntimeError(
                    "AMBF late WorldState overshot target: "
                    f"{self.last_step_barrier}"
                )
            return
        if before < stable_after < target:
            remaining = target - stable_after
            recovery = {
                "partial_after": int(stable_after),
                "remaining_steps": int(remaining),
            }
            with self._cmd_lock:
                self._world_cmd.n_skip_steps = int(remaining)
                self._world_cmd.step_clock = not self._world_cmd.step_clock
                self._world_cmd_pub.publish(self._world_cmd)
            recovery_deadline = time.monotonic() + self.barrier_timeout_s
            while time.monotonic() < recovery_deadline:
                current = self._get_sim_step()
                if current is not None:
                    after = current
                    if current >= target:
                        self.last_step_barrier = {
                            "before": int(before),
                            "target": int(target),
                            "after": int(current),
                            "requested_steps": int(requested_steps),
                            "observed_steps": int(current - before),
                            "wait_s": float(time.monotonic() - start),
                            "timed_out": False,
                            "recovery": recovery,
                        }
                        if current != target:
                            raise RuntimeError(
                                "AMBF remainder recovery overshot target: "
                                f"{self.last_step_barrier}"
                            )
                        return
                time.sleep(0.001)
        self.last_step_barrier = {
            "before": int(before),
            "target": int(target),
            "after": int(after),
            "requested_steps": int(requested_steps),
            "observed_steps": int(after - before),
            "wait_s": float(time.monotonic() - start),
            "timed_out": True,
            "recovery": recovery,
        }
        raise RuntimeError(f"AMBF physics-step barrier timed out: {self.last_step_barrier}")
