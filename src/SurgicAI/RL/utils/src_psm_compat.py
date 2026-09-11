"""SurgicAI API compatibility layer over SRC PSM (AMBF client I/O)."""

import logging
import time
from threading import Lock, RLock, Thread

from ambf_msgs.msg import RigidBodyCmd
from surgical_robotics_challenge.psm_arm import PSM as _SrcPSM

logger = logging.getLogger(__name__)

# AMBF CmdWatchDog (RigidBodyRosCom) resets when ROS /Command messages arrive.
# Pair with ADF `time out: 86400.0` on BODY base link and/or --override_min_comm_freq 10.
DEFAULT_COMMAND_RATE_HZ = 150
KEEPALIVE_LOG_INTERVAL_S = 10.0
MIN_KEEPALIVE_HZ_WARN = 90.0

# Shared across psm1/psm2 keepalive threads (same ral node in ROS2).
_ral_publish_lock = RLock()


def _resolve_command_topic(psm_arm, name: str) -> str:
    """Use simulator-advertised name; fall back to conventional baselink path."""
    try:
        ros_name = psm_arm.base.get_ros_name().rstrip("/")
        if ros_name:
            return f"{ros_name}/Command"
    except Exception as exc:
        logger.warning("PSM %s get_ros_name failed: %s", name, exc)
    return f"/ambf/env/{name}/baselink/Command"


def _pick_ral(simulation_manager, ral_instance):
    """Prefer the AMBF client's ral (same node that talks to the simulator)."""
    client_ral = None
    try:
        client_ral = simulation_manager.get_ral()
    except Exception:
        pass
    if client_ral is not None:
        return client_ral, "simulation_manager.get_ral()"
    if ral_instance is not None:
        return ral_instance, "env ral_instance"
    return None, "none"


def _busy_wait_until(deadline: float) -> None:
    """Spin until perf_counter reaches deadline (WSL-safe vs time.sleep)."""
    while time.perf_counter() < deadline:
        pass


def _spin_ral_once(ral) -> bool:
    for method_name in ("spin_once", "spin_some"):
        method = getattr(ral, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except TypeError:
            try:
                method(0)
            except TypeError:
                method(0.0)
        return True
    return False


class PSM:
    """Wraps surgical_robotics_challenge.PSM for SceneManager call sites."""

    def __init__(
        self,
        simulation_manager,
        name,
        add_joint_errors=False,
        ral_instance=None,
        command_rate_hz=DEFAULT_COMMAND_RATE_HZ,
        debug_keepalive=False,
        **kwargs,
    ):
        self.name = name
        self.simulation_manager = simulation_manager
        self._command_rate_hz = max(int(command_rate_hz), DEFAULT_COMMAND_RATE_HZ)
        self._debug_keepalive = debug_keepalive
        self._cmd_lock = Lock()
        self._last_jp = [0.0] * 6
        self._last_jaw = 0.5
        self._cmd_pub = None
        self._cmd_topic = None
        self._ral = None
        self._ral_source = "none"
        self._cmd_thread = None
        self._keepalive_stats = {
            "publish_count": 0,
            "skip_none_jp_count": 0,
            "publish_error_count": 0,
            "spin_ok_count": 0,
            "started": False,
            "last_hz": 0.0,
        }

        self._arm = _SrcPSM(
            simulation_manager,
            name,
            add_joint_errors=add_joint_errors,
            **kwargs,
        )
        self._refresh_joint_cache_from_arm()

        self._ral, self._ral_source = _pick_ral(simulation_manager, ral_instance)
        if self._ral is not None:
            self._cmd_topic = _resolve_command_topic(self._arm, name)
            self._cmd_pub = self._ral.publisher(
                self._cmd_topic, RigidBodyCmd, queue_size=1
            )
            if self._cmd_pub is None:
                logger.error(
                    "PSM %s: ral.publisher returned None for %s",
                    name,
                    self._cmd_topic,
                )
            else:
                # Publish once before starting thread so watchdog is fed immediately.
                self.publish_keepalive_once()
                self._cmd_thread = Thread(
                    target=self._ros_command_keepalive,
                    name=f"psm_cmd_keepalive_{name}",
                    daemon=True,
                )
                self._cmd_thread.start()
                logger.info(
                    "PSM %s keepalive: topic=%s rate=%s Hz ral=%s jp_cache=%s",
                    name,
                    self._cmd_topic,
                    self._command_rate_hz,
                    self._ral_source,
                    self._last_jp,
                )
        else:
            logger.warning(
                "PSM %s: no ral for ROS Command keepalive; watchdog may expire",
                name,
            )

    def _refresh_joint_cache_from_arm(self):
        try:
            jp = self._arm.measured_jp()
            if jp is not None and len(jp) >= 6:
                self._update_cmd_cache(jp, self._last_jaw)
                return
        except Exception as exc:
            logger.warning("PSM %s measured_jp failed, using zeros: %s", self.name, exc)
        self._update_cmd_cache([0.0] * 6, self._last_jaw)

    def _update_cmd_cache(self, jp, jaw_angle):
        with self._cmd_lock:
            self._last_jp = list(jp[:6])
            self._last_jaw = float(jaw_angle)

    def _build_rigid_body_cmd(self, jp, jaw_angle):
        msg = RigidBodyCmd()
        msg.joint_cmds = [0.0] * 8
        msg.joint_cmds_types = [RigidBodyCmd.TYPE_POSITION] * 8
        for i in range(6):
            msg.joint_cmds[i] = float(jp[i])
        msg.joint_cmds[6] = float(jaw_angle)
        msg.joint_cmds[7] = float(jaw_angle)
        return msg

    def _publish_ros_command(self, jp, jaw):
        if self._cmd_pub is None:
            return False
        try:
            with _ral_publish_lock:
                self._cmd_pub.publish(self._build_rigid_body_cmd(jp, jaw))
                if self._ral is not None and _spin_ral_once(self._ral):
                    self._keepalive_stats["spin_ok_count"] += 1
            return True
        except Exception as exc:
            self._keepalive_stats["publish_error_count"] += 1
            logger.error("PSM %s publish failed on %s: %s", self.name, self._cmd_topic, exc)
            return False

    def _ros_command_keepalive(self):
        period = 1.0 / float(self._command_rate_hz)
        self._keepalive_stats["started"] = True
        window_start = time.perf_counter()
        window_publishes = 0
        next_tick = window_start + period
        logger.info(
            "PSM %s keepalive thread running (busy-wait, target %s Hz)",
            self.name,
            self._command_rate_hz,
        )

        while not self.simulation_manager.is_shutdown():
            try:
                with self._cmd_lock:
                    jp = None if self._last_jp is None else list(self._last_jp)
                    jaw = self._last_jaw

                if jp is None:
                    self._keepalive_stats["skip_none_jp_count"] += 1
                elif self._publish_ros_command(jp, jaw):
                    self._keepalive_stats["publish_count"] += 1
                    window_publishes += 1

                next_tick += period
                now = time.perf_counter()
                if now < next_tick:
                    _busy_wait_until(next_tick)
                else:
                    next_tick = now + period

                now = time.perf_counter()
                if now - window_start >= KEEPALIVE_LOG_INTERVAL_S:
                    dt = now - window_start
                    hz = window_publishes / dt if dt > 0 else 0.0
                    self._keepalive_stats["last_hz"] = hz
                    if self._debug_keepalive:
                        logger.info(
                            "PSM %s keepalive: %.1f Hz (target %s), total_pub=%s, "
                            "skip_none=%s, errors=%s, spin_ok=%s, topic=%s",
                            self.name,
                            hz,
                            self._command_rate_hz,
                            self._keepalive_stats["publish_count"],
                            self._keepalive_stats["skip_none_jp_count"],
                            self._keepalive_stats["publish_error_count"],
                            self._keepalive_stats["spin_ok_count"],
                            self._cmd_topic,
                        )
                    elif hz < MIN_KEEPALIVE_HZ_WARN:
                        logger.warning(
                            "PSM %s keepalive below %.0f Hz (%.1f)",
                            self.name,
                            MIN_KEEPALIVE_HZ_WARN,
                            hz,
                        )
                    window_start = now
                    window_publishes = 0
                    next_tick = now + period
            except Exception as exc:
                self._keepalive_stats["publish_error_count"] += 1
                logger.exception("PSM %s keepalive loop error: %s", self.name, exc)

    def publish_keepalive_once(self):
        """Publish one ROS Command (for watchdog test / env step backup)."""
        with self._cmd_lock:
            jp = None if self._last_jp is None else list(self._last_jp)
            jaw = self._last_jaw
        if jp is None:
            return False
        ok = self._publish_ros_command(jp, jaw)
        if ok:
            self._keepalive_stats["publish_count"] += 1
        return ok

    def test_command_publish(self, burst_count=20, burst_rate_hz=200):
        """
        Burst-publish Commands and log rate. Call once after init while AMBF is running.
        If AMBF watchdog messages stop during this burst, ROS publishing works.
        """
        if self._cmd_pub is None:
            logger.error(
                "PSM %s test_command_publish: no publisher (ral=%s)",
                self.name,
                self._ral_source,
            )
            return False

        period = 1.0 / float(max(burst_rate_hz, DEFAULT_COMMAND_RATE_HZ))
        t0 = time.perf_counter()
        sent = 0
        next_tick = t0 + period
        for _ in range(burst_count):
            if self.publish_keepalive_once():
                sent += 1
            next_tick += period
            now = time.perf_counter()
            if now < next_tick:
                _busy_wait_until(next_tick)
            else:
                next_tick = now + period
        dt = time.perf_counter() - t0
        hz = sent / dt if dt > 0 else 0.0
        logger.info(
            "PSM %s watchdog test: sent %s/%s msgs on %s in %.3fs (%.1f Hz). "
            "Check AMBF console for 'Watch Dog Expired' during burst.",
            self.name,
            sent,
            burst_count,
            self._cmd_topic,
            dt,
            hz,
        )
        return sent == burst_count

    def get_keepalive_status(self):
        return {
            "name": self.name,
            "topic": self._cmd_topic,
            "ral_source": self._ral_source,
            "rate_hz_target": self._command_rate_hz,
            "thread_started": self._keepalive_stats["started"],
            "has_publisher": self._cmd_pub is not None,
            "last_jp_ready": self._last_jp is not None,
            **self._keepalive_stats,
        }

    def actuate(self, name):
        if self._arm.actuators:
            self._arm.actuators[0].actuate(name)

    def deactuate(self):
        if self._arm.actuators:
            self._arm.actuators[0].deactuate()

    def get_jaw_angle(self):
        return self._arm.base.get_joint_pos(6)

    def move_base(self, pos_vector):
        self._arm.base.set_pos(pos_vector)

    def servo_jp(self, jp):
        self._update_cmd_cache(jp, self._last_jaw)
        self.publish_keepalive_once()
        return self._arm.servo_jp(jp)

    def set_jaw_angle(self, jaw_angle):
        jaw_angle = float(jaw_angle)
        if self._last_jp is not None:
            self._update_cmd_cache(self._last_jp, jaw_angle)
        else:
            with self._cmd_lock:
                self._last_jaw = jaw_angle
        self.publish_keepalive_once()
        return self._arm.set_jaw_angle(jaw_angle)

    def servo_cp(self, T_t_b):
        result = self._arm.servo_cp(T_t_b)
        self._update_cmd_cache(self._arm._ik_solution, self._last_jaw)
        self.publish_keepalive_once()
        return result

    def move_cp(self, T_t_b, execute_time=0.5, control_rate=120):
        return self._arm.move_cp(T_t_b, execute_time, control_rate)

    def __getattr__(self, item):
        return getattr(self._arm, item)
