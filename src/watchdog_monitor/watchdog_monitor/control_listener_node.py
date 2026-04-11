#!/usr/bin/env python3
"""
Control Listener Node
=====================
A purely passive observer node that integrates with the control package
context WITHOUT modifying any existing control logic.

It subscribes to all monitoring topics and logs structured summaries.
This is the "bridge" between the monitoring subsystem and the control layer.

Topics subscribed (read-only, no publishing to control topics):
  /system_status     (std_msgs/String)  — from system_monitor
  /system_alerts     (std_msgs/String)  — from watchdog
  /waypoint_reached  (std_msgs/String)  — from waypoint_manager
  /waypoint_status   (std_msgs/String)  — from waypoint_manager

No control topics are published or subscribed — this node is purely
observational. To trigger actual control actions, extend this class and
override the handler methods.
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ControlListenerNode(Node):
    """
    Observes system health and waypoint topics from a control-package
    perspective. Safe to run alongside the existing control package.
    """

    def __init__(self) -> None:
        super().__init__('control_listener_node')

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            String, '/system_status', self._on_system_status, 10
        )
        self.create_subscription(
            String, '/system_alerts', self._on_system_alerts, 10
        )
        self.create_subscription(
            String, '/waypoint_reached', self._on_waypoint_reached, 10
        )
        self.create_subscription(
            String, '/waypoint_status', self._on_waypoint_status, 10
        )

        # ── Internal state (read-only mirror) ────────────────────────────────
        self._last_system_ok: bool | None = None
        self._last_alert_count: int = 0
        self._waypoints_reached: list[str] = []

        # ── Throttle — only log system_status/waypoint_status periodically ───
        self._last_status_log: float = 0.0
        self._last_wp_status_log: float = 0.0
        self._status_log_interval: float = 5.0   # seconds

        self.get_logger().info('[ControlListener] Node started — observing monitoring topics')

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_system_status(self, msg: String) -> None:
        """Receive periodic system health summary."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('[ControlListener] Malformed /system_status message')
            return

        all_ok: bool = data.get('all_ok', False)
        missing: list[str] = data.get('missing', [])
        total_running: int = data.get('total_running', 0)
        total_expected: int = data.get('total_expected', 0)

        # Log a concise summary at most every N seconds to avoid log spam
        now = time.time()
        if now - self._last_status_log >= self._status_log_interval:
            self._last_status_log = now
            if all_ok:
                self.get_logger().info(
                    f'[ControlListener] System OK — '
                    f'{total_running}/{total_expected} nodes running'
                )
            else:
                self.get_logger().warn(
                    f'[ControlListener] System DEGRADED — '
                    f'missing nodes: {missing}'
                )

        # State-change log (always immediate)
        if self._last_system_ok is not None and all_ok != self._last_system_ok:
            if all_ok:
                self.get_logger().info(
                    '[ControlListener] ✓ System recovered — all expected nodes running'
                )
            else:
                self.get_logger().error(
                    f'[ControlListener] ✗ System degraded — missing: {missing}'
                )
        self._last_system_ok = all_ok

    def _on_system_alerts(self, msg: String) -> None:
        """Receive watchdog alerts — log them for the control layer."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('[ControlListener] Malformed /system_alerts message')
            return

        alerts: list[dict] = data.get('alerts', [])
        alert_count: int = data.get('alert_count', 0)

        if alert_count == 0:
            return  # silence is golden

        # Only log when alert count changes to avoid flooding
        if alert_count != self._last_alert_count:
            self._last_alert_count = alert_count
            self.get_logger().warn(
                f'[ControlListener] {alert_count} watchdog alert(s) active:'
            )

        for alert in alerts:
            level = alert.get('level', 'WARN')
            category = alert.get('category', 'unknown')
            subject = alert.get('subject', '?')
            message = alert.get('message', '')
            fix = alert.get('suggested_fix', '')

            log_fn = self.get_logger().error if level == 'ERROR' else self.get_logger().warn
            log_fn(
                f'[ControlListener]  [{level}] {category.upper()} — {subject}: {message}'
            )
            if fix:
                self.get_logger().info(f'[ControlListener]    ↳ Fix: {fix}')

    def _on_waypoint_reached(self, msg: String) -> None:
        """Handle waypoint arrival event — always log immediately."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('[ControlListener] Malformed /waypoint_reached message')
            return

        wp = data.get('waypoint', {})
        wp_name: str = wp.get('name', '?')
        reach_count: int = wp.get('reach_count', 1)
        dist: float = data.get('distance', 0.0)
        robot_x: float = data.get('robot_x', 0.0)
        robot_y: float = data.get('robot_y', 0.0)

        self.get_logger().info(
            f'[ControlListener] *** WAYPOINT EVENT: {wp_name} reached '
            f'(visit #{reach_count}) — '
            f'robot=({robot_x:.2f}, {robot_y:.2f}), dist={dist:.3f}m ***'
        )

        if wp_name not in self._waypoints_reached:
            self._waypoints_reached.append(wp_name)
            self.get_logger().info(
                f'[ControlListener] Waypoints reached so far: '
                f'{self._waypoints_reached}'
            )

    def _on_waypoint_status(self, msg: String) -> None:
        """Receive periodic waypoint overview — log only when all complete."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        all_completed: bool = data.get('all_completed', False)
        reached: int = data.get('reached_at_least_once', 0)
        total: int = data.get('total', 0)

        now = time.time()
        if now - self._last_wp_status_log >= self._status_log_interval:
            self._last_wp_status_log = now
            self.get_logger().info(
                f'[ControlListener] Waypoint progress: {reached}/{total}'
            )

        if all_completed:
            self.get_logger().info(
                '[ControlListener] ✓✓✓ All waypoints completed — mission objective met!'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlListenerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
