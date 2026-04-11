#!/usr/bin/env python3
"""
Monitoring Dashboard Node
=========================
Subscribes to all monitoring topics and renders a clean, color-coded
terminal dashboard. Clears and redraws the screen every refresh cycle.

Topics subscribed:
  /system_status     (std_msgs/String)
  /system_alerts     (std_msgs/String)
  /waypoint_reached  (std_msgs/String)
  /waypoint_status   (std_msgs/String)
"""

import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── ANSI color codes ──────────────────────────────────────────────────────────
R  = '\033[0m'        # reset
B  = '\033[1m'        # bold
DIM = '\033[2m'       # dim

BLACK  = '\033[30m'
RED    = '\033[31m'
GREEN  = '\033[32m'
YELLOW = '\033[33m'
BLUE   = '\033[34m'
CYAN   = '\033[36m'
WHITE  = '\033[37m'

BG_RED    = '\033[41m'
BG_GREEN  = '\033[42m'
BG_YELLOW = '\033[43m'
BG_BLUE   = '\033[44m'
BG_CYAN   = '\033[46m'

# ── Box-drawing chars ─────────────────────────────────────────────────────────
TL, TR, BL, BR = '╔', '╗', '╚', '╝'
H, V           = '═', '║'
LM, RM         = '╠', '╣'
TT, BT         = '╦', '╩'

WIDTH = 72


def box_top(title='', width=WIDTH):
    if title:
        pad = width - len(title) - 4
        left = pad // 2
        right = pad - left
        return f'{TL}{H * left} {B}{title}{R} {H * right}{TR}'
    return f'{TL}{H * (width - 2)}{TR}'


def box_mid(width=WIDTH):
    return f'{LM}{H * (width - 2)}{RM}'


def box_bot(width=WIDTH):
    return f'{BL}{H * (width - 2)}{BR}'


def box_row(text='', width=WIDTH):
    # Strip ANSI codes for length calculation
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    visible = ansi_escape.sub('', text)
    pad = width - 2 - len(visible)
    return f'{V} {text}{" " * max(pad, 0)}{V}'


def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime('%H:%M:%S') if ts else '—'


def status_badge(ok: bool):
    if ok:
        return f'{B}{BG_GREEN}{BLACK}  OK  {R}'
    return f'{B}{BG_RED}{WHITE}  !!  {R}'


def alert_badge(level: str):
    if level == 'ERROR':
        return f'{B}{BG_RED}{WHITE} ERROR {R}'
    if level == 'WARN':
        return f'{B}{BG_YELLOW}{BLACK}  WARN {R}'
    return f'{B}{BG_BLUE}{WHITE}  INFO {R}'


def wp_badge(reached: bool):
    if reached:
        return f'{B}{GREEN}✔ REACHED{R}'
    return f'{DIM}{YELLOW}◌ pending {R}'


class DashboardNode(Node):

    def __init__(self):
        super().__init__('monitoring_dashboard')

        self._system_status  = None
        self._system_alerts  = None
        self._waypoint_status = None
        self._recent_events: list[dict] = []   # last 5 waypoint arrivals

        self.create_subscription(String, '/system_status',    self._on_status,   10)
        self.create_subscription(String, '/system_alerts',    self._on_alerts,   10)
        self.create_subscription(String, '/waypoint_reached', self._on_wp_event, 10)
        self.create_subscription(String, '/waypoint_status',  self._on_wp_status, 10)

        # Redraw at 1 Hz
        self.create_timer(1.0, self._draw)

    # ── Subscribers ───────────────────────────────────────────────────────────

    def _on_status(self, msg):
        try:
            self._system_status = json.loads(msg.data)
        except Exception:
            pass

    def _on_alerts(self, msg):
        try:
            self._system_alerts = json.loads(msg.data)
        except Exception:
            pass

    def _on_wp_event(self, msg):
        try:
            data = json.loads(msg.data)
            self._recent_events.append(data)
            self._recent_events = self._recent_events[-5:]   # keep last 5
        except Exception:
            pass

    def _on_wp_status(self, msg):
        try:
            self._waypoint_status = json.loads(msg.data)
        except Exception:
            pass

    # ── Renderer ──────────────────────────────────────────────────────────────

    def _draw(self):
        lines = []
        now_str = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

        # ── Header ────────────────────────────────────────────────────────────
        lines.append(box_top(f'MERCURY ROBOT  —  MONITORING DASHBOARD'))
        lines.append(box_row(f'{DIM}Updated: {now_str}{R}'))
        lines.append(box_mid())

        # ── Section 1: System Health ──────────────────────────────────────────
        lines.append(box_row(f'{B}{CYAN}  SYSTEM HEALTH{R}'))
        lines.append(box_row())

        if self._system_status is None:
            lines.append(box_row(f'  {YELLOW}Waiting for /system_status ...{R}'))
        else:
            s = self._system_status
            ok = s.get('all_ok', False)
            running = s.get('total_running', 0)
            expected = s.get('total_expected', 0)
            missing  = s.get('missing', [])
            unexpected = s.get('unexpected', [])

            badge = status_badge(ok)
            lines.append(box_row(
                f'  {badge}  Nodes running: {B}{GREEN}{running}{R}   '
                f'Expected: {B}{expected}{R}'
            ))

            if missing:
                lines.append(box_row())
                lines.append(box_row(f'  {RED}{B}Missing nodes:{R}'))
                for n in missing:
                    lines.append(box_row(f'    {RED}✖ {n}{R}'))
            else:
                lines.append(box_row(f'  {GREEN}✔ All expected nodes are running{R}'))

            if unexpected:
                lines.append(box_row(f'  {DIM}Extra nodes: {", ".join(unexpected[:3])}'
                                     + (f' (+{len(unexpected)-3} more)' if len(unexpected) > 3 else '') + R))

            # Launch order (last 5 detected)
            order = s.get('launch_order', [])
            if order:
                lines.append(box_row())
                lines.append(box_row(f'  {DIM}Launch order (first detected):{R}'))
                for entry in order[-5:]:
                    lines.append(box_row(
                        f'    {DIM}+{entry["detected_at_s"]:5.1f}s  {entry["node"]}{R}'
                    ))

        lines.append(box_mid())

        # ── Section 2: Watchdog Alerts ────────────────────────────────────────
        lines.append(box_row(f'{B}{CYAN}  WATCHDOG ALERTS{R}'))
        lines.append(box_row())

        if self._system_alerts is None:
            lines.append(box_row(f'  {YELLOW}Waiting for /system_alerts ...{R}'))
        else:
            a = self._system_alerts
            alerts = a.get('alerts', [])
            if not alerts:
                lines.append(box_row(f'  {GREEN}✔ No active alerts — all systems nominal{R}'))
            else:
                for alert in alerts:
                    badge = alert_badge(alert.get('level', 'WARN'))
                    cat   = alert.get('category', '').replace('_', ' ').upper()
                    subj  = alert.get('subject', '')
                    msg   = alert.get('message', '')
                    fix   = alert.get('suggested_fix', '')
                    lines.append(box_row(f'  {badge} {B}{cat}{R}  {subj}'))
                    # Word-wrap the message to fit box width
                    max_w = WIDTH - 6
                    words, cur = msg.split(), ''
                    for word in words:
                        if len(cur) + len(word) + 1 > max_w:
                            lines.append(box_row(f'       {DIM}{cur}{R}'))
                            cur = word
                        else:
                            cur = (cur + ' ' + word).strip()
                    if cur:
                        lines.append(box_row(f'       {DIM}{cur}{R}'))
                    if fix:
                        lines.append(box_row(f'       {YELLOW}↳ Fix: {fix}{R}'))
                    lines.append(box_row())

        lines.append(box_mid())

        # ── Section 3: Waypoints ──────────────────────────────────────────────
        lines.append(box_row(f'{B}{CYAN}  WAYPOINT STATUS{R}'))
        lines.append(box_row())

        if self._waypoint_status is None:
            lines.append(box_row(f'  {YELLOW}Waiting for /waypoint_status ...{R}'))
        else:
            ws = self._waypoint_status
            total    = ws.get('total', 0)
            reached  = ws.get('reached_at_least_once', 0)
            all_done = ws.get('all_completed', False)
            rx = ws.get('robot_x', 0.0)
            ry = ws.get('robot_y', 0.0)

            # Progress bar
            filled = int((reached / total * 20)) if total else 0
            bar = f'{GREEN}{"█" * filled}{DIM}{"░" * (20 - filled)}{R}'
            pct = int(reached / total * 100) if total else 0
            lines.append(box_row(
                f'  Robot: ({rx:6.2f}, {ry:6.2f})   '
                f'Progress: [{bar}] {pct}%  ({reached}/{total})'
            ))
            lines.append(box_row())

            for wp in ws.get('waypoints', []):
                badge = wp_badge(wp.get('reach_count', 0) > 0)
                name  = wp.get('name', '?')
                x, y  = wp.get('x', 0.0), wp.get('y', 0.0)
                count = wp.get('reach_count', 0)
                ts    = wp.get('reached_at')
                count_str = f'{DIM}(×{count}){R}' if count > 0 else ''
                time_str  = f'{DIM}@ {fmt_time(ts)}{R}' if ts else ''
                lines.append(box_row(
                    f'    {badge}  {B}{name}{R}  '
                    f'({x:.2f}, {y:.2f})  {count_str} {time_str}'
                ))

            if all_done:
                lines.append(box_row())
                lines.append(box_row(
                    f'  {B}{BG_GREEN}{BLACK}  ✔ ALL WAYPOINTS COMPLETED — MISSION OBJECTIVE MET!  {R}'
                ))

        # ── Section 4: Recent Waypoint Events ────────────────────────────────
        if self._recent_events:
            lines.append(box_mid())
            lines.append(box_row(f'{B}{CYAN}  RECENT WAYPOINT EVENTS{R}'))
            lines.append(box_row())
            for ev in reversed(self._recent_events):
                wp   = ev.get('waypoint', {})
                name = wp.get('name', '?')
                dist = ev.get('distance', 0.0)
                rx2  = ev.get('robot_x', 0.0)
                ry2  = ev.get('robot_y', 0.0)
                ts   = ev.get('timestamp')
                lines.append(box_row(
                    f'  {GREEN}▶{R} {B}{name}{R}  '
                    f'robot=({rx2:.2f}, {ry2:.2f})  '
                    f'dist={dist:.3f}m  '
                    f'{DIM}{fmt_time(ts)}{R}'
                ))

        # ── Footer ────────────────────────────────────────────────────────────
        lines.append(box_bot())

        # ── Render ────────────────────────────────────────────────────────────
        os.system('clear')
        print('\n'.join(lines))
        print()


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
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
