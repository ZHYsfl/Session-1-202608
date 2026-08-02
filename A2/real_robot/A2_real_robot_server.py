#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 真机 WebSocket Server（替换 001 Webots 仿真）

协议：兼容 A2 api.md v1.1，额外扩展 human/human_confirm 用于真机人工介入。

人工介入流程（默认 manual-drive）：
  1. client 发 reset
  2. server 回 human(action="record_goal")，线下把车摆到目标点
  3. client 发 human_confirm(action="record_goal")，server 记录当前 /odom 为目标点
  4. server 回 human(action="drive_to_start")，用户把车推到/开到起点（可任意朝向）
  5. client 发 human_confirm(action="drive_to_start")，server 记录当前 /odom 为起点
  6. server 发初始 obs，进入 RUNNING
  7. client 发 action → server 执行 0.1s → 回 obs

运行：
    source /opt/ros/jazzy/setup.bash
    source ~/ros2_ws/install/setup.bash
    python3 ~/A2_real_robot_server.py

参数：
    --port 8765
    --lidar-front-offset-deg 0.0
    --log-dir ~/a2_real_robot_logs
    --goal-mode relative|manual-drive
    --goal-relative-x 2.0
    --goal-relative-y 0.0

说明：
    默认 --goal-mode=manual-drive。真机 /odom 只能跟踪车轮移动，不能跟踪
    人手搬车。manual-drive 模式下先摆目标点，再推车/开车到起点，odom 会
    记录真实位移；relative 模式只用于测试，目标点由 start + 相对偏移计算。
"""

import argparse
import asyncio
import csv
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("a2_real_robot")


def normalize_angle(angle: float) -> float:
    """把角度归一化到 (-π, π]"""
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """四元数转 yaw（绕 Z 轴）"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class RealRobotServer(Node):
    PROTOCOL_VERSION = "1.1"
    ENV_NAME = "real_diffbot_v1"

    # A2 协议默认常量（hello.config 会逐项下发）
    DEFAULT_CFG = {
        "lidar_count": 64,
        "lidar_max_range": 3.5,
        "obs_dim": 68,
        "act_dim": 2,
        "control_dt": 0.1,
        "max_episode_time": 60.0,
        "v_max": 0.5,
        "w_max": 1.5,
        "goal_tolerance": 0.15,
        "robot_radius": 0.18,
        "collision_dist": 0.28,   # 真机碰撞判定距离（m）：雷达 range_min=0.15m，
                                  # 0.18 判定太晚会真撞上；0.28 留出惯性滑行余量
        "arena_size": 4.0,
    }

    def __init__(self, args):
        super().__init__("a2_real_robot_server")

        # 参数
        self.declare_parameter("port", args.port)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("lidar_front_offset_deg", args.lidar_front_offset_deg)
        self.declare_parameter("lidar_occluded_deg", args.lidar_occluded_deg)
        self.declare_parameter("min_linear_vel", args.min_linear_vel)
        self.declare_parameter("log_dir", str(args.log_dir))
        self.declare_parameter("goal_mode", args.goal_mode)
        self.declare_parameter("goal_relative_x", args.goal_relative_x)
        self.declare_parameter("goal_relative_y", args.goal_relative_y)

        self.port = self.get_parameter("port").value
        self.goal_mode = self.get_parameter("goal_mode").value
        self.goal_relative = Pose2D(
            self.get_parameter("goal_relative_x").value,
            self.get_parameter("goal_relative_y").value,
            0.0,
        )
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.lidar_front_offset_rad = math.radians(
            self.get_parameter("lidar_front_offset_deg").value
        )
        # 车身遮挡扇区（雷达原始角度，度）：该角度内的读数被车身/外壳遮挡，
        # 屏蔽为 max_range，避免碰撞误判与 obs 污染（2026-08-02 实测 120°~210°）。
        # 注意：屏蔽后该方向对 RL 表现为"开阔"，因此必须配合 --min-linear-vel 0
        # 禁止倒车，否则策略会倒向雷达盲区造成真碰撞。
        occ = self.get_parameter("lidar_occluded_deg").value
        if occ and "," in str(occ):
            a0, a1 = (float(x) for x in str(occ).split(","))
            self.lidar_occluded_rad = (math.radians(a0), math.radians(a1))
        else:
            self.lidar_occluded_rad = None
        self.min_linear_vel = float(
            self.get_parameter("min_linear_vel").value
        )
        self.log_dir = Path(self.get_parameter("log_dir").value).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 配置（reset 的 config_override 只允许改 max_episode_time）
        self.cfg = dict(self.DEFAULT_CFG)

        # ROS2 pub/sub
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, 10
        )

        # 数据锁
        self._lock = threading.Lock()
        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Pose2D] = None
        self.latest_twist = (0.0, 0.0)  # (v, w)
        self.latest_scan_time = 0.0
        self.latest_odom_time = 0.0
        self._scan_count = 0
        self._odom_count = 0

        # 状态机
        # WAIT_RESET, RECORD_GOAL, DRIVE_TO_START, RECORD_START, RUNNING, FINISHED
        self.state = "WAIT_RESET"
        self.websocket: Optional[WebSocketServerProtocol] = None

        # episode 状态
        self.episode_id = 0
        self.step_id = 0
        self.episode_t = 0.0
        self.goal_abs = Pose2D()
        self.start_abs = Pose2D()
        self._last_seed = -1
        self._step_in_progress = False
        self._step_task: Optional[asyncio.Task] = None

        # server 退出信号
        self._stop_event = asyncio.Event()

        # 日志
        self._episode_csv_path = self.log_dir / "episodes.csv"
        self._episode_csv_exists = self._episode_csv_path.exists()
        self._episode_csv_lock = threading.Lock()

    # ---------------- ROS2 回调 ----------------
    def _scan_callback(self, msg: LaserScan):
        with self._lock:
            self.latest_scan = msg
            self.latest_scan_time = time.time()
        self._scan_count += 1
        if self._scan_count % 10 == 0:
            log.debug("scan callback #%d", self._scan_count)

    def _odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        t = msg.twist.twist
        with self._lock:
            self.latest_odom = Pose2D(p.x, p.y, yaw)
            self.latest_twist = (float(t.linear.x), float(t.angular.z))
            self.latest_odom_time = time.time()
        self._odom_count += 1
        if self._odom_count % 60 == 0:
            log.debug("odom callback #%d", self._odom_count)

    # ---------------- 工具函数 ----------------
    def _send(self, msg: dict):
        if self.websocket is None:
            return
        asyncio.create_task(self._async_send(msg))

    async def _async_send(self, msg: dict):
        if self.websocket is None:
            return
        try:
            await self.websocket.send(json.dumps(msg, allow_nan=False))
        except Exception as e:
            log.error("发送消息失败: %s", e)

    def _error(self, code: str, detail: str):
        log.error("error %s: %s", code, detail)
        self._send({"type": "error", "code": code, "detail": detail})

    # ---------------- lidar 处理 ----------------
    def _resample_lidar(self, scan: LaserScan) -> np.ndarray:
        """
        把 /scan 重采样到 cfg['lidar_count'] 线。
        第 0 线 = 车头正前，逆时针排列。
        """
        n = self.cfg["lidar_count"]
        max_range = self.cfg["lidar_max_range"]
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = max_range
        ranges = np.clip(ranges, 0.0, max_range)

        m = len(ranges)
        raw_angles = scan.angle_min + np.arange(m, dtype=np.float32) * scan.angle_increment
        # 车身遮挡扇区（原始角度）：屏蔽为 max_range
        if self.lidar_occluded_rad is not None:
            a0, a1 = self.lidar_occluded_rad
            if a0 <= a1:
                mask = (raw_angles >= a0) & (raw_angles <= a1)
            else:  # 跨 0°（如 350°~10°）
                mask = (raw_angles >= a0) | (raw_angles <= a1)
            ranges[mask] = max_range
        # 安装偏移：车头正前在 raw scan 中的角度 = offset
        # 因此 target=0 时应取 raw angle ≈ offset 的值
        raw_angles = raw_angles - self.lidar_front_offset_rad

        target_angles = (np.arange(n, dtype=np.float32) * 2.0 * math.pi / n)
        # 角度差归一化到 [-π, π]，再取绝对值
        diffs = np.mod((raw_angles - target_angles[:, None]) + math.pi, 2.0 * math.pi) - math.pi
        diffs = np.abs(diffs)
        idx = np.argmin(diffs, axis=1)
        out = ranges[idx]
        return out

    @staticmethod
    def _median_filter_circular(arr: np.ndarray) -> np.ndarray:
        """3 邻域中位数滤波（环形）"""
        n = len(arr)
        out = np.empty_like(arr)
        for i in range(n):
            a = arr[(i - 1) % n]
            b = arr[i]
            c = arr[(i + 1) % n]
            out[i] = float(np.median([a, b, c]))
        return out

    # ---------------- 观测计算 ----------------
    def _get_current_pose(self) -> Optional[Pose2D]:
        with self._lock:
            return self.latest_odom

    def _get_latest_twist(self) -> tuple:
        with self._lock:
            return self.latest_twist

    def _get_latest_scan(self) -> Optional[LaserScan]:
        with self._lock:
            return self.latest_scan

    def _compute_goal_relative(self, current: Pose2D) -> tuple:
        """返回 (dist, bearing)，bearing 左正右负，范围 (-π, π]"""
        dx = self.goal_abs.x - current.x
        dy = self.goal_abs.y - current.y
        # 转到车体坐标系：x 前，y 左
        local_x = dx * math.cos(current.yaw) + dy * math.sin(current.yaw)
        local_y = -dx * math.sin(current.yaw) + dy * math.cos(current.yaw)
        dist = math.hypot(local_x, local_y)
        bearing = normalize_angle(math.atan2(local_y, local_x))
        return dist, bearing

    def _build_obs(self) -> dict:
        scan = self._get_latest_scan()
        current = self._get_current_pose()
        v_meas, w_meas = self._get_latest_twist()

        # lidar
        if scan is None:
            lidar = [self.cfg["lidar_max_range"]] * self.cfg["lidar_count"]
        else:
            raw = self._resample_lidar(scan)
            filtered = self._median_filter_circular(raw)
            lidar = [float(x) for x in filtered]

        # goal & vel
        if current is None:
            dist, bearing = 0.0, 0.0
            v_meas, w_meas = 0.0, 0.0
        else:
            dist, bearing = self._compute_goal_relative(current)

        flags = {
            "collision": False,
            "goal_reached": False,
            "timeout": False,
        }
        done = False

        min_lidar = min(lidar)
        if min_lidar < self.cfg["collision_dist"]:
            flags["collision"] = True
            done = True
        elif dist <= self.cfg["goal_tolerance"]:
            flags["goal_reached"] = True
            done = True
        elif self.episode_t >= self.cfg["max_episode_time"]:
            flags["timeout"] = True
            done = True

        return {
            "type": "obs",
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "t": round(self.episode_t, 3),
            "lidar": lidar,
            "goal": {"dist": round(dist, 3), "bearing": round(bearing, 3)},
            "vel": {"v": round(v_meas, 3), "w": round(w_meas, 3)},
            "flags": flags,
            "done": done,
        }

    # ---------------- 控制循环 ----------------
    def _publish_cmd(self, v: float, w: float):
        """发布一次 /cmd_vel。"""
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_vel_pub.publish(twist)

    def _safety_clamp(self, v: float, w: float, safe_dist: float = 0.30) -> tuple:
        """
        遥控/动作安全保护：前方（车头 ±30°）障碍 < safe_dist 时禁止前进，
        后方障碍 < safe_dist 时禁止后退。返回 clamped (v, w)。
        雷达 range_min=0.15m，safe_dist 取 0.30 保证有刹车余量。
        """
        scan = self._get_latest_scan()
        if scan is None or v == 0.0:
            return v, w
        try:
            lidar = self._resample_lidar(scan)
        except Exception:
            return v, w
        n = len(lidar)
        if n == 0:
            return v, w
        arc = max(1, int(30.0 / 360.0 * n))          # ±30° 对应的线数
        front = min(lidar[:arc] + lidar[n - arc:])   # index 0 = 车头正前
        rear = min(lidar[n // 2 - arc:n // 2 + arc])
        if v > 0 and front < safe_dist:
            log.info("safety: 前方 %.2fm < %.2fm，禁止前进", front, safe_dist)
            v = 0.0
        if v < 0 and rear < safe_dist:
            log.info("safety: 后方 %.2fm < %.2fm，禁止后退", rear, safe_dist)
            v = 0.0
        return v, w

    # ---------------- episode 推进 ----------------
    async def _run_step(self, v: float, w: float):
        """执行一个 control_dt，然后发 obs"""
        self._step_in_progress = True
        t0 = time.time()
        self._publish_cmd(v, w)
        t1 = time.time()

        await asyncio.sleep(self.cfg["control_dt"])
        t2 = time.time()

        self.step_id += 1
        self.episode_t = round(self.step_id * self.cfg["control_dt"], 3)
        obs = self._build_obs()
        t3 = time.time()
        self._send(obs)
        t4 = time.time()

        if obs["done"]:
            self._log_episode(obs)
            self.state = "WAIT_RESET"
            # 连发几次零速刹车，抵消真实小车惯性
            for _ in range(3):
                self._publish_cmd(0.0, 0.0)
                await asyncio.sleep(0.02)

        log.info("step timing: publish=%.3f sleep=%.3f build_obs=%.3f send=%.3f total=%.3f",
                 t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0)
        self._step_in_progress = False

    def _log_episode(self, obs: dict):
        outcome = "timeout"
        if obs["flags"]["collision"]:
            outcome = "collision"
        elif obs["flags"]["goal_reached"]:
            outcome = "goal_reached"
        with self._episode_csv_lock:
            mode = "a" if self._episode_csv_exists else "w"
            with open(self._episode_csv_path, mode, newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not self._episode_csv_exists:
                    w.writerow([
                        "episode_id", "steps", "outcome", "min_lidar_ever",
                        "final_dist", "seed", "timestamp"
                    ])
                    self._episode_csv_exists = True
                min_lidar = round(min(obs["lidar"]), 3)
                w.writerow([
                    self.episode_id, self.step_id, outcome, min_lidar,
                    obs["goal"]["dist"], self._last_seed,
                    datetime.now().isoformat()
                ])

    # ---------------- WebSocket 消息处理 ----------------
    def _handle_hello(self):
        self._send({
            "type": "hello",
            "protocol_version": self.PROTOCOL_VERSION,
            "env_name": self.ENV_NAME,
            "config": self.cfg,
        })

    async def _wait_for_odom(self, timeout: float = 1.5):
        """等待收到 /odom，最多 timeout 秒。"""
        deadline = time.time() + timeout
        while self._get_current_pose() is None and time.time() < deadline:
            await asyncio.sleep(0.05)
        return self._get_current_pose() is not None

    async def _handle_reset(self, msg: dict):
        if self.state not in ("WAIT_RESET", "FINISHED"):
            self._error("WRONG_STATE",
                        f"reset 只能在 WAIT_RESET/FINISHED 状态发，当前 {self.state}")
            return

        override = msg.get("config_override")
        if override is not None:
            if not isinstance(override, dict):
                self._error("BAD_FIELD", "config_override 必须是 object 或 null")
                return
            allowed = {"max_episode_time"}
            bad = set(override.keys()) - allowed
            if bad:
                self._error("BAD_FIELD", f"config_override 不允许的键: {bad}")
                return
            if "max_episode_time" in override:
                self.cfg["max_episode_time"] = float(override["max_episode_time"])

        seed = msg.get("seed", -1)
        self._last_seed = seed

        # 等待 /odom 就绪，避免用户秒回 human_confirm 时 current 为 None
        if not await self._wait_for_odom(timeout=1.5):
            self._error("INTERNAL", "等待 1.5s 仍未收到 /odom")
            return

        if self.goal_mode == "manual-drive":
            # manual-drive：先摆目标点，再推车/开车到起点（odom 记录位移）
            self.state = "RECORD_GOAL"
            log.info("reset 收到，seed=%s，goal_mode=manual-drive，"
                     "等待人工摆放目标点", seed)
            self._send({
                "type": "human",
                "action": "record_goal",
                "detail": "请把车放到目标点，摆好后发送 human_confirm('record_goal')",
            })
        else:
            # relative 模式：只需摆起点，goal 由 start + goal_relative 计算
            self.state = "RECORD_START"
            log.info("reset 收到，seed=%s，goal_mode=relative，"
                     "goal=(%.2f, %.2f)，等待人工摆放起点",
                     seed, self.goal_relative.x, self.goal_relative.y)
            self._send({
                "type": "human",
                "action": "record_start",
                "detail": "请把车放到起点（任意朝向），摆好后发送 human_confirm('record_start')",
            })

    def _start_episode(self):
        """从已记录的 start_abs 和 goal_abs 开始一局，发初始 obs。"""
        self.episode_id += 1
        self.step_id = 0
        self.episode_t = 0.0
        self.state = "RUNNING"
        log.info("episode %d 开始: start=(%.3f,%.3f) goal=(%.3f,%.3f)",
                 self.episode_id, self.start_abs.x, self.start_abs.y,
                 self.goal_abs.x, self.goal_abs.y)
        obs = self._build_obs()
        self._send(obs)

    def _handle_human_confirm(self, msg: dict):
        action = msg.get("action")
        current = self._get_current_pose()

        if action == "record_goal":
            if self.state != "RECORD_GOAL":
                self._error("WRONG_STATE",
                            f"当前不是 RECORD_GOAL 状态，而是 {self.state}")
                return
            if current is None:
                self._error("INTERNAL", "尚未收到 /odom，无法记录目标点")
                return
            self.goal_abs = Pose2D(current.x, current.y, current.yaw)
            log.info("记录目标点: x=%.3f y=%.3f yaw=%.3f",
                     current.x, current.y, current.yaw)
            if self.goal_mode == "manual-drive":
                self.state = "DRIVE_TO_START"
                self._send({
                    "type": "human",
                    "action": "drive_to_start",
                    "detail": "请用键盘遥控把车开到起点（可以任意旋转朝向），到位后发送 human_confirm('drive_to_start')",
                })
            else:
                self.state = "RECORD_START"
                self._send({
                    "type": "human",
                    "action": "record_start",
                    "detail": "请把车放到起点（任意朝向），摆好后发送 human_confirm('record_start')",
                })

        elif action == "drive_to_start":
            if self.state != "DRIVE_TO_START":
                self._error("WRONG_STATE",
                            f"当前不是 DRIVE_TO_START 状态，而是 {self.state}")
                return
            if current is None:
                self._error("INTERNAL", "尚未收到 /odom，无法记录起点")
                return
            self.start_abs = Pose2D(current.x, current.y, current.yaw)
            dx = self.start_abs.x - self.goal_abs.x
            dy = self.start_abs.y - self.goal_abs.y
            dist = math.hypot(dx, dy)
            log.info("记录起点: x=%.3f y=%.3f yaw=%.3f (距目标 %.3fm)",
                     current.x, current.y, current.yaw, dist)
            self._start_episode()

        elif action == "record_start":
            if self.state != "RECORD_START":
                self._error("WRONG_STATE",
                            f"当前不是 RECORD_START 状态，而是 {self.state}")
                return
            if current is None:
                self._error("INTERNAL", "尚未收到 /odom，无法记录起点")
                return
            self.start_abs = Pose2D(current.x, current.y, current.yaw)
            log.info("记录起点: x=%.3f y=%.3f yaw=%.3f",
                     current.x, current.y, current.yaw)

            if self.goal_mode == "relative":
                # 相对目标点：goal = start + R(yaw) * goal_relative
                # goal_relative 是车体坐标（x 前，y 左），要旋转到世界坐标
                cos_yaw = math.cos(self.start_abs.yaw)
                sin_yaw = math.sin(self.start_abs.yaw)
                self.goal_abs = Pose2D(
                    self.start_abs.x
                    + self.goal_relative.x * cos_yaw
                    - self.goal_relative.y * sin_yaw,
                    self.start_abs.y
                    + self.goal_relative.x * sin_yaw
                    + self.goal_relative.y * cos_yaw,
                    self.start_abs.yaw,
                )

            self._start_episode()

        else:
            self._error("BAD_FIELD", f"未知的 human_confirm action: {action}")

    def _handle_teleop(self, msg: dict):
        """DRIVE_TO_START 阶段遥控：直接发布 /cmd_vel。"""
        if self.state != "DRIVE_TO_START":
            self._error("WRONG_STATE",
                        f"teleop 只能在 DRIVE_TO_START 状态发，当前 {self.state}")
            return
        v = float(msg.get("v", 0.0))
        w = float(msg.get("w", 0.0))
        v = max(self.min_linear_vel, min(self.cfg["v_max"], v))
        w = max(-self.cfg["w_max"], min(self.cfg["w_max"], w))
        v, w = self._safety_clamp(v, w)
        log.info("teleop: v=%.3f w=%.3f", v, w)
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_vel_pub.publish(twist)

    def _handle_action(self, msg: dict):
        if self.state != "RUNNING":
            self._error("WRONG_STATE",
                        f"action 只能在 RUNNING 状态发，当前 {self.state}")
            return

        if msg.get("episode_id") != self.episode_id:
            self._error("BAD_FIELD",
                        f"episode_id 不匹配: 期望 {self.episode_id}, 收到 {msg.get('episode_id')}")
            return
        if msg.get("step_id") != self.step_id:
            self._error("BAD_FIELD",
                        f"step_id 不匹配: 期望 {self.step_id}, 收到 {msg.get('step_id')}")
            return

        if self._step_in_progress:
            self._error("WRONG_STATE", "上一个 action 尚未执行完，请勿重发")
            return

        v = float(msg.get("v", 0.0))
        w = float(msg.get("w", 0.0))
        v = max(self.min_linear_vel, min(self.cfg["v_max"], v))
        w = max(-self.cfg["w_max"], min(self.cfg["w_max"], w))

        self._step_task = asyncio.create_task(self._run_step(v, w))

    def _handle_all_finish(self, msg: dict):
        log.info("all_finish 收到，准备退出: %s", msg)
        self.state = "FINISHED"
        self._publish_cmd(0.0, 0.0)
        self._send({"type": "bye", "reason": "all_finish received"})
        self._stop_event.set()

    async def _handle_message(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            self._error("BAD_JSON", str(e))
            return

        mtype = msg.get("type")
        if mtype == "reset":
            await self._handle_reset(msg)
        elif mtype == "action":
            self._handle_action(msg)
        elif mtype == "teleop":
            self._handle_teleop(msg)
        elif mtype == "all_finish":
            self._handle_all_finish(msg)
        elif mtype == "human_confirm":
            self._handle_human_confirm(msg)
        else:
            self._error("BAD_TYPE", f"未知消息类型: {mtype}")

    # ---------------- WebSocket server ----------------
    async def _ws_handler(self, websocket: WebSocketServerProtocol, path: str):
        if self.websocket is not None:
            log.warning("已有 client 连接，拒绝新连接")
            await websocket.close(1013, "server busy")
            return

        log.info("client 已连接: %s", websocket.remote_address)
        self.websocket = websocket
        self.state = "WAIT_RESET"
        self._handle_hello()

        try:
            while True:
                raw = await websocket.recv()
                await self._handle_message(raw)
                # 让出控制权，确保 _run_step 等 task 能被调度
                await asyncio.sleep(0)
        except websockets.exceptions.ConnectionClosed:
            log.info("client 断开")
        finally:
            self.websocket = None
            self._publish_cmd(0.0, 0.0)
            self.state = "WAIT_RESET"

    async def run_server(self):
        log.info("启动 WebSocket server: 0.0.0.0:%d", self.port)
        async with websockets.serve(
            self._ws_handler, "0.0.0.0", self.port,
            ping_interval=None, ping_timeout=None,
        ):
            await self._stop_event.wait()  # 等待退出信号


def main():
    parser = argparse.ArgumentParser(description="A2 真机 WebSocket Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--lidar-front-offset-deg", type=float, default=0.0,
        help="雷达 0° 相对于车头正前的偏移（度）。正数表示车头正前在雷达原始 0° 的左侧"
    )
    parser.add_argument(
        "--lidar-occluded-deg", type=str, default="115,215",
        help="车身遮挡扇区（雷达原始角度范围，度，逗号分隔）：该角度的读数被车身遮挡，"
             "屏蔽为 max_range。本车实测 120°~210° 为车壳，默认 115,215"
    )
    parser.add_argument(
        "--min-linear-vel", type=float, default=0.0,
        help="v 下界（m/s）。遮挡扇区屏蔽后倒车是雷达盲区，默认 0.0 禁止倒车"
    )
    parser.add_argument("--log-dir", type=Path, default=Path("~/a2_real_robot_logs"))
    parser.add_argument(
        "--goal-mode", type=str, default="manual-drive",
        choices=["relative", "manual-drive"],
        help="目标点设定方式：relative=由起点相对偏移计算（测试用）；"
             "manual-drive=先摆放目标点，再推车/开车到起点（训练用）"
    )
    parser.add_argument(
        "--goal-relative-x", type=float, default=2.0,
        help="relative 模式下目标点相对于起点的 x 偏移（米，车头前为正）"
    )
    parser.add_argument(
        "--goal-relative-y", type=float, default=0.0,
        help="relative 模式下目标点相对于起点的 y 偏移（米，左侧为正）"
    )
    args = parser.parse_args()

    rclpy.init()
    node = RealRobotServer(args)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    try:
        asyncio.run(node.run_server())
    except KeyboardInterrupt:
        log.info("收到 Ctrl+C，退出")
    finally:
        node._publish_cmd(0.0, 0.0)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
