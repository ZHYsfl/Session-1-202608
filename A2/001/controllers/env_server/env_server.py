# -*- coding: utf-8 -*-
"""
env_server.py — A2 项目（VOA 强化学习防碰撞）001 硬件侧
Webots 仿真环境的 WebSocket server，实现 src/001/A2/api.md v1.0 协议。

运行方式：由 Webots 作为 rl_arena.wbt 中 ROBOT 节点的 controller 自动启动。
开发测试：配合 tools/random_client.py 使用。

协议要点（详见 api.md）：
  - 连接即发 hello；之后严格同步步进：收 action → 推进 0.1 s → 回 obs
  - reset(seed, config_override) 布置新场景并回初始 obs
  - 终止优先级：collision > goal_reached > timeout，三者互斥
  - 雷达无效回波替换为 max_range（JSON 传不了 inf）
"""

import asyncio
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:  # websockets >= 14
    from websockets.asyncio.server import serve
except ImportError:  # websockets < 14
    from websockets import serve

from controller import Supervisor

# ================= 全局常量（与 api.md §0 完全一致，改动必须同步升协议版本） =================
PROTOCOL_VERSION = "1.0"
ENV_NAME = "webots_diffbot_v1"
# 默认只监听本机回环（127.0.0.1）：本机训练直接用，且不会触发 Windows 防火墙弹窗。
# 若 003 需要从局域网另一台机器连接，改为 "0.0.0.0"（首次会弹防火墙授权，允许即可）。
WS_HOST, WS_PORT = "127.0.0.1", 8765

LIDAR_COUNT = 64
LIDAR_MAX_RANGE = 3.5
CONTROL_DT = 0.1            # s，一个 action 推进的仿真时间
PHYSICS_DT_MS = 10          # 与 world 文件 basicTimeStep 一致
SUBSTEPS = int(CONTROL_DT * 1000 / PHYSICS_DT_MS)  # 10
MAX_EPISODE_TIME = 30.0
V_MAX = 0.5                 # m/s
W_MAX = 1.5                 # rad/s
GOAL_TOLERANCE = 0.15       # m
ROBOT_RADIUS = 0.18         # m，碰撞判定半径（车身外接圆 ~0.170 + 余量）
ARENA_SIZE = 4.0            # m
OBS_DIM = 68
ACT_DIM = 2

# 底盘几何（与 .wbt 一致；真机对齐时改这里）
WHEEL_RADIUS = 0.05         # R
WHEEL_TRACK = 0.20          # L

# reset 采样约束（api.md §5.2）
START_GOAL_MIN_DIST = 2.0   # 起点-目标最小间距
SAMPLE_CLEARANCE = 0.4      # 起点/目标距障碍物表面的最小距离
ROBOT_SAMPLE_LIM = 1.6      # 起点/目标采样范围 [-1.6, 1.6]^2
OBSTACLE_POS_LIM = 1.5      # 障碍物中心采样范围
MIN_OBSTACLES = 5           # 每局激活障碍物数量 [5, 8]
N_OBSTACLES = 8
# 各障碍物外接圆半径（与 .wbt 几何一致，用于采样间距检查）
OBSTACLE_RADII = [0.212, 0.25, 0.177, 0.247, 0.15, 0.20, 0.18, 0.12]

# 围墙内表面位置（墙中心 ±2.0，厚 0.05）
WALL_INNER = ARENA_SIZE / 2 - 0.025  # 1.975

CONFIG_KEYS = {
    "lidar_count": LIDAR_COUNT,
    "lidar_max_range": LIDAR_MAX_RANGE,
    "obs_dim": OBS_DIM,
    "act_dim": ACT_DIM,
    "control_dt": CONTROL_DT,
    "max_episode_time": MAX_EPISODE_TIME,
    "v_max": V_MAX,
    "w_max": W_MAX,
    "goal_tolerance": GOAL_TOLERANCE,
    "robot_radius": ROBOT_RADIUS,
    "arena_size": ARENA_SIZE,
}
OVERRIDABLE_KEYS = {"max_episode_time"}  # reset 的 config_override 唯一允许的键

LOG_DIR = Path(__file__).resolve().parent / "logs"


def wrap_pi(a: float) -> float:
    """角度绕到 (-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class EnvServer:
    def __init__(self):
        self.robot = Supervisor()
        # 加速仿真（不影响仿真时间语义）
        self.robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

        # ---- 设备 ----
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(PHYSICS_DT_MS)
        self.motor_l = self.robot.getDevice("left wheel motor")
        self.motor_r = self.robot.getDevice("right wheel motor")
        self.pos_l = self.robot.getDevice("left wheel sensor")
        self.pos_r = self.robot.getDevice("right wheel sensor")
        for m in (self.motor_l, self.motor_r):
            m.setPosition(float("inf"))   # 速度控制模式
            m.setVelocity(0.0)
        for p in (self.pos_l, self.pos_r):
            p.enable(PHYSICS_DT_MS)

        # ---- supervisor 节点句柄 ----
        self.self_node = self.robot.getSelf()
        self.tf_translation = self.self_node.getField("translation")
        self.tf_rotation = self.self_node.getField("rotation")
        goal_node = self.robot.getFromDef("GOAL")
        self.tf_goal = goal_node.getField("translation")
        self.obstacles = []  # (translation_field, rotation_field, radius)
        for i in range(N_OBSTACLES):
            node = self.robot.getFromDef(f"OBSTACLE_{i}")
            self.obstacles.append((
                node.getField("translation"),
                node.getField("rotation"),
                OBSTACLE_RADII[i],
            ))

        # 记录世界文件里的初始位姿（标定雷达后还原用）
        self.robot.step(PHYSICS_DT_MS)  # 让设备出第一帧数据
        self.init_translation = self.tf_translation.getSFVec3f()
        self.init_rotation = self.tf_rotation.getSFRotation()

        # ---- episode 状态 ----
        self.episode_id = 0
        self.step_id = 0
        self.max_episode_time = MAX_EPISODE_TIME
        self.seed_used = -1
        self.goal_xy = (0.0, 0.0)
        self.min_lidar_ever = LIDAR_MAX_RANGE
        self.rng = np.random.default_rng()

        # ---- 日志 ----
        LOG_DIR.mkdir(exist_ok=True)
        log_path = LOG_DIR / "episodes.csv"
        new_file = not log_path.exists()
        self.log_fp = open(log_path, "a", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_fp)
        if new_file:
            # api.md §5.5 五列 + seed（附加列，便于复现场景）
            self.log_writer.writerow(
                ["episode_id", "steps", "outcome", "min_lidar_ever", "final_dist", "seed"])
            self.log_fp.flush()

        # ---- 雷达自动标定（确定原始数组 → api 顺序的重排索引） ----
        self._lidar_index = self._calibrate_lidar()

    # ================= 雷达重排标定 =================
    def _analytic_wall_ranges(self, x0, y0):
        """车停在 (x0, y0)、朝向 +x 时，k 号射线（角度 k*2pi/64，逆时针）到围墙的理论距离。"""
        d = np.full(LIDAR_COUNT, LIDAR_MAX_RANGE)
        for k in range(LIDAR_COUNT):
            th = k * 2 * math.pi / LIDAR_COUNT
            dx, dy = math.cos(th), math.sin(th)
            best = LIDAR_MAX_RANGE
            if abs(dx) > 1e-9:
                for wx in (WALL_INNER, -WALL_INNER):
                    t = (wx - x0) / dx
                    if 0.01 < t < best and abs(y0 + t * dy) <= WALL_INNER:
                        best = t
            if abs(dy) > 1e-9:
                for wy in (WALL_INNER, -WALL_INNER):
                    t = (wy - y0) / dy
                    if 0.01 < t < best and abs(x0 + t * dx) <= WALL_INNER:
                        best = t
            d[k] = min(best, LIDAR_MAX_RANGE)
        return d

    def _calibrate_lidar(self):
        """
        Webots 雷达数组的第 0 条朝向、排列方向随版本/节点实现而定。
        这里不猜约定：把车传送到已知位姿、对比实测与解析距离，
        在 128 种候选重排（2 方向 x 64 偏移）里选误差最小的一种。
        """
        # 撤走全部障碍物（沉到地板下），避免干扰标定
        for tf_trans, _, _ in self.obstacles:
            tf_trans.setSFVec3f([50.0, 50.0, -1.0])
        self.tf_translation.setSFVec3f([1.5, 0.0, 0.0])
        self.tf_rotation.setSFRotation([0, 0, 1, 0])
        self.robot.simulationResetPhysics()
        for _ in range(5):
            self.robot.step(PHYSICS_DT_MS)

        raw = self._read_lidar_raw()
        truth = self._analytic_wall_ranges(1.5, 0.0)

        best_err, best_index = float("inf"), None
        for direction in (1, -1):
            for off in range(LIDAR_COUNT):
                idx = [(off + direction * k) % LIDAR_COUNT for k in range(LIDAR_COUNT)]
                err = float(np.mean(np.abs(raw[idx] - truth)))
                if err < best_err:
                    best_err, best_index = err, idx
        direction_name = "CCW" if best_index[1] == (best_index[0] + 1) % LIDAR_COUNT else "CW"
        print(f"[env_server] lidar 标定: 方向={direction_name} 平均误差={best_err * 1000:.1f} mm",
              flush=True)
        if best_err > 0.08:
            print("[env_server] 警告: 雷达标定误差偏大，请检查场地尺寸/雷达配置是否被改动",
                  flush=True)

        # 还原机器人初始位姿
        self.tf_translation.setSFVec3f(self.init_translation)
        self.tf_rotation.setSFRotation(self.init_rotation)
        self.robot.simulationResetPhysics()
        for _ in range(3):
            self.robot.step(PHYSICS_DT_MS)
        return np.array(best_index)

    # ================= 传感器读取 =================
    def _read_lidar_raw(self):
        """原始雷达数组：inf/nan → max_range，clip 到 [0, max_range]。"""
        r = np.asarray(self.lidar.getRangeImage(), dtype=np.float64)
        if r.size != LIDAR_COUNT:
            raise RuntimeError(
                f"雷达数组长度 {r.size} != {LIDAR_COUNT}，请检查 .wbt 中 Lidar 配置")
        r = np.where(np.isfinite(r), r, LIDAR_MAX_RANGE)
        return np.clip(r, 0.0, LIDAR_MAX_RANGE)

    def _read_lidar(self):
        """重排到 api 顺序（第 0 条 = 车头正前，逆时针排列）。"""
        return self._read_lidar_raw()[self._lidar_index]

    def _pose(self):
        """返回 (x, y, yaw)，supervisor 真值。"""
        x, y, _ = self.tf_translation.getSFVec3f()
        o = self.self_node.getOrientation()  # 行优先旋转矩阵
        yaw = math.atan2(o[3], o[0])
        return x, y, yaw

    # ================= obs 组装 =================
    def _build_obs(self, v_meas, w_meas, collision, goal_reached, timeout):
        x, y, yaw = self._pose()
        gx, gy = self.goal_xy
        dist = math.hypot(gx - x, gy - y)
        bearing = wrap_pi(math.atan2(gy - y, gx - x) - yaw)
        done = collision or goal_reached or timeout
        return {
            "type": "obs",
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "t": round(self.step_id * CONTROL_DT, 6),
            "lidar": [round(float(v), 4) for v in self._read_lidar()],
            "goal": {"dist": round(dist, 4), "bearing": round(bearing, 4)},
            "vel": {"v": round(v_meas, 4), "w": round(w_meas, 4)},
            "flags": {
                "collision": collision,
                "goal_reached": goal_reached,
                "timeout": timeout,
            },
            "done": done,
        }

    # ================= reset（api.md §2.2 / §5.2） =================
    def reset(self, seed, config_override):
        if seed == -1:
            seed = int(np.random.SeedSequence().entropy) % (2 ** 31)
        self.seed_used = seed
        self.rng = np.random.default_rng(seed)
        self.max_episode_time = MAX_EPISODE_TIME
        if config_override is not None:
            self.max_episode_time = float(config_override["max_episode_time"])

        # 1) 采起点与目标（间距 >= 2.0 m）
        for _ in range(200):
            sx, sy = self.rng.uniform(-ROBOT_SAMPLE_LIM, ROBOT_SAMPLE_LIM, 2)
            gx, gy = self.rng.uniform(-ROBOT_SAMPLE_LIM, ROBOT_SAMPLE_LIM, 2)
            if math.hypot(gx - sx, gy - sy) >= START_GOAL_MIN_DIST:
                break
        self.goal_xy = (float(gx), float(gy))

        # 2) 随机摆 N 个障碍物（互不重叠、离起点/目标表面 >= 0.4 m）
        n_active = int(self.rng.integers(MIN_OBSTACLES, N_OBSTACLES + 1))
        order = self.rng.permutation(N_OBSTACLES)
        placed = []  # (x, y, r)
        for rank, obs_i in enumerate(order):
            tf_trans, tf_rot, r = self.obstacles[obs_i]
            if rank < n_active:
                pos = None
                for _ in range(100):
                    ox, oy = self.rng.uniform(-OBSTACLE_POS_LIM, OBSTACLE_POS_LIM, 2)
                    if math.hypot(ox - sx, oy - sy) < r + SAMPLE_CLEARANCE:
                        continue
                    if math.hypot(ox - gx, oy - gy) < r + SAMPLE_CLEARANCE:
                        continue
                    if any(math.hypot(ox - px, oy - py) < r + pr + 0.3
                           for px, py, pr in placed):
                        continue
                    pos = (float(ox), float(oy))
                    break
                if pos is None:  # 摆不下就弃用这个障碍物
                    tf_trans.setSFVec3f([50.0, 50.0, -1.0])
                    continue
                placed.append((pos[0], pos[1], r))
                tf_trans.setSFVec3f([pos[0], pos[1], self._obstacle_z(obs_i)])
                tf_rot.setSFRotation([0, 0, 1, float(self.rng.uniform(0, 2 * math.pi))])
            else:
                tf_trans.setSFVec3f([50.0, 50.0, -1.0])  # 闲置的沉到地板下

        # 3) 传送机器人到起点，随机朝向
        self.tf_translation.setSFVec3f([float(sx), float(sy), 0.0])
        self.tf_rotation.setSFRotation([0, 0, 1, float(self.rng.uniform(-math.pi, math.pi))])
        self.tf_goal.setSFVec3f([self.goal_xy[0], self.goal_xy[1], 0.005])

        # 4) 停电机、清物理、静置几拍让传感器刷新
        for m in (self.motor_l, self.motor_r):
            m.setVelocity(0.0)
        self.robot.simulationResetPhysics()
        for _ in range(3):
            self.robot.step(PHYSICS_DT_MS)

        # 5) episode 状态复位，回初始 obs
        self.episode_id += 1
        self.step_id = 0
        self.min_lidar_ever = LIDAR_MAX_RANGE
        self.min_lidar_ever = float(np.min(self._read_lidar()))
        print(f"[env_server] episode {self.episode_id} 开始 (seed={seed}, "
              f"障碍物={len(placed)}, 目标=({self.goal_xy[0]:.2f},{self.goal_xy[1]:.2f}))",
              flush=True)
        return self._build_obs(0.0, 0.0, False, False, False)

    def _obstacle_z(self, obs_i):
        # 障碍物底面贴地：按 .wbt 中几何高度取半高
        heights = [0.4, 0.35, 0.3, 0.45, 0.4, 0.35, 0.45, 0.3]
        return heights[obs_i] / 2

    # ================= action（api.md §2.4 / §5.3） =================
    def step_action(self, v, w):
        v = float(np.clip(v, -V_MAX, V_MAX))
        w = float(np.clip(w, -W_MAX, W_MAX))

        # 差速运动学：api.md §5.3  ω_r = (v + w*L/2)/R, ω_l = (v - w*L/2)/R
        omega_r = (v + w * WHEEL_TRACK / 2) / WHEEL_RADIUS
        omega_l = (v - w * WHEEL_TRACK / 2) / WHEEL_RADIUS
        self.motor_r.setVelocity(omega_r)
        self.motor_l.setVelocity(omega_l)

        pos_l_prev, pos_r_prev = self.pos_l.getValue(), self.pos_r.getValue()

        # 推进 SUBSTEPS 个物理步，每个物理步都判碰撞（§5.3）
        collision = False
        executed = 0
        for _ in range(SUBSTEPS):
            if self.robot.step(PHYSICS_DT_MS) == -1:
                raise SystemExit("Webots 仿真已退出")
            executed += 1
            m = float(np.min(self._read_lidar()))
            self.min_lidar_ever = min(self.min_lidar_ever, m)
            if m < ROBOT_RADIUS:
                collision = True
                break

        dt = executed * PHYSICS_DT_MS / 1000.0
        # 实测速度：由轮速（编码器）换算，api.md §2.3
        om_l = (self.pos_l.getValue() - pos_l_prev) / dt
        om_r = (self.pos_r.getValue() - pos_r_prev) / dt
        v_meas = (om_l + om_r) * WHEEL_RADIUS / 2
        w_meas = (om_r - om_l) * WHEEL_RADIUS / WHEEL_TRACK

        # 终止判定，优先级 collision > goal_reached > timeout（§5.4）
        x, y, yaw = self._pose()
        goal_reached = (not collision) and (
            math.hypot(self.goal_xy[0] - x, self.goal_xy[1] - y) <= GOAL_TOLERANCE)
        self.step_id += 1
        timeout = (not collision and not goal_reached
                   and self.step_id * CONTROL_DT >= self.max_episode_time - 1e-9)

        obs = self._build_obs(v_meas, w_meas, collision, goal_reached, timeout)
        if obs["done"]:
            outcome = ("collision" if collision else
                       "goal_reached" if goal_reached else "timeout")
            self._log_episode(outcome, math.hypot(self.goal_xy[0] - x, self.goal_xy[1] - y))
            for m_ in (self.motor_l, self.motor_r):
                m_.setVelocity(0.0)
        return obs

    def _log_episode(self, outcome, final_dist):
        self.log_writer.writerow([
            self.episode_id, self.step_id, outcome,
            round(self.min_lidar_ever, 4), round(final_dist, 4), self.seed_used])
        self.log_fp.flush()
        print(f"[env_server] episode {self.episode_id} 结束: {outcome}, "
              f"步数={self.step_id}, 最终距离={final_dist:.2f} m", flush=True)

    def stop_motors(self):
        for m in (self.motor_l, self.motor_r):
            m.setVelocity(0.0)


# ================= WebSocket 协议层 =================
ENV = None  # 在 main 里初始化


def hello_msg():
    return {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "env_name": ENV_NAME,
        "config": dict(CONFIG_KEYS),
    }


async def send_error(ws, code, detail):
    await ws.send(json.dumps({"type": "error", "code": code, "detail": detail}))
    print(f"[env_server] 发出 error {code}: {detail}", flush=True)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


async def handle_client(ws, path=None):
    """单客户端协议状态机：WAIT_RESET ⇄ RUNNING → 关闭（api.md §3.1）"""
    peer = getattr(ws, "remote_address", "?")
    print(f"[env_server] 客户端已连接: {peer}", flush=True)
    await ws.send(json.dumps(hello_msg()))
    state = "WAIT_RESET"
    try:
        async for raw in ws:
            # ---- JSON 解析 ----
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                await send_error(ws, "BAD_JSON", f"无法解析 JSON: {raw[:200]!r}")
                break
            if not isinstance(msg, dict) or "type" not in msg:
                await send_error(ws, "BAD_TYPE", "消息缺少 type 字段")
                break
            mtype = msg["type"]

            # ---- reset ----
            if mtype == "reset":
                if state != "WAIT_RESET":
                    await send_error(ws, "WRONG_STATE", "RUNNING 状态下收到 reset")
                    break
                seed = msg.get("seed")
                if not isinstance(seed, int) or isinstance(seed, bool):
                    await send_error(ws, "BAD_FIELD", "seed 必须是 int（-1 表示随机）")
                    break
                override = msg.get("config_override", "__missing__")
                if override == "__missing__":
                    await send_error(ws, "BAD_FIELD", "缺少 config_override 字段（不用时填 null）")
                    break
                if override is not None:
                    if not isinstance(override, dict):
                        await send_error(ws, "BAD_FIELD", "config_override 必须是 object 或 null")
                        break
                    bad = set(override) - OVERRIDABLE_KEYS
                    if bad:
                        await send_error(ws, "BAD_FIELD",
                                         f"config_override 不允许的键: {sorted(bad)}，"
                                         f"仅支持 {sorted(OVERRIDABLE_KEYS)}")
                        break
                    if not _is_number(override["max_episode_time"]) \
                            or override["max_episode_time"] <= 0:
                        await send_error(ws, "BAD_FIELD", "max_episode_time 必须是正数")
                        break
                obs = ENV.reset(seed, override)
                state = "RUNNING"
                await ws.send(json.dumps(obs))

            # ---- action ----
            elif mtype == "action":
                if state != "RUNNING":
                    await send_error(ws, "WRONG_STATE", "等待 reset 时收到 action")
                    break
                if (msg.get("episode_id") != ENV.episode_id
                        or msg.get("step_id") != ENV.step_id):
                    await send_error(
                        ws, "BAD_FIELD",
                        f"action id 不匹配: 期望 ({ENV.episode_id}, {ENV.step_id})，"
                        f"收到 ({msg.get('episode_id')}, {msg.get('step_id')})")
                    break
                if not (_is_number(msg.get("v")) and _is_number(msg.get("w"))):
                    await send_error(ws, "BAD_FIELD", "v / w 必须是有限数值")
                    break
                obs = ENV.step_action(msg["v"], msg["w"])
                await ws.send(json.dumps(obs))
                if obs["done"]:
                    state = "WAIT_RESET"

            # ---- all_finish ----
            elif mtype == "all_finish":
                reason = msg.get("reason")
                total = msg.get("total_episodes")
                if reason not in ("converged", "interrupted", "error") \
                        or not isinstance(total, int):
                    await send_error(ws, "BAD_FIELD",
                                     "reason 须为 converged/interrupted/error，"
                                     "total_episodes 须为 int")
                    break
                print(f"[env_server] all_finish: {reason}, 共 {total} episodes", flush=True)
                ENV.stop_motors()
                await ws.send(json.dumps({"type": "bye", "reason": "all_finish received"}))
                break

            else:
                await send_error(ws, "BAD_TYPE", f"未知消息类型: {mtype!r}")
                break

    except SystemExit:
        raise
    except Exception as exc:  # 连接断开或内部异常
        print(f"[env_server] 连接结束: {type(exc).__name__}: {exc}", flush=True)

    print("[env_server] 会话结束，关闭仿真", flush=True)
    ENV.stop_motors()
    ENV.robot.simulationQuit(0)


async def _serve():
    async with serve(handle_client, WS_HOST, WS_PORT,
                     ping_interval=20, ping_timeout=60, max_size=1 << 20):
        print(f"[env_server] WebSocket server 已启动: ws://{WS_HOST}:{WS_PORT}", flush=True)
        await asyncio.Future()  # 一直运行


def main():
    global ENV
    ENV = EnvServer()
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
    finally:
        ENV.log_fp.close()


if __name__ == "__main__":
    main()
