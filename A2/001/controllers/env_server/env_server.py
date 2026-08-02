# -*- coding: utf-8 -*-
"""
env_server.py — A2 项目（VOA 强化学习防碰撞）001 硬件侧
Webots 仿真环境的 WebSocket server，实现 src/001/A2/api.md v1.1 协议。

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
import os
import sys
from pathlib import Path

import numpy as np

try:  # websockets >= 14
    from websockets.asyncio.server import serve
except ImportError:  # websockets < 14
    from websockets import serve

from controller import Supervisor

# ================= 全局常量（与 api.md §0 完全一致，改动必须同步升协议版本） =================
PROTOCOL_VERSION = "1.1"
ENV_NAME = "webots_diffbot_v1"
# 默认只监听本机回环（127.0.0.1）：本机训练直接用，且不会触发 Windows 防火墙弹窗。
# 若 003 需要从局域网另一台机器连接，改为 "0.0.0.0"（首次会弹防火墙授权，允许即可）。
# 异步 RL 多实例：每个 webots 实例用环境变量 ENV_WS_PORT 指定各自端口
# （启动器 export 后 webots 把它传给 controller 进程）；缺省 8765 保持 api.md 约定。
WS_HOST = "127.0.0.1"
WS_PORT = int(os.environ.get("ENV_WS_PORT", "8765"))

LIDAR_COUNT = 64
LIDAR_MAX_RANGE = 3.5
CONTROL_DT = 0.1            # s，一个 action 推进的仿真时间
PHYSICS_DT_MS = 10          # 与 world 文件 basicTimeStep 一致
SUBSTEPS = int(CONTROL_DT * 1000 / PHYSICS_DT_MS)  # 10
MAX_EPISODE_TIME = 60.0     # 默认单局时限 60 s（v1.2 起放宽：绕障往返需 15~25 s，30 s 太紧）
V_MAX = 0.5                 # m/s
W_MAX = 1.5                 # rad/s
GOAL_TOLERANCE = 0.15       # m
ROBOT_RADIUS = 0.18         # m，碰撞判定半径（车身外接圆 ~0.170 + 余量）
COLLISION_DIST = 0.28       # m，真机对齐（2026-08-02）：雷达 range_min=0.15m，
                            # 0.18 判定太晚会真撞；0.28 留出惯性滑行余量
MIN_LINEAR_VEL = 0.0        # m/s，真机对齐：后方是雷达盲区（车壳遮挡），禁止倒车
# 模拟真机车壳遮挡扇区（车体角度，度，0=车头正前，逆时针）：该扇区读数屏蔽为
# max_range。真机实测原始角度 115°~215° + offset -61.88° ≈ 车体 177°~277°。
LIDAR_OCCLUDED_BODY = (177.0, 277.0)
ARENA_SIZE = 4.0            # m
OBS_DIM = 68
ACT_DIM = 2

# 底盘几何（与 .wbt 一致；真机对齐时改这里）
WHEEL_RADIUS = 0.05         # R
WHEEL_TRACK = 0.20          # L

# reset 采样约束（api.md §5.2）
START_GOAL_MIN_DIST = 2.0   # 起点-目标最小间距
SAMPLE_CLEARANCE = 0.4      # 起点/目标距障碍物表面的最小距离
OBSTACLE_GAP = 0.55         # 障碍物表面之间的最小距离；需 > 车身直径 0.36 才过得去
ROBOT_SAMPLE_LIM = 1.6      # 起点/目标采样范围 [-1.6, 1.6]^2
OBSTACLE_POS_LIM = 1.5      # 障碍物中心采样范围
MIN_ACTIVE = 5              # 每局激活障碍物数量区间 [MIN_ACTIVE, MAX_ACTIVE]
MAX_ACTIVE = 8              # 调成相等即固定数量；课程学习可先稀疏后加密
N_OBSTACLES = 8             # 世界文件里可用的障碍物节点总数
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
    "collision_dist": COLLISION_DIST,
    "min_linear_vel": MIN_LINEAR_VEL,
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
        # 仿真速度：训练默认 FAST（最大化吞吐）；GUI 观看演示时用
        # ENV_SIM_MODE=realtime（由启动命令设置，WSLENV 透传）
        if os.environ.get("ENV_SIM_MODE", "fast") == "realtime":
            self.robot.simulationSetMode(
                Supervisor.SIMULATION_MODE_REAL_TIME)
        else:
            self.robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST)

        # ---- 设备 ----
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(PHYSICS_DT_MS)
        self.lidar.enablePointCloud()
        self.yaw = 0.0   # 运动学朝向（无轮：由 (v,w) 直接积分，见 step_action）

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

        实战教训：标定位姿必须同时偏离 x、y 两条对称轴（取 (1.5, 0.7)）。
        曾经把车放在 (1.5, 0)——左/右墙等距，场景镜像对称，CCW/CW 两种
        拟合误差完全相同，方向纯粹是蒙的；蒙错的后果是 obs 里雷达左右镜像，
        脚本专家的"避障"变成"朝障碍撞"，三种控制律成功率全部只有 ~30%，
        且碰撞复核全部"真实"（几何确实碰上了，是观测把方向报反了）。
        两个方向的误差必须悬殊，否则视为标定不可信。
        """
        # 撤走全部障碍物（沉到地板下），避免干扰标定
        for tf_trans, _, _ in self.obstacles:
            tf_trans.setSFVec3f([50.0, 50.0, -1.0])
        # 目标标记也撤走（即便改成 Transform 后理论不可见，标定场景必须空场验证）
        self.tf_goal.setSFVec3f([50.0, 50.0, -1.0])
        self.tf_translation.setSFVec3f([1.5, 0.7, 0.0])
        self.tf_rotation.setSFRotation([0, 0, 1, 0])
        self.robot.simulationResetPhysics()
        for _ in range(5):
            self.robot.step(PHYSICS_DT_MS)

        raw = self._read_lidar_raw()
        truth = self._analytic_wall_ranges(1.5, 0.7)

        best_err, best_index = float("inf"), None
        dir_best_err = {}
        for direction in (1, -1):
            d_err = float("inf")
            for off in range(LIDAR_COUNT):
                idx = [(off + direction * k) % LIDAR_COUNT for k in range(LIDAR_COUNT)]
                err = float(np.mean(np.abs(raw[idx] - truth)))
                if err < d_err:
                    d_err = err
                    if err < best_err:
                        best_err, best_index = err, idx
            dir_best_err[direction] = d_err
        direction_name = "CCW" if best_index[1] == (best_index[0] + 1) % LIDAR_COUNT else "CW"
        print(f"[env_server] lidar 标定: 方向={direction_name} 平均误差={best_err * 1000:.1f} mm "
              f"(CCW最优={dir_best_err[1] * 1000:.0f} mm, CW最优={dir_best_err[-1] * 1000:.0f} mm)",
              flush=True)
        if min(dir_best_err.values()) * 3 > max(dir_best_err.values()):
            print("[env_server] 警告: 两个方向的标定误差相近——场景对称，方向不可信！",
                  flush=True)
        # DEBUG（标定审计）：逐射线残差，确认"最优重排"是否真的每根射线都对得上
        resid = np.abs(raw[best_index] - truth)
        worst = int(np.argmax(resid))
        bad = [f"k{k}∠{k * 360.0 / LIDAR_COUNT:.0f}° raw={raw[best_index[k]]:.2f} "
               f"truth={truth[k]:.2f}" for k in range(LIDAR_COUNT) if resid[k] > 0.1]
        print(f"[env_server] DEBUG 标定: 最大残差={resid[worst]:.3f} m @ 射线{worst} "
              f"残差>0.1m 的射线数={int(np.sum(resid > 0.1))} "
              f"重排前8={list(best_index[:8])}\n"
              f"[env_server] DEBUG 异常射线: {'; '.join(bad)}",
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

        # ---- DEBUG 静态探针（两个决定性检验）----
        # 检验1：空场（障碍+目标全撤走）静止车，任何 <3.5 的读数都是自检测幻影
        for tf_trans, _, _ in self.obstacles:
            tf_trans.setSFVec3f([50.0, 50.0, -1.0])
        self.tf_goal.setSFVec3f([50.0, 50.0, -1.0])
        self.tf_translation.setSFVec3f([0.0, 0.0, 0.0])
        self.tf_rotation.setSFRotation([0, 0, 1, 0])
        self.robot.simulationResetPhysics()
        for _ in range(5):
            self.robot.step(PHYSICS_DT_MS)
        lid = self._read_lidar_raw()  # 标定中 _lidar_index 尚未生成，用原始序
        m, k = float(np.min(lid)), int(np.argmin(lid))
        print(f"[env_server] DEBUG 空场探针: min={m:.3f} @k{k} "
              f"(>3.4 为正常；<0.2 为自检测幻影)", flush=True)
        # 检验2：车停在目标旁 0.15 m 正对它，看目标标记是否被雷达看到
        self.tf_goal.setSFVec3f([1.2, 1.2, 0.005])
        self.tf_translation.setSFVec3f([1.2, 1.35, 0.0])
        self.tf_rotation.setSFRotation([0, 0, 1, -math.pi / 2])
        self.robot.simulationResetPhysics()
        for _ in range(5):
            self.robot.step(PHYSICS_DT_MS)
        lid = self._read_lidar_raw()
        m, k = float(np.min(lid)), int(np.argmin(lid))
        print(f"[env_server] DEBUG 目标探针: min={m:.3f} @k{k} "
              f"(若 ≈0.15 则目标标记对雷达可见——成功圈不可达；应 ≫0.18)",
              flush=True)
        self.tf_goal.setSFVec3f([50.0, 50.0, -1.0])
        self.tf_translation.setSFVec3f(self.init_translation)
        self.tf_rotation.setSFRotation(self.init_rotation)
        self.robot.simulationResetPhysics()
        for _ in range(3):
            self.robot.step(PHYSICS_DT_MS)

        # 检验3：静止车 + 两个已知障碍物（已知位姿），连读 10 帧
        # 区分"幻影来自运动"还是"幻影来自障碍物/传感器配置"
        o0_t, _, o0_r = self.obstacles[0]
        o1_t, _, o1_r = self.obstacles[1]
        o0_t.setSFVec3f([0.8, 0.3, self._obstacle_z(0)])
        o1_t.setSFVec3f([-0.5, 0.6, self._obstacle_z(1)])
        self.tf_translation.setSFVec3f([0.0, 0.0, 0.0])
        self.tf_rotation.setSFRotation([0, 0, 1, 0])
        self.robot.simulationResetPhysics()
        worst = (3.5, -1)
        for _ in range(10):
            self.robot.step(PHYSICS_DT_MS)
            lid = self._read_lidar_raw()
            m, k = float(np.min(lid)), int(np.argmin(lid))
            worst = min(worst, (m, k))
        # 理论：障碍0(Box 0.3x0.3)中心(0.8,0.3)→前表面约 0.65；障碍1中心(-0.5,0.6)
        print(f"[env_server] DEBUG 静态障碍探针: 10帧最小读数={worst[0]:.3f} @k{worst[1]} "
              f"(应 ≈0.65；<0.5 即幻影)", flush=True)
        # 还原
        for tf_trans, _, _ in self.obstacles:
            tf_trans.setSFVec3f([50.0, 50.0, -1.0])
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

    @staticmethod
    def _median3(lid):
        """逐射线 3 邻域中位数滤波（环形），返回等长数组。
        a+b+c-min-max 即逐元素 median(lid[k-1], lid[k], lid[k+1]) 的无循环实现。"""
        a, b, c = np.roll(lid, 1), lid, np.roll(lid, -1)
        return a + b + c - np.minimum(np.minimum(a, b), c) - np.maximum(np.maximum(a, b), c)

    @classmethod
    def _despiked_min(cls, lid):
        """空间相干最小值：3 邻域中位数滤波后取 min。

        实战教训：Webots Lidar 在运动中会偶发孤立单射线尖峰（读数 0.05~0.18 m
        但对应方向 0.3~3 m 内无任何实物，静止时不出现），直接把 min(lidar)
        送进碰撞判定会产生大量幻影碰撞。真实近物在近距离必被 ≥2 根相邻射线
        同时看到，中位数滤波保留它们、抹除孤立尖峰。对已滤波的扫描幂等。
        """
        return float(np.min(cls._median3(lid)))

    def _read_lidar(self):
        """重排到 api 顺序（第 0 条 = 车头正前，逆时针排列）并做中位数滤波。

        obs、碰撞判定、客户端奖励统一使用这份滤波后的扫描，保证全链路语义一致：
        尖峰既然是传感器伪影，就不该出现在任何下游消费者的输入里。
        同时模拟真机车壳遮挡：LIDAR_OCCLUDED_BODY 扇区屏蔽为 max_range。
        """
        lid = self._median3(self._read_lidar_raw()[self._lidar_index])
        i0 = int(LIDAR_OCCLUDED_BODY[0] / 360.0 * LIDAR_COUNT)
        i1 = int(LIDAR_OCCLUDED_BODY[1] / 360.0 * LIDAR_COUNT)
        lid[i0:i1] = LIDAR_MAX_RANGE
        return lid

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
        n_active = int(self.rng.integers(MIN_ACTIVE, MAX_ACTIVE + 1))
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
                    if any(math.hypot(ox - px, oy - py) < r + pr + OBSTACLE_GAP
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

        # 4) 运动学模式无电机；记录朝向并静置几拍让传感器刷新
        self.yaw = float(self.tf_rotation.getSFRotation()[3])
        self.robot.simulationResetPhysics()
        for _ in range(3):
            self.robot.step(PHYSICS_DT_MS)

        # 5) episode 状态复位，回初始 obs
        self.episode_id += 1
        self.step_id = 0
        self.min_lidar_ever = float(np.min(self._read_lidar()))
        self.max_tilt = 0.0      # DEBUG 尖峰溯源
        self.spike_info = None   # DEBUG 尖峰溯源
        self.spikes_suppressed = 0  # 被空间相干滤波拦下的孤立尖峰计数
        print(f"[env_server] episode {self.episode_id} 开始 (seed={seed}, "
              f"障碍物={len(placed)}, 目标=({self.goal_xy[0]:.2f},{self.goal_xy[1]:.2f}))",
              flush=True)
        # DEBUG（几何审计）：场景布局 + 起点位姿，供离线复算碰撞是否真实
        print(f"[env_server] DEBUG 场景: 起点=({sx:.3f},{sy:.3f},yaw={float(self.tf_rotation.getSFRotation()[3]):.3f}) "
              f"障碍={[(round(px,3), round(py,3), r) for px, py, r in placed]}",
              flush=True)
        return self._build_obs(0.0, 0.0, False, False, False)

    def _obstacle_z(self, obs_i):
        # 障碍物底面贴地：按 .wbt 中几何高度取半高
        heights = [0.4, 0.35, 0.3, 0.45, 0.4, 0.35, 0.45, 0.3]
        return heights[obs_i] / 2

    # ================= action（api.md §2.4 / §5.3） =================
    def step_action(self, v, w):
        v = float(np.clip(v, MIN_LINEAR_VEL, V_MAX))   # 真机对齐：禁止倒车
        w = float(np.clip(w, -W_MAX, W_MAX))

        # 理想运动学积分（无轮，api.md §5.3）：yaw += w·dt, pos += v·dt·(cos,sin)
        # 每 10 ms 子步推进一次并即时判碰撞（§5.3），运动精确、无动力学误差
        collision = False
        executed = 0
        for _ in range(SUBSTEPS):
            dt_s = PHYSICS_DT_MS / 1000.0
            self.yaw += w * dt_s
            self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
            x, y, z = self.tf_translation.getSFVec3f()
            self.tf_translation.setSFVec3f([
                x + v * math.cos(self.yaw) * dt_s,
                y + v * math.sin(self.yaw) * dt_s, z])
            self.tf_rotation.setSFRotation([0, 0, 1, self.yaw])
            if self.robot.step(PHYSICS_DT_MS) == -1:
                raise SystemExit("Webots 仿真已退出")
            executed += 1
            lid_raw = self._read_lidar_raw()[self._lidar_index]  # 未滤波（诊断对照用）
            lid = self._median3(lid_raw)                         # 滤波后：判定与 obs 统一用
            m_raw = float(np.min(lid_raw))
            m = float(np.min(lid))                # 空间相干最小值：孤立尖峰不参与判定
            self.min_lidar_ever = min(self.min_lidar_ever, m)
            if m_raw < 0.25 < m:
                self.spikes_suppressed += 1
            # DEBUG：尖峰溯源——首次 RAW <0.3 时记录瞬时俯仰/横滚；全程统计最大姿态角
            o = self.self_node.getOrientation()
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, -o[2]))))
            roll = math.degrees(math.asin(max(-1.0, min(1.0, o[5]))))
            self.max_tilt = max(self.max_tilt, abs(pitch), abs(roll))
            if m_raw < 0.3 and self.spike_info is None:
                self.spike_info = (m_raw, int(np.argmin(lid_raw)), pitch, roll)
            if m < COLLISION_DIST:
                collision = True
                # DEBUG（几何审计）：碰撞瞬间的位姿 + 最短射线编号/角度/读数，
                # 与 reset 时的场景布局对照，可离线复算该读数是否有实物对应；
                # pitch/roll 用于排查车身倾斜导致雷达扫到自身/地面
                k = int(np.argmin(lid))
                dx, dy, dyaw = self._pose()
                o = self.self_node.getOrientation()  # 行优先
                pitch = math.asin(max(-1.0, min(1.0, -o[2])))
                roll = math.asin(max(-1.0, min(1.0, o[5])))
                # 点云里最近的点（雷达系坐标）：直接看幻影点落在哪个部件/方向/高度
                try:
                    pc = self.lidar.getPointCloud()
                    if pc:
                        pmin = min(pc, key=lambda p: p.x * p.x + p.y * p.y)
                        pdesc = (f"点云最近点=({pmin.x:.3f},{pmin.y:.3f},"
                                 f"{pmin.z:.3f}) 共{len(pc)}点")
                        pts = " ".join(f"({p.x:.2f},{p.y:.2f},{p.z:.2f})" for p in pc)
                        pdesc += f"\n[env_server] DEBUG 点云全量: {pts}"
                    else:
                        pdesc = "点云为空"
                except Exception as e:
                    pdesc = f"点云获取失败:{e}"
                rays = " ".join(f"{v:.2f}" for v in lid)
                print(f"[env_server] DEBUG 碰撞: 位姿=({dx:.3f},{dy:.3f},"
                      f"yaw={dyaw:.3f}) 俯仰={math.degrees(pitch):.1f}° "
                      f"横滚={math.degrees(roll):.1f}° 最短射线=k{k} "
                      f"(机体系 {k * 360.0 / LIDAR_COUNT:.1f}°) 读数={m:.3f} "
                      f"{pdesc}\n[env_server] DEBUG 全帧: {rays}",
                      flush=True)
                break

        dt = executed * PHYSICS_DT_MS / 1000.0
        # 运动学模式：实测速度 = 指令速度（api.md §2.3 obs 的 v/w 字段）
        v_meas = v
        w_meas = w

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
        return obs

    def _log_episode(self, outcome, final_dist):
        self.log_writer.writerow([
            self.episode_id, self.step_id, outcome,
            round(self.min_lidar_ever, 4), round(final_dist, 4), self.seed_used])
        self.log_fp.flush()
        spike = (f"尖峰={self.spike_info}" if self.spike_info else "尖峰=无")
        print(f"[env_server] episode {self.episode_id} 结束: {outcome}, "
              f"步数={self.step_id}, 最终距离={final_dist:.2f} m "
              f"[DEBUG 最大姿态角={self.max_tilt:.1f}° {spike} "
              f"拦尖峰={self.spikes_suppressed}]", flush=True)

    def stop_motors(self):
        pass  # 运动学模式无电机；保留接口兼容调用方（handle_client/main 退出路径）


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
                if not isinstance(seed, int) or isinstance(seed, bool) or seed < -1:
                    await send_error(ws, "BAD_FIELD",
                                     "seed 必须是 int：-1 表示随机，非负整数表示复现该种子场景")
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
                    if "max_episode_time" not in override:
                        await send_error(ws, "BAD_FIELD",
                                         "config_override 不能为空字典，必须含 max_episode_time")
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
        # §2.7：内部异常尽力回 INTERNAL（若连接已断开则静默）
        if not ws.closed:
            try:
                await send_error(ws, "INTERNAL", f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

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
