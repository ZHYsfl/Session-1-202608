# -*- coding: utf-8 -*-
"""
pysim_server.py — 纯 Python 仿真 env server（api.md v1.1 协议）

与 001 Webots env_server 的关系：
    Webots 版里机器人本来就是"理想运动学"（env_server.step_action：无轮、
    直接 v/w 积分），唯一的真实物理是 Webots Lidar 射线检测。本 server 用
    解析式射线检测（64 射线 vs 圆障碍 + 围墙）复刻同一环境，协议消息、
    采样约束、死局检查、碰撞判定（空间相干最小值 < COLLISION_DIST）、
    雷达遮挡扇区全部与 env_server.py 对齐。

用途：异步 RL 的采集/评估实例。单实例吞吐比 Webots 高一个数量级以上，
7 个实例 CPU 开销可忽略；Webots 只留 1 个 GUI 实例做演示与最终验收。

与 Webots 版的已知差异（可接受，最终策略在 Webots/真机上验收）：
    - 障碍物按外接圆处理（Webots 里是 Box，正对盒面时读数略小于圆）
    - 雷达无噪声/无运动尖峰（Webots 偶发幻影尖峰，两侧都已中位数滤波）
    - 忽略 z 轴/姿态（Webots 版也是平面运动学）

用法：
    python pysim_server.py --port 8766 [--host 127.0.0.1]
"""

import argparse
import asyncio
import json
import math

import numpy as np

try:  # websockets >= 14
    from websockets.asyncio.server import serve
except ImportError:  # websockets < 14
    from websockets import serve

# ================= 常量（与 env_server.py 完全一致） =================
PROTOCOL_VERSION = "1.1"
ENV_NAME = "pysim_diffbot_v1"

LIDAR_COUNT = 64
LIDAR_MAX_RANGE = 3.5
CONTROL_DT = 0.1
PHYSICS_DT = 0.01                  # env_server PHYSICS_DT_MS=10 → 10 子步
SUBSTEPS = int(CONTROL_DT / PHYSICS_DT)  # 10
MAX_EPISODE_TIME = 60.0
V_MAX = 0.5
W_MAX = 1.5
GOAL_TOLERANCE = 0.15
ROBOT_RADIUS = 0.18
COLLISION_DIST = 0.18
MIN_LINEAR_VEL = 0.0
LIDAR_OCCLUDED_BODY = (177.0, 277.0)
ARENA_SIZE = 4.0
OBS_DIM = 132
ACT_DIM = 2

START_GOAL_MIN_DIST = 2.0
SAMPLE_CLEARANCE = 0.4
OBSTACLE_GAP = 0.60
ROBOT_SAMPLE_LIM = 1.6
OBSTACLE_POS_LIM = 1.5
MIN_ACTIVE = 5
MAX_ACTIVE = 8
N_OBSTACLES = 8
OBSTACLE_RADII = [0.212, 0.25, 0.177, 0.247, 0.15, 0.20, 0.18, 0.12]
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
OVERRIDABLE_KEYS = {"max_episode_time"}

# 雷达射线机体角（第 0 条 = 车头正前，逆时针），与 env_server 标定后的 api 顺序一致
_RAY_BODY_ANG = np.arange(LIDAR_COUNT) * (2.0 * math.pi / LIDAR_COUNT)
_OCCL_I0 = int(LIDAR_OCCLUDED_BODY[0] / 360.0 * LIDAR_COUNT)   # 31
_OCCL_I1 = int(LIDAR_OCCLUDED_BODY[1] / 360.0 * LIDAR_COUNT)   # 49


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _median3(lid):
    """逐射线 3 邻域中位数滤波（环形），与 env_server._median3 同算法。"""
    a, b, c = np.roll(lid, 1), lid, np.roll(lid, -1)
    return a + b + c - np.minimum(np.minimum(a, b), c) - np.maximum(np.maximum(a, b), c)


class PySim:
    """平面差速小车 + 解析雷达，物理/判定逻辑对齐 env_server.EnvServer。"""

    def __init__(self):
        self.episode_id = 0
        self.step_id = 0
        self.max_episode_time = MAX_EPISODE_TIME
        self.seed_used = -1
        self.goal_xy = (0.0, 0.0)
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.placed = []           # [(x, y, r)] 本局激活障碍（圆）
        self.rng = np.random.default_rng()

    # ================= 解析雷达 =================
    def _lidar(self):
        """64 射线解析距离：vs 圆障碍 + 围墙，中位数滤波 + 遮挡扇区屏蔽。"""
        ang = self.yaw + _RAY_BODY_ANG
        dx, dy = np.cos(ang), np.sin(ang)
        best = np.full(LIDAR_COUNT, LIDAR_MAX_RANGE)

        # ---- 围墙（内表面 ±WALL_INNER；逻辑同 env_server._analytic_wall_ranges）----
        for d, p, other_d, other_p in ((dx, self.x, dy, self.y),
                                       (dy, self.y, dx, self.x)):
            for w in (WALL_INNER, -WALL_INNER):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(np.abs(d) > 1e-9, (w - p) / d, np.inf)
                hit = (t > 0.01) & (t < best) & (
                    np.abs(other_p + t * other_d) <= WALL_INNER)
                best = np.where(hit, t, best)

        # ---- 圆障碍（向量化解二次方程 |p + t·d − c|² = r² 的近正根）----
        for cx, cy, r in self.placed:
            ox, oy = self.x - cx, self.y - cy
            b = ox * dx + oy * dy
            c = ox * ox + oy * oy - r * r
            disc = b * b - c
            hit = disc > 0.0
            t = -b - np.sqrt(np.maximum(disc, 0.0))
            valid = hit & (t > 0.01) & (t < best)
            best = np.where(valid, t, best)

        lid = _median3(np.clip(best, 0.0, LIDAR_MAX_RANGE))
        lid[_OCCL_I0:_OCCL_I1] = LIDAR_MAX_RANGE   # 车壳遮挡扇区
        return lid

    # ================= reset（采样约束与死局检查对齐 env_server.reset） =================
    def reset(self, seed, config_override):
        if seed == -1:
            seed = int(np.random.SeedSequence().entropy) % (2 ** 31)
        self.seed_used = seed
        self.rng = np.random.default_rng(seed)
        self.max_episode_time = MAX_EPISODE_TIME
        if config_override is not None:
            self.max_episode_time = float(config_override["max_episode_time"])

        placed = []
        for _ in range(200):
            sx, sy = self.rng.uniform(-ROBOT_SAMPLE_LIM, ROBOT_SAMPLE_LIM, 2)
            gx, gy = self.rng.uniform(-ROBOT_SAMPLE_LIM, ROBOT_SAMPLE_LIM, 2)
            if math.hypot(gx - sx, gy - sy) < START_GOAL_MIN_DIST:
                continue
            self.goal_xy = (float(gx), float(gy))

            n_active = int(self.rng.integers(MIN_ACTIVE, MAX_ACTIVE + 1))
            order = self.rng.permutation(N_OBSTACLES)
            placed = []
            for rank, obs_i in enumerate(order):
                r = OBSTACLE_RADII[obs_i]
                if rank < n_active:
                    pos = None
                    for _ in range(100):
                        ox, oy = self.rng.uniform(-OBSTACLE_POS_LIM,
                                                  OBSTACLE_POS_LIM, 2)
                        if math.hypot(ox - sx, oy - sy) < r + SAMPLE_CLEARANCE:
                            continue
                        if math.hypot(ox - gx, oy - gy) < r + SAMPLE_CLEARANCE:
                            continue
                        if any(math.hypot(ox - px, oy - py) < r + pr + OBSTACLE_GAP
                               for px, py, pr in placed):
                            continue
                        pos = (float(ox), float(oy))
                        break
                    if pos is not None:
                        placed.append((pos[0], pos[1], r))

            if self._is_reachable(sx, sy, gx, gy, placed):
                break
            placed = []

        self.placed = placed
        self.x, self.y = float(sx), float(sy)
        self.yaw = float(self.rng.uniform(-math.pi, math.pi))
        self.episode_id += 1
        self.step_id = 0
        return self._build_obs(0.0, 0.0, False, False, False)

    @staticmethod
    def _is_reachable(sx, sy, gx, gy, placed,
                      res: float = 0.1, inflate: float = 0.30) -> bool:
        """栅格 BFS 死局检查（与 env_server._is_reachable 同算法：
        障碍圆与围墙同膨胀 inflate，4 连通）。"""
        half = ARENA_SIZE / 2.0
        n = int(ARENA_SIZE / res) + 1
        occ = np.zeros((n, n), dtype=bool)
        for px, py, r in placed:
            rr = (r + inflate) ** 2
            i0 = max(0, int((px - r - inflate + half) / res))
            i1 = min(n - 1, int((px + r + inflate + half) / res))
            j0 = max(0, int((py - r - inflate + half) / res))
            j1 = min(n - 1, int((py + r + inflate + half) / res))
            xs = np.arange(i0, i1 + 1) * res - half + res / 2
            ys = np.arange(j0, j1 + 1) * res - half + res / 2
            xx, yy = np.meshgrid(xs, ys, indexing="ij")
            occ[i0:i1 + 1, j0:j1 + 1] |= ((xx - px) ** 2 + (yy - py) ** 2) < rr
        m = max(1, int(round(inflate / res)))
        occ[:m, :] = True
        occ[-m:, :] = True
        occ[:, :m] = True
        occ[:, -m:] = True
        si, sj = int((sx + half) / res), int((sy + half) / res)
        gi, gj = int((gx + half) / res), int((gy + half) / res)
        if not (0 <= si < n and 0 <= sj < n and 0 <= gi < n and 0 <= gj < n):
            return False
        if occ[si, sj] or occ[gi, gj]:
            return False
        q = [(si, sj)]
        seen = {(si, sj)}
        while q:
            i, j = q.pop()
            if (i, j) == (gi, gj):
                return True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not occ[ni, nj] \
                        and (ni, nj) not in seen:
                    seen.add((ni, nj))
                    q.append((ni, nj))
        return False

    # ================= step（理想运动学，对齐 env_server.step_action） =================
    def step_action(self, v, w):
        v = float(np.clip(v, MIN_LINEAR_VEL, V_MAX))
        w = float(np.clip(w, -W_MAX, W_MAX))

        collision = False
        for _ in range(SUBSTEPS):
            self.yaw = wrap_pi(self.yaw + w * PHYSICS_DT)
            self.x += v * math.cos(self.yaw) * PHYSICS_DT
            self.y += v * math.sin(self.yaw) * PHYSICS_DT
            if float(np.min(self._lidar())) < COLLISION_DIST:
                collision = True
                break

        goal_reached = (not collision) and (
            math.hypot(self.goal_xy[0] - self.x,
                       self.goal_xy[1] - self.y) <= GOAL_TOLERANCE)
        self.step_id += 1
        timeout = (not collision and not goal_reached
                   and self.step_id * CONTROL_DT >= self.max_episode_time - 1e-9)
        return self._build_obs(v, w, collision, goal_reached, timeout)

    # ================= obs 组装（字段与 env_server._build_obs 一致） =================
    def _build_obs(self, v_meas, w_meas, collision, goal_reached, timeout):
        gx, gy = self.goal_xy
        dist = math.hypot(gx - self.x, gy - self.y)
        bearing = wrap_pi(math.atan2(gy - self.y, gx - self.x) - self.yaw)
        done = collision or goal_reached or timeout
        return {
            "type": "obs",
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "t": round(self.step_id * CONTROL_DT, 6),
            "lidar": [float(v) for v in self._lidar()],
            "goal": {"dist": round(dist, 4), "bearing": round(bearing, 4)},
            "vel": {"v": round(v_meas, 4), "w": round(w_meas, 4)},
            "flags": {
                "collision": collision,
                "goal_reached": goal_reached,
                "timeout": timeout,
            },
            "done": done,
        }


# ================= WebSocket 协议层（状态机与 env_server.handle_client 一致） =================

def hello_msg():
    return {
        "type": "hello",
        "protocol_version": PROTOCOL_VERSION,
        "env_name": ENV_NAME,
        "config": dict(CONFIG_KEYS),
    }


async def send_error(ws, code, detail):
    await ws.send(json.dumps({"type": "error", "code": code, "detail": detail}))
    print(f"[pysim] 发出 error {code}: {detail}", flush=True)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


async def handle_client(ws, env: PySim):
    """单客户端协议状态机：WAIT_RESET ⇄ RUNNING；连接断开后可接受下一个客户端
    （与 env_server 断线即关仿真不同——pysim 没有仿真可关，保活供复用）。"""
    peer = getattr(ws, "remote_address", "?")
    print(f"[pysim] 客户端已连接: {peer}", flush=True)
    await ws.send(json.dumps(hello_msg()))
    state = "WAIT_RESET"
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                await send_error(ws, "BAD_JSON", f"无法解析 JSON: {raw[:200]!r}")
                break
            if not isinstance(msg, dict) or "type" not in msg:
                await send_error(ws, "BAD_TYPE", "消息缺少 type 字段")
                break
            mtype = msg["type"]

            if mtype == "reset":
                if state != "WAIT_RESET":
                    await send_error(ws, "WRONG_STATE", "RUNNING 状态下收到 reset")
                    break
                seed = msg.get("seed")
                if not isinstance(seed, int) or isinstance(seed, bool) or seed < -1:
                    await send_error(ws, "BAD_FIELD", "seed 必须是 int")
                    break
                override = msg.get("config_override", "__missing__")
                if override == "__missing__":
                    await send_error(ws, "BAD_FIELD",
                                     "缺少 config_override 字段（不用时填 null）")
                    break
                if override is not None:
                    if not isinstance(override, dict):
                        await send_error(ws, "BAD_FIELD",
                                         "config_override 必须是 object 或 null")
                        break
                    bad = set(override) - OVERRIDABLE_KEYS
                    if bad:
                        await send_error(ws, "BAD_FIELD",
                                         f"config_override 不允许的键: {sorted(bad)}")
                        break
                    if "max_episode_time" not in override \
                            or not _is_number(override["max_episode_time"]) \
                            or override["max_episode_time"] <= 0:
                        await send_error(ws, "BAD_FIELD",
                                         "max_episode_time 必须是正数")
                        break
                obs = env.reset(seed, override)
                state = "RUNNING"
                await ws.send(json.dumps(obs))

            elif mtype == "action":
                if state != "RUNNING":
                    await send_error(ws, "WRONG_STATE", "等待 reset 时收到 action")
                    break
                if (msg.get("episode_id") != env.episode_id
                        or msg.get("step_id") != env.step_id):
                    await send_error(
                        ws, "BAD_FIELD",
                        f"action id 不匹配: 期望 ({env.episode_id}, {env.step_id})，"
                        f"收到 ({msg.get('episode_id')}, {msg.get('step_id')})")
                    break
                if not (_is_number(msg.get("v")) and _is_number(msg.get("w"))):
                    await send_error(ws, "BAD_FIELD", "v / w 必须是有限数值")
                    break
                obs = env.step_action(msg["v"], msg["w"])
                await ws.send(json.dumps(obs))
                if obs["done"]:
                    state = "WAIT_RESET"

            elif mtype == "all_finish":
                reason = msg.get("reason")
                total = msg.get("total_episodes")
                if reason not in ("converged", "interrupted", "error") \
                        or not isinstance(total, int):
                    await send_error(ws, "BAD_FIELD", "all_finish 字段非法")
                    break
                print(f"[pysim] all_finish: {reason}, 共 {total} episodes", flush=True)
                await ws.send(json.dumps(
                    {"type": "bye", "reason": "all_finish received"}))
                state = "WAIT_RESET"   # 复用：等下一个客户端，进程不退出
                break

            else:
                await send_error(ws, "BAD_TYPE", f"未知消息类型: {mtype!r}")
                break

    except Exception as exc:  # 连接断开等
        print(f"[pysim] 连接结束: {type(exc).__name__}: {exc}", flush=True)


async def _serve(host: str, port: int):
    env = PySim()

    async def handler(ws):
        await handle_client(ws, env)

    async with serve(handler, host, port,
                     ping_interval=20, ping_timeout=60, max_size=1 << 20):
        print(f"[pysim] WebSocket server 已启动: ws://{host}:{port}", flush=True)
        await asyncio.Future()


def main():
    ap = argparse.ArgumentParser(description="纯 Python 仿真 env server（api.md v1.1）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    asyncio.run(_serve(args.host, args.port))


if __name__ == "__main__":
    main()
