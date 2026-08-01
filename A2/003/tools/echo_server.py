# -*- coding: utf-8 -*-
"""
echo_server.py — A2 项目开发桩（api.md §8.2）

003 离线用的假 env server：不依赖 Webots，模拟 api.md 协议主流程。
用于先把 SAC 主循环跑通（连接 → hello → reset → obs/action 循环 → all_finish/bye）。

模拟的动力学：
  - 4×4 m 场地围墙 + 一个固定障碍物；差速模型 pos/yaw 积分
  - lidar 为真实射线-围墙/障碍物求交（无噪声模拟，仅用于流程验证）
  - 目标点固定放起点前方 2.5 m；碰撞/到达/超时判定与 api.md 一致

用法：
    python tools/echo_server.py --port 8766
配合 train.py --uri ws://127.0.0.1:8766 使用。
"""

import argparse
import asyncio
import json
import math
import random

try:  # websockets >= 14
    from websockets.asyncio.server import serve
except ImportError:  # websockets < 14
    from websockets import serve

# 与 api.md 默认常量一致（桩不实现 hello 下发 config 之外的覆盖，除 max_episode_time）
DT = 0.1
MAX_RANGE = 3.5
V_MAX, W_MAX = 0.5, 1.5
ARENA_HALF = 2.0
ROBOT_RADIUS = 0.18
GOAL_TOL = 0.15
OBSTACLE = (1.2, 0.6, 0.25)  # (x, y, r)


def ray_hit(ox, oy, ang):
    """射线 (ox,oy) 沿 ang 方向，与 4 面墙 + 1 个圆障碍求交，返回最近距离（≤MAX_RANGE）。"""
    dx, dy = math.cos(ang), math.sin(ang)
    best = MAX_RANGE
    # 围墙（内表面 ±2.0）
    for x in (-ARENA_HALF, ARENA_HALF):
        if abs(dx) > 1e-9:
            t = (x - ox) / dx
            if t > 0 and abs(oy + t * dy) <= ARENA_HALF:
                best = min(best, t)
    for y in (-ARENA_HALF, ARENA_HALF):
        if abs(dy) > 1e-9:
            t = (y - oy) / dy
            if t > 0 and abs(ox + t * dx) <= ARENA_HALF:
                best = min(best, t)
    # 圆形障碍物
    cx, cy, cr = OBSTACLE
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - cr * cr
    disc = b * b - c
    if disc >= 0 and b > 0:  # 射线起点在圆外时 b>0 表示朝圆走
        t = b - math.sqrt(disc)
        if t > 0:
            best = min(best, t)
    return best


def make_obs(ep, step, pos, yaw, v, w, lidar, done_flags, done):
    dx, dy = 2.5 - pos[0], -pos[1]  # 目标固定在 (2.5, 0)
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - yaw
    while bearing > math.pi:
        bearing -= 2 * math.pi
    while bearing <= -math.pi:
        bearing += 2 * math.pi
    return {
        "type": "obs",
        "episode_id": ep,
        "step_id": step,
        "t": round(step * DT, 3),
        "lidar": [round(min(ray_hit(pos[0], pos[1], yaw + i * 2 * math.pi / 64),
                             MAX_RANGE), 4) for i in range(64)],
        "goal": {"dist": round(dist, 4), "bearing": round(bearing, 4)},
        "vel": {"v": round(v, 4), "w": round(w, 4)},
        "flags": done_flags,
        "done": done,
    }


async def handle_client(ws, path=None):
    cfg = {"lidar_count": 64, "lidar_max_range": MAX_RANGE, "obs_dim": 68,
           "act_dim": 2, "control_dt": DT, "max_episode_time": 30.0,
           "v_max": V_MAX, "w_max": W_MAX, "goal_tolerance": GOAL_TOL,
           "robot_radius": ROBOT_RADIUS, "arena_size": 4.0}
    await ws.send(json.dumps({
        "type": "hello", "protocol_version": "1.1",
        "env_name": "webots_diffbot_v1", "config": cfg}))
    print("[echo_server] hello 已发，等待 reset …")

    state = "WAIT_RESET"
    ep = 0
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        t = msg.get("type")

        if t == "reset" and state == "WAIT_RESET":
            ep += 1
            max_time = cfg["max_episode_time"]
            if msg.get("config_override"):
                max_time = msg["config_override"]["max_episode_time"]
            rng = random.Random(msg.get("seed") if msg.get("seed") != -1 else None)
            state = {
                "ep": ep, "step": 0, "max_steps": int(max_time / DT),
                "pos": [0.0, 0.0], "yaw": rng.uniform(-math.pi, math.pi),
                "v": 0.0, "w": 0.0, "rng": rng}
            await ws.send(json.dumps(make_obs(
                ep, 0, state["pos"], state["yaw"], 0.0, 0.0, None, None, False)))

        elif t == "action" and state and state["step"] < state["max_steps"]:
            v = max(-V_MAX, min(V_MAX, msg.get("v", 0.0)))
            w = max(-W_MAX, min(W_MAX, msg.get("w", 0.0)))
            st = state
            st["pos"][0] += v * math.cos(st["yaw"]) * DT
            st["pos"][1] += v * math.sin(st["yaw"]) * DT
            st["yaw"] += w * DT
            st["v"], st["w"] = v, w
            st["step"] += 1

            dist = math.hypot(2.5 - st["pos"][0], -st["pos"][1])
            flags = {"collision": False, "goal_reached": False, "timeout": False}
            if dist <= GOAL_TOL:
                flags["goal_reached"] = True
            elif ray_hit(st["pos"][0], st["pos"][1],
                         st["yaw"]) < ROBOT_RADIUS or \
                 math.hypot(st["pos"][0] - OBSTACLE[0],
                            st["pos"][1] - OBSTACLE[1]) < OBSTACLE[2] + 0.1:
                flags["collision"] = True
            elif st["step"] >= st["max_steps"]:
                flags["timeout"] = True
            done = any(flags.values())
            if done:
                state = "WAIT_RESET"
            await ws.send(json.dumps(make_obs(
                st["ep"], st["step"], st["pos"], st["yaw"], v, w, None,
                flags, done)))

        elif t == "all_finish" and state == "WAIT_RESET":
            print(f"[echo_server] all_finish(reason={msg.get('reason')}, "
                  f"total={msg.get('total_episodes')}) 收到")
            await ws.send(json.dumps({"type": "bye",
                                      "reason": "all_finish received"}))
            return

        else:
            await ws.send(json.dumps(
                {"type": "error", "code": "WRONG_STATE",
                 "detail": f"state={state} 收到非法消息 {t}"}))
            return


async def start(port: int = 8766):
    server = await serve(handle_client, "127.0.0.1", port)
    print(f"[echo_server] 桩 server 已启动: ws://127.0.0.1:{port}")
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    asyncio.run(start(args.port))  # 本函数内部自持事件循环


if __name__ == "__main__":
    main()
