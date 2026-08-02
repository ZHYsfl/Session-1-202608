# -*- coding: utf-8 -*-
"""probe_motion.py — 开环运动探针：直接测量轮速符号约定。

Phase 1: 15 步 (v=+0.3, w=0)   纯平移：正确 => 沿 (cos yaw, sin yaw) 前进，dist 变化视朝向而定
Phase 2: 10 步 (v=0, w=+1.0)   纯旋转：正确 => yaw 每步 +0.1 rad（w>0 = 逆时针）

服务器 [dbg] 打印逐步位姿；本端打印 obs 概要。看 yaw 与位移方向的夹角即可判定 v 符号，
看 phase2 yaw 增减即可判定 w 符号。
"""
import asyncio
import json
import math

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect

PHASES = [(12, 0.3, 0.0), (8, 0.0, 0.0), (12, 0.0, 1.0)]


async def main():
    async with connect("ws://localhost:8765", ping_interval=20, ping_timeout=60) as ws:
        hello = json.loads(await ws.recv())
        print("hello", hello["protocol_version"])
        await ws.send(json.dumps({"type": "reset", "seed": 2008, "config_override": None}))

        plan = [(v, w) for n, v, w in PHASES for _ in range(n)]
        i = 0
        prev_bearing = None
        while True:
            obs = json.loads(await ws.recv())
            g = obs["goal"]
            db = "" if prev_bearing is None else f" d_bearing={wrap(g['bearing']-prev_bearing):+.3f}"
            prev_bearing = g["bearing"]
            print(f"s={obs['step_id']:3d} dist={g['dist']:.3f} bearing={g['bearing']:+.3f} "
                  f"vel=({obs['vel']['v']:+.2f},{obs['vel']['w']:+.2f}) "
                  f"minL={min(obs['lidar']):.2f}{db}")
            if obs["done"]:
                print("结局:", obs["flags"])
                break
            if i >= len(plan):
                break
            v, w = plan[i]
            i += 1
            await ws.send(json.dumps({
                "type": "action", "episode_id": obs["episode_id"],
                "step_id": obs["step_id"], "v": v, "w": w}))

        await ws.send(json.dumps({"type": "all_finish", "reason": "converged",
                                  "total_episodes": 1}))
        try:
            await ws.send(json.dumps({"type": "bye"}))
        except Exception:
            pass


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


if __name__ == "__main__":
    asyncio.run(main())
