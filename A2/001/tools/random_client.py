# -*- coding: utf-8 -*-
"""
random_client.py — A2 项目开发桩（api.md §8）
随机动作客户端：用于 001 侧独立验证 WebSocket 协议与仿真环境全流程。

用法：
    python tools/random_client.py                 # 随机种子跑 3 局
    python tools/random_client.py --episodes 5 --seed 42
"""

import argparse
import asyncio
import json
import random

try:  # websockets >= 14
    from websockets.asyncio.client import connect
except ImportError:  # websockets < 14
    from websockets import connect


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://localhost:8765")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=-1, help="-1 随机；非负整数复现场景")
    args = ap.parse_args()

    async with connect(args.uri, ping_interval=20, ping_timeout=60) as ws:
        # ---- hello 校验 ----
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", f"首条消息不是 hello: {hello}"
        cfg = hello["config"]
        print(f"[client] hello OK: 协议 {hello['protocol_version']}, "
              f"obs_dim={cfg['obs_dim']}, 雷达 {cfg['lidar_count']} 线 / "
              f"{cfg['lidar_max_range']} m, dt={cfg['control_dt']} s")

        for ep in range(args.episodes):
            # ---- reset ----
            await ws.send(json.dumps({
                "type": "reset", "seed": args.seed, "config_override": None}))
            steps = 0
            while True:
                obs = json.loads(await ws.recv())
                assert obs["type"] == "obs", f"期望 obs 收到: {obs}"
                assert len(obs["lidar"]) == cfg["lidar_count"], "雷达维数不对"

                if obs["step_id"] == 0:
                    # 雷达方向自检：第 0/16/32/48 条应对应 前/左/后/右
                    d = obs["lidar"]
                    print(f"[client] ep{obs['episode_id']} 初始雷达 "
                          f"前={d[0]:.2f} 左={d[16]:.2f} 后={d[32]:.2f} 右={d[48]:.2f} "
                          f"(请对照 Webots 画面目测检查)")

                if obs["done"]:
                    outcome = ("碰撞" if obs["flags"]["collision"] else
                               "到达" if obs["flags"]["goal_reached"] else "超时")
                    print(f"[client] episode {obs['episode_id']} 结束: {outcome}, "
                          f"{obs['step_id']} 步, t={obs['t']:.1f}s, "
                          f"最终距离={obs['goal']['dist']:.2f} m")
                    break

                # ---- 随机动作（回声 api.md §2.4 全部字段） ----
                await ws.send(json.dumps({
                    "type": "action",
                    "episode_id": obs["episode_id"],
                    "step_id": obs["step_id"],
                    "v": random.uniform(-cfg["v_max"], cfg["v_max"]),
                    "w": random.uniform(-cfg["w_max"], cfg["w_max"]),
                }))
                steps += 1

        # ---- all_finish ----
        await ws.send(json.dumps({
            "type": "all_finish", "reason": "interrupted",
            "total_episodes": args.episodes}))
        bye = json.loads(await ws.recv())
        assert bye["type"] == "bye", f"期望 bye 收到: {bye}"
        print("[client] bye 收到，全流程 OK")


if __name__ == "__main__":
    asyncio.run(main())
