#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 真机低速移动测试（交互式，按回车确认人工操作完成）。

用法（必须在自己有键盘输入的 WSL 终端里跑）：
    cd A2/003 && uv run python ../real_robot/movement_test.py

流程：
  1. 连 server，收 hello
  2. 发 reset
  3. 提示：把车放到目标点，按回车 → human_confirm(record_goal)
  4. 提示：把车放到起点（离目标点 ≥0.3m），按回车 → human_confirm(record_start)
  5. 发 v=0.1, w=0 走 3 秒（30 步）
  6. 发 v=0, w=0 停 1 步
  7. 打印初始/最终距离、odom 速度、lidar[0]
  8. 发 all_finish 收 bye
"""

import asyncio
import json
import sys

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


async def recv(ws) -> dict:
    raw = await ws.recv()
    return json.loads(raw)


async def send(ws, msg: dict):
    await ws.send(json.dumps(msg, allow_nan=False))


def wait_for_human(detail: str):
    print(f"\n>>> [需要人工操作] {detail}", flush=True)
    input("完成后按回车继续...")


async def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://192.168.43.114:8765"
    print(f"连接 {uri} ...")

    async with connect(uri) as ws:
        hello = await recv(ws)
        print(f"hello: {hello['env_name']} v{hello['protocol_version']}")
        cfg = hello["config"]
        dt = cfg["control_dt"]

        # reset
        await send(ws, {"type": "reset", "seed": -1, "config_override": None})

        # record goal
        msg = await recv(ws)
        assert msg["type"] == "human" and msg["action"] == "record_goal"
        wait_for_human(msg["detail"])
        await send(ws, {"type": "human_confirm", "action": "record_goal"})

        # record start
        msg = await recv(ws)
        assert msg["type"] == "human" and msg["action"] == "record_start"
        wait_for_human(msg["detail"])
        await send(ws, {"type": "human_confirm", "action": "record_start"})

        # 初始 obs
        obs = await recv(ws)
        print(f"\n初始: step={obs['step_id']} t={obs['t']:.1f} "
              f"goal_dist={obs['goal']['dist']:.3f} "
              f"vel=({obs['vel']['v']:.3f},{obs['vel']['w']:.3f}) "
              f"lidar[0]={obs['lidar'][0]:.3f}")

        if obs["done"]:
            print(f"\n注意：初始 obs 就 done 了（{obs['flags']}），"
                  "说明目标点和起点太近，请重跑并把两点分开。")
            await send(ws, {"type": "all_finish", "reason": "interrupted",
                            "total_episodes": 1})
            await recv(ws)
            return

        # 走 3 秒
        n_steps = int(3.0 / dt)
        for i in range(n_steps):
            await send(ws, {
                "type": "action",
                "episode_id": obs["episode_id"],
                "step_id": obs["step_id"],
                "v": 0.1,
                "w": 0.0,
            })
            obs = await recv(ws)
            print(f"step={obs['step_id']:3d} t={obs['t']:.1f} "
                  f"goal_dist={obs['goal']['dist']:.3f} "
                  f"vel=({obs['vel']['v']:.3f},{obs['vel']['w']:.3f}) "
                  f"lidar[0]={obs['lidar'][0]:.3f}")
            if obs["done"]:
                print(f"\n第 {i+1} 步 episode 结束: {obs['flags']}")
                break

        # 停车（如果 episode 还没结束）
        if not obs["done"]:
            await send(ws, {
                "type": "action",
                "episode_id": obs["episode_id"],
                "step_id": obs["step_id"],
                "v": 0.0,
                "w": 0.0,
            })
            obs = await recv(ws)
            print(f"\n最终: step={obs['step_id']} t={obs['t']:.1f} "
                  f"goal_dist={obs['goal']['dist']:.3f} "
                  f"vel=({obs['vel']['v']:.3f},{obs['vel']['w']:.3f}) "
                  f"lidar[0]={obs['lidar'][0]:.3f}")

        await send(ws, {"type": "all_finish", "reason": "interrupted",
                        "total_episodes": 1})
        bye = await recv(ws)
        print(f"bye: {bye}")


if __name__ == "__main__":
    asyncio.run(main())
