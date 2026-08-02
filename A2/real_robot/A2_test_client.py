#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 真机 server 简易测试 client。

用法：
    python3 A2_test_client.py ws://192.168.43.114:8765

流程：
    1. 连上后打印 hello
    2. 发 reset
    3. 收到 human(record_goal) 后暂停，等你按回车，然后发 human_confirm
    4. 收到 human(record_start) 后暂停，等你按回车，然后发 human_confirm
    5. 进入自动循环：收到 obs → 打印 → 发随机/零 action
    6. 跑 10 步后发 all_finish
"""

import asyncio
import json
import sys

import numpy as np

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


async def prompt(msg: str):
    await asyncio.get_event_loop().run_in_executor(
        None, input, f"\n>>> {msg}\n按回车继续..."
    )


async def recv(ws) -> dict:
    raw = await ws.recv()
    msg = json.loads(raw)
    print("<--", json.dumps(msg, ensure_ascii=False, indent=2)[:500])
    return msg


async def send(ws, msg: dict):
    raw = json.dumps(msg, allow_nan=False)
    print("-->", raw)
    await ws.send(raw)


async def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"
    print(f"连接 {uri} ...")

    async with connect(uri, ping_interval=None, ping_timeout=None) as ws:
        hello = await recv(ws)
        cfg = hello["config"]

        # reset
        await send(ws, {"type": "reset", "seed": -1, "config_override": None})

        # record goal
        msg = await recv(ws)
        assert msg["type"] == "human" and msg["action"] == "record_goal"
        await prompt("现在把车放到目标点")
        await send(ws, {"type": "human_confirm", "action": "record_goal"})

        # record start
        msg = await recv(ws)
        assert msg["type"] == "human" and msg["action"] == "record_start"
        await prompt("现在把车放到起点")
        await send(ws, {"type": "human_confirm", "action": "record_start"})

        # 初始 obs
        obs = await recv(ws)
        print("初始 obs lidar min:", min(obs["lidar"]))

        # 自动跑 10 步（零速，仅测试协议）
        for _ in range(10):
            v, w = 0.0, 0.0  # 安全：零速
            await send(ws, {
                "type": "action",
                "episode_id": obs["episode_id"],
                "step_id": obs["step_id"],
                "v": v,
                "w": w,
            })
            obs = await recv(ws)
            if obs["done"]:
                print("episode 结束:", obs["flags"])
                break

        await send(ws, {"type": "all_finish", "reason": "interrupted", "total_episodes": 1})
        bye = await recv(ws)
        print("bye:", bye)


if __name__ == "__main__":
    asyncio.run(main())
