#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2 真机低速移动测试（单终端，集成键盘遥控）。

用法：
    cd A2/003 && uv run python ../real_robot/movement_test.py

流程：
  1. 连 server，收 hello
  2. 发 reset
  3. 提示 record_goal：把车放到目标点，按回车
  4. 提示 drive_to_start：按 W/A/S/D 遥控车到起点，Q 结束遥控
  5. 程序自动发 human_confirm(drive_to_start)
  6. 收到初始 obs 后，发 v=0.1, w=0 走 3 秒
  7. 发 v=0, w=0 停 1 步
  8. 发 all_finish 收 bye
"""

import asyncio
import json
import sys
import termios
import tty

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


LINEAR_SPEED = 0.2   # m/s
ANGULAR_SPEED = 0.5  # rad/s
TELEOP_HZ = 20       # 遥控发布频率


async def recv(ws) -> dict:
    raw = await ws.recv()
    return json.loads(raw)


async def send(ws, msg: dict):
    await ws.send(json.dumps(msg, allow_nan=False))


def read_key():
    """读取单个按键（不回车）"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def async_input(prompt: str) -> str:
    """让 input() 不阻塞 asyncio 事件循环，保持 websocket 心跳。"""
    return await asyncio.get_event_loop().run_in_executor(None, input, prompt)


async def human_confirm(ws, action: str, detail: str):
    print(f"\n>>> [需要人工操作] {detail}", flush=True)
    await async_input("完成后按回车继续...")
    await send(ws, {"type": "human_confirm", "action": action})


async def teleop_drive_to_start(ws, action: str, detail: str):
    """通过 WebSocket 连续发送 teleop 命令，遥控车到起点。"""
    print(f"\n>>> [需要人工操作] {detail}", flush=True)
    print("遥控启动：按住 W/S 前进后退，A/D 左右转，空格停止，Q 结束遥控")
    print("命令会以 20Hz 持续发给 server，server 再转发给 /cmd_vel\n")

    teleop_state = {"v": 0.0, "w": 0.0, "running": True}
    last_key = ""

    async def sender_loop():
        while teleop_state["running"]:
            await send(ws, {
                "type": "teleop",
                "v": teleop_state["v"],
                "w": teleop_state["w"],
            })
            print(f"\r  teleop: v={teleop_state['v']:+.2f}  w={teleop_state['w']:+.2f}  |  last_key={last_key!r}  |  W/S/A/D/space/Q",
                  end="", flush=True)
            await asyncio.sleep(1.0 / TELEOP_HZ)
        # 结束前再发一次零速
        await send(ws, {"type": "teleop", "v": 0.0, "w": 0.0})

    def key_reader():
        nonlocal last_key
        while teleop_state["running"]:
            key = read_key()
            last_key = key
            if key == "w" or key == "\x1b[A":  # ↑
                teleop_state["v"] = LINEAR_SPEED
                teleop_state["w"] = 0.0
            elif key == "s" or key == "\x1b[B":  # ↓
                teleop_state["v"] = -LINEAR_SPEED
                teleop_state["w"] = 0.0
            elif key == "a" or key == "\x1b[D":  # ←
                teleop_state["v"] = 0.0
                teleop_state["w"] = ANGULAR_SPEED
            elif key == "d" or key == "\x1b[C":  # →
                teleop_state["v"] = 0.0
                teleop_state["w"] = -ANGULAR_SPEED
            elif key == " ":
                teleop_state["v"] = 0.0
                teleop_state["w"] = 0.0
            elif key == "q" or key == "Q":
                teleop_state["v"] = 0.0
                teleop_state["w"] = 0.0
                teleop_state["running"] = False
                return

    sender = asyncio.create_task(sender_loop())
    reader = asyncio.get_event_loop().run_in_executor(None, key_reader)
    await asyncio.gather(sender, reader)

    print("\n\n遥控结束，发送 human_confirm(drive_to_start)...")
    await send(ws, {"type": "human_confirm", "action": action})


async def recv_until_obs(ws):
    """处理 human 消息直到收到 obs。"""
    while True:
        msg = await recv(ws)
        mtype = msg.get("type")
        if mtype == "obs":
            return msg
        if mtype == "human":
            action = msg.get("action", "unknown")
            detail = msg.get("detail", "")
            if action == "drive_to_start":
                await teleop_drive_to_start(ws, action, detail)
            else:
                await human_confirm(ws, action, detail)
            continue
        raise RuntimeError(f"reset 后收到意外消息: {msg}")


async def main():
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://192.168.43.114:8765"
    print(f"连接 {uri} ...")

    async with connect(uri, ping_interval=None, ping_timeout=None) as ws:
        hello = await recv(ws)
        print(f"hello: {hello['env_name']} v{hello['protocol_version']}")
        cfg = hello["config"]
        dt = cfg["control_dt"]

        # reset
        await send(ws, {"type": "reset", "seed": -1, "config_override": None})

        # 初始 obs（自动处理 human 提示，包括 drive_to_start 遥控）
        obs = await recv_until_obs(ws)
        print(f"\n初始: step={obs['step_id']} t={obs['t']:.1f} "
              f"goal_dist={obs['goal']['dist']:.3f} "
              f"vel=({obs['vel']['v']:.3f},{obs['vel']['w']:.3f}) "
              f"lidar[0]={obs['lidar'][0]:.3f}")

        if obs["done"]:
            print(f"\n注意：初始 obs 就 done 了（{obs['flags']}），"
                  "说明目标点和起点太近或 odom 未更新。")
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
