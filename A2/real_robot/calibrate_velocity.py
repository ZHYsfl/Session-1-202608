#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真机速度标定脚本。

用法：
    # 1. 先在 Pi 上启动 server（relative 模式，目标点在前方 2m）
    # 2. 在 WSL 跑：
    cd A2/003 && uv run python ../real_robot/calibrate_velocity.py

流程：
  1. 把车放在开阔地面，车头朝向开阔方向。
  2. 脚本对一系列速度指令 v 分别跑：
       - 记录当前 odom 为 start
       - 目标点自动设为 start 前方 2m（server relative 模式）
       - 以固定 v 直线前进 N 秒（默认 30 步 = 3 秒）
       - 读取终点 odom，用 goal.dist 的减少量计算实际走过的距离
  3. 输出“命令速度 -> 实际速度”的标定表。

注意：
  - 必须搭配 server 的 `--goal-mode relative` 使用，否则目标点与起点重合，
    episode 会立即 goal_reached，无法标定。
  - 确保正前方 2~3 米内没有障碍物，否则 episode 会因 collision 提前终止。
  - 车头要大致正朝开阔方向，w=0 时尽量走直线。
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Optional

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


URI = "ws://192.168.43.114:8765"
DEFAULT_SPEEDS = [0.05, 0.1, 0.15, 0.2]  # 默认不测太高，避免场地不够
DEFAULT_STEPS = 15          # 1.5 秒，场地不够可改 10


async def recv(ws) -> dict:
    raw = await ws.recv()
    return json.loads(raw)


async def send(ws, msg: dict):
    await ws.send(json.dumps(msg, allow_nan=False))


async def async_input(prompt: str) -> str:
    return await asyncio.get_event_loop().run_in_executor(None, input, prompt)


async def recv_until_obs(ws, *, auto_actions: Optional[set] = None,
                         prompt_prefix: str = "",
                         auto_confirm: bool = False) -> dict:
    """收消息直到拿到 obs；human 消息按规则处理。"""
    auto_actions = auto_actions or set()
    if auto_confirm:
        auto_actions = set(auto_actions)
        auto_actions.add("record_start")
        auto_actions.add("record_goal")
        auto_actions.add("drive_to_start")
    while True:
        msg = await recv(ws)
        mtype = msg.get("type")
        if mtype == "obs":
            return msg
        if mtype == "human":
            action = msg.get("action", "unknown")
            detail = msg.get("detail", "")
            if action in auto_actions:
                print(f"{prompt_prefix}自动确认 human({action}): {detail}")
                await send(ws, {"type": "human_confirm", "action": action})
            else:
                print(f"\n{prompt_prefix}[需要人工操作] {detail}")
                await async_input("完成后按回车继续...")
                await send(ws, {"type": "human_confirm", "action": action})
            continue
        if mtype == "error":
            raise RuntimeError(f"server error: {msg}")
        raise RuntimeError(f"收到意外消息: {msg}")


async def calibrate_speed(ws, speed: float, cfg: dict, steps: int,
                          auto_confirm: bool = False) -> dict:
    """标定一个速度。返回 {"cmd": speed, "dist": m, "actual": m/s, "outcome": str}。"""
    dt = cfg["control_dt"]

    await send(ws, {"type": "reset", "seed": -1, "config_override": None})
    # 先摆车，车头朝前
    obs = await recv_until_obs(ws, auto_actions={"drive_to_start"},
                               prompt_prefix=f"v={speed:.2f}: ",
                               auto_confirm=auto_confirm)
    if obs.get("type") != "obs":
        raise RuntimeError(f"reset 后未收到 obs: {obs}")

    # 确认初始距离
    start_dist = obs["goal"]["dist"]
    print(f"v={speed:.2f}: 起始距离 {start_dist:.3f}m，开始跑 {steps * dt:.1f}s ...")

    # 跑直线
    last_obs = obs
    t0 = time.monotonic()
    for i in range(steps):
        await send(ws, {
            "type": "action",
            "episode_id": last_obs["episode_id"],
            "step_id": last_obs["step_id"],
            "v": float(speed),
            "w": 0.0,
        })
        last_obs = await recv(ws)
        if last_obs.get("type") != "obs":
            raise RuntimeError(f"action 后未收到 obs: {last_obs}")
        if last_obs["done"]:
            break
    wall_elapsed = time.monotonic() - t0

    end_dist = last_obs["goal"]["dist"]
    sim_elapsed = last_obs["t"]
    outcome = next((k for k, v in last_obs["flags"].items() if v), "running")
    # 相对目标点模式下：车朝 goal 前进，goal.dist 减小，因此移动距离 = start_dist - end_dist
    dist_traveled = start_dist - end_dist
    actual_sim = dist_traveled / max(sim_elapsed, 1e-6)
    actual_wall = dist_traveled / max(wall_elapsed, 1e-6)

    direction = "前进" if dist_traveled > 0 else "后退"
    print(f"v={speed:.2f}: 结束距离 {end_dist:.3f}m | sim 耗时 {sim_elapsed:.1f}s | "
          f"wall 耗时 {wall_elapsed:.1f}s | 移动 {dist_traveled:.3f}m ({direction}) | "
          f"实际速度(sim) {actual_sim:.3f}m/s | 实际速度(wall) {actual_wall:.3f}m/s | "
          f"终止原因 {outcome}")
    if dist_traveled < 0:
        print("  提示：移动距离为负，说明本轮车朝目标点反方向走了。"
              "请把车头（正方向）朝向目标点/开阔方向再跑。")
    return {"cmd": speed, "dist": dist_traveled,
            "actual_sim": actual_sim, "actual_wall": actual_wall,
            "outcome": outcome}


async def main():
    ap = argparse.ArgumentParser(description="真机速度标定")
    ap.add_argument("--uri", default=URI)
    ap.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS,
                    help="要标定的命令速度列表（m/s）")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help="每个速度跑多少步（每步 0.1s）")
    ap.add_argument("--auto-confirm", action="store_true",
                    help="自动确认所有 human 提示（标定时不需要人工按回车）")
    args = ap.parse_args()

    print("速度列表:", args.speeds)
    print(f"每速度跑 {args.steps * 0.1:.1f} 秒")
    if args.auto_confirm:
        print("自动确认模式：所有 human 提示会自动确认")

    results = []
    for speed in args.speeds:
        print(f"\n连接 {args.uri} ...")
        async with connect(args.uri, ping_interval=None, ping_timeout=None) as ws:
            hello = await recv(ws)
            if hello.get("type") != "hello":
                raise RuntimeError(f"首条消息不是 hello: {hello}")
            print(f"hello: {hello['env_name']} v{hello['protocol_version']}")
            cfg = hello["config"]
            r = await calibrate_speed(ws, speed, cfg, args.steps,
                                      auto_confirm=args.auto_confirm)
            results.append(r)

    # 标定完成后礼貌关闭 server
    print("\n发送 all_finish 关闭 server ...")
    try:
        async with connect(args.uri, ping_interval=None, ping_timeout=None) as ws:
            await recv(ws)
            await send(ws, {"type": "all_finish", "reason": "interrupted",
                            "total_episodes": len(results)})
            try:
                await asyncio.wait_for(recv(ws), timeout=5.0)
            except asyncio.TimeoutError:
                pass
    except Exception as e:
        print(f"关闭 server 时出错（可忽略）: {e}")

    print("\n========== 标定结果 ==========")
    print("  命令速度   移动距离   实际速度(sim)  实际速度(wall)  比例(wall/cmd)")
    for r in results:
        ratio = r["actual_wall"] / r["cmd"] if r["cmd"] else 0.0
        print(f"  {r['cmd']:.2f} m/s  {r['dist']:.3f} m   {r['actual_sim']:.3f} m/s   "
              f"{r['actual_wall']:.3f} m/s   {ratio:.2f}x")

    # 建议统一缩放
    valid = [r for r in results
             if r["cmd"] and r["outcome"] != "goal_reached" and r["dist"] > 0]
    if valid:
        ratios = [r["actual_wall"] / r["cmd"] for r in valid]
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n平均比例(wall): {avg_ratio:.2f}x")
        print(f"建议：把 server 命令再除以 {avg_ratio:.2f}，" ""
              f"使 0.1m/s 指令 ≈ 0.1m/s 实际")
    else:
        print("\n有效数据不足（多轮都因 goal_reached 提前终止，或移动方向为负），" ""
              "请把车正向朝向目标点再跑。")


if __name__ == "__main__":
    asyncio.run(main())
