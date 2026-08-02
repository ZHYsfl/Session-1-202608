# -*- coding: utf-8 -*-
"""diag_goal.py — 诊断 seed=2008 直线到点过程，逐 10 步打印轨迹。"""
import asyncio
import json

try:
    from websockets.asyncio.client import connect
except ImportError:
    from websockets import connect


async def main():
    async with connect("ws://localhost:8765", ping_interval=20, ping_timeout=60) as ws:
        hello = json.loads(await ws.recv())
        print("hello", hello["protocol_version"])
        await ws.send(json.dumps(
            {"type": "reset", "seed": 2008, "config_override": None}))
        while True:
            obs = json.loads(await ws.recv())
            lidar = obs["lidar"]
            if obs["step_id"] % 10 == 0 or obs["done"]:
                print(f"s={obs['step_id']:3d} dist={obs['goal']['dist']:.3f} "
                      f"bearing={obs['goal']['bearing']:+.3f} "
                      f"vel=({obs['vel']['v']:+.2f},{obs['vel']['w']:+.2f}) "
                      f"minL={min(lidar):.2f}@i{lidar.index(min(lidar)):02d}")
            if obs["done"]:
                print("结局:", obs["flags"], f"最终 dist={obs['goal']['dist']:.3f}")
                break
            b = obs["goal"]["bearing"]
            w = max(-1.5, min(1.5, 2.5 * b))
            v = 0.4 if abs(b) < 0.3 else 0.05
            await ws.send(json.dumps({
                "type": "action", "episode_id": obs["episode_id"],
                "step_id": obs["step_id"], "v": v, "w": w}))


if __name__ == "__main__":
    asyncio.run(main())
