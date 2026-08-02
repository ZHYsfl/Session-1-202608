# -*- coding: utf-8 -*-
"""
demo_play.py — 用训练好的权重跑避障演示（GUI 观看用）

连接 env_server（Webots 需以 GUI + ENV_SIM_MODE=realtime 启动），
用确定性策略（取 mean，无探索噪声）跑若干局，每局之间停顿几秒方便观看。

用法：
    python tools/demo_play.py --ckpt checkpoints/final_best.pt --episodes 5
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train as T  # 复用 run_episode / recv_msg / validate_hello / connect
from config import PING_INTERVAL, PING_TIMEOUT, WS_URI
from models import get_model_module
from sac import SACAgent


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final_best.pt")
    ap.add_argument("--uri", default=WS_URI)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=30000)
    ap.add_argument("--pause-s", type=float, default=3.0,
                    help="每局之间停顿秒数，方便观看")
    args = ap.parse_args()

    model_mod = get_model_module()
    models = model_mod.build_models("cpu")
    agent = SACAgent("cpu", models)
    meta = model_mod.load_checkpoint(args.ckpt, models,
                                     agent.optimizers_dict(),
                                     map_location="cpu")
    print(f"权重: {args.ckpt}  meta={meta}", flush=True)

    async with T.connect(args.uri, ping_interval=PING_INTERVAL,
                         ping_timeout=PING_TIMEOUT) as ws:
        hello = await T.recv_msg(ws)
        cfg = T.validate_hello(hello)
        print(f"hello OK: {hello.get('env_name')} "
              f"v{hello.get('protocol_version')}", flush=True)
        succ = 0
        for i in range(args.episodes):
            seed = args.seed_base + i
            r = await T.run_episode(ws, cfg, agent, None, args,
                                    seed=seed, override=None,
                                    deterministic=True, train=False)
            print(f"演示 {i + 1}/{args.episodes} seed={seed}: "
                  f"{r['outcome']:<12s} steps={r['steps']:>4d} "
                  f"return={r['return_']:>9.2f}", flush=True)
            succ += r["success"]
            if i < args.episodes - 1:
                await asyncio.sleep(args.pause_s)
        print(f"演示完成: {succ}/{args.episodes} 成功", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
