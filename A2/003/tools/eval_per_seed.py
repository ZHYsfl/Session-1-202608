# -*- coding: utf-8 -*-
"""
eval_per_seed.py — 逐种子诊断：对固定评估种子逐局打印结果（只读，不训练）。

用途：eval 成功率卡住时（如连续 80%），定位是哪几个种子、以什么方式失败：
  timeout    → 局部极小/绕不出（导航能力问题）
  collision  → 风险校准问题（危险罚/动作平滑）
评估种子与 train.py 评估完全一致（EVAL_SEED_BASE+i，确定性策略），
但**绝不**把这些种子的数据用于训练——那是评估泄漏。

用法（001 server 已启动且无其他 client 占用时）：
    python tools/eval_per_seed.py --ckpt checkpoints/run7/ckpt_best.pt
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train as T  # 复用 run_episode / recv_msg / validate_hello / connect
from config import EVAL_SEED_BASE, PING_INTERVAL, PING_TIMEOUT, WS_URI
from models import get_model_module
from sac import SACAgent


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint 路径")
    ap.add_argument("--uri", default=WS_URI)
    ap.add_argument("--seed-base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--n", type=int, default=10, help="评估局数（与 §6.7 一致）")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_mod = get_model_module()
    models = model_mod.build_models(args.device)
    agent = SACAgent(args.device, models)
    meta = model_mod.load_checkpoint(args.ckpt, models,
                                     agent.optimizers_dict(),
                                     map_location=args.device)
    print(f"checkpoint: {args.ckpt}  meta={meta}", flush=True)

    async with T.connect(args.uri, ping_interval=PING_INTERVAL,
                         ping_timeout=PING_TIMEOUT) as ws:
        hello = await T.recv_msg(ws)
        cfg = T.validate_hello(hello)
        print(f"hello OK: {hello.get('env_name')} "
              f"v{hello.get('protocol_version')}", flush=True)
        rows = []
        for i in range(args.n):
            seed = args.seed_base + i
            r = await T.run_episode(ws, cfg, agent, None, args,
                                    seed=seed, override=None,
                                    deterministic=True, train=False)
            rows.append((seed, r["outcome"], r["steps"], r["return_"]))
            print(f"seed={seed}: {r['outcome']:<12s} steps={r['steps']:>4d} "
                  f"return={r['return_']:>9.2f}", flush=True)

    succ = sum(1 for row in rows if row[1] == "goal_reached")
    print(f"\n合计 {succ}/{len(rows)} 成功")
    fails = [row for row in rows if row[1] != "goal_reached"]
    if fails:
        print("失败种子:", ", ".join(f"{s}({o})" for s, o, _, _ in fails))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
