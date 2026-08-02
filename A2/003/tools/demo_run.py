# -*- coding: utf-8 -*-
"""
demo_run.py — 003 离线联调（api.md §8.2 全流程验收）

在同一事件循环里启动 echo_server 桩（tools/echo_server.py，端口 8766），
再驱动 train.py 的主循环跑若干局，验证完整链路：
hello 校验 → reset/obs/action 循环 → buffer → SAC 更新 → 周期性评估
（确定性策略 + 固定种子）→ checkpoint 存盘 → all_finish/bye。

不连 Webots、不需要 009 交付（回退 model_stub 桩）。真机/仿真联调时改用
`python train.py --uri ws://127.0.0.1:8765` 即可，本脚本的产物全部写 logs/demo、checkpoints/demo。

用法：python tools/demo_run.py [--episodes 12]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import train
from tools import echo_server


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--eval-interval", type=int, default=6,
                    help="桩上评估间隔（应小于 --episodes）")
    ap.add_argument("--eval-episodes", type=int, default=2)
    ap.add_argument("--no-curriculum", action="store_true",
                    help="透传给 train.py：关闭课程学习（完整时限，便于尽快越过 warmup）")
    args = ap.parse_args()

    server = await echo_server.start(port=8766)
    try:
        argv = ["--uri", "ws://127.0.0.1:8766",
                "--episodes", str(args.episodes),
                "--eval-interval", str(args.eval_interval),
                "--eval-episodes", str(args.eval_episodes),
                "--log-dir", "logs/demo",
                "--save-dir", "checkpoints/demo"]
        if args.no_curriculum:
            argv.append("--no-curriculum")
        targs = train.parse_args(argv)
        await train.amain(targs)
        print("\n[demo_run] 离线联调完成：全流程 OK")
        print("  - episodes.csv / eval.csv / train.log → logs/demo/")
        print("  - checkpoints → checkpoints/demo/")
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
