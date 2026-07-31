"""用 Pygame 观察手动、随机或强化学习策略控制 Cart-Pole。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from common import DEFAULT_CONFIG, load_config
from environment import CartPoleEnv
from evaluate import load_policy
from renderer import CartPoleRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 B3 Cart-Pole 可视化")
    parser.add_argument(
        "--algorithm",
        choices=("dqn", "q_learning", "manual", "random"),
        default="dqn",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    environment = CartPoleEnv(config, args.seed)
    renderer = CartPoleRenderer(config)
    rng = np.random.default_rng(args.seed)
    policy = None
    if args.algorithm in {"dqn", "q_learning"}:
        policy, metadata = load_policy(
            args.algorithm,
            config,
            args.checkpoint,
            args.device,
        )
        print(f"已加载 {args.algorithm} 模型：{metadata}")

    episode = 1
    state = environment.reset(args.seed + episode)
    episode_return = 0.0
    action = 0
    paused = False
    status = "RUNNING"
    terminal_frames = 0
    running = True

    try:
        while running:
            events = renderer.poll_events()
            if events.quit:
                break
            if events.pause_toggle:
                paused = not paused
            if events.reset:
                episode += 1
                state = environment.reset(args.seed + episode)
                episode_return = 0.0
                terminal_frames = 0
                status = "RUNNING"

            if terminal_frames > 0:
                terminal_frames -= 1
                if terminal_frames == 0:
                    episode += 1
                    state = environment.reset(args.seed + episode)
                    episode_return = 0.0
                    status = "RUNNING"
            elif not paused:
                if args.algorithm == "manual":
                    if events.manual_action is not None:
                        action = events.manual_action
                elif args.algorithm == "random":
                    action = int(rng.integers(0, 2))
                else:
                    assert policy is not None
                    action = policy(state)

                state, reward, terminated, truncated, _ = environment.step(action)
                episode_return += reward
                if terminated or truncated:
                    status = "SUCCESS" if truncated else "FAILED"
                    # 保留约一秒终局画面，便于观察成功或失败原因。
                    terminal_frames = int(config["render"]["fps"])

            renderer.render(
                state=state,
                algorithm=args.algorithm,
                episode=episode,
                step=environment.step_count,
                episode_return=episode_return,
                action=action,
                status=status,
                paused=paused,
            )
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
