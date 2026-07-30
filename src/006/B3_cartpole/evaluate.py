"""在固定留出种子上评估训练后的 Cart-Pole 策略。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from common import DEFAULT_CONFIG, load_config, resolve_project_path, select_device
from dqn import load_dqn
from environment import CartPoleEnv
from q_learning import load_q_table

Policy = Callable[[np.ndarray], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 B3 Cart-Pole 模型")
    parser.add_argument(
        "--algorithm",
        choices=("dqn", "q_learning"),
        default="dqn",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def load_policy(
    algorithm: str,
    config: dict,
    checkpoint: Path | None,
    device_name: str,
) -> tuple[Policy, dict]:
    """加载策略，并返回只负责状态到动作映射的决策函数。"""

    if algorithm == "dqn":
        path = checkpoint or resolve_project_path(config["paths"]["dqn_checkpoint"])
        if not path.exists():
            raise FileNotFoundError(f"找不到 DQN 模型：{path}，请先训练")
        agent, metadata = load_dqn(path, config, select_device(device_name))
        return lambda state: agent.act(state, epsilon=0.0), metadata

    path = checkpoint or resolve_project_path(config["paths"]["q_learning_checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"找不到 Q-Learning 模型：{path}，请先训练")
    agent, metadata = load_q_table(path, config)
    return (
        lambda state: agent.act(state, epsilon=0.0, deterministic=True),
        metadata,
    )


def evaluate_policy(
    policy: Policy,
    config: dict,
    episodes: int,
    seed: int,
) -> dict[str, float | int]:
    """使用不与训练重合的连续随机种子计算成功率等指标。"""

    environment = CartPoleEnv(config, seed)
    steps: list[int] = []
    returns: list[float] = []
    successes = 0
    for episode in range(episodes):
        state = environment.reset(seed + episode)
        episode_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(state)
            state, reward, terminated, truncated, _ = environment.step(action)
            episode_return += reward
        steps.append(environment.step_count)
        returns.append(episode_return)
        successes += int(truncated)

    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "mean_return": float(np.mean(returns)),
        "min_steps": min(steps),
        "max_steps": max(steps),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    episodes = (
        int(config["evaluation"]["episodes"])
        if args.episodes is None
        else int(args.episodes)
    )
    seed = int(config["evaluation"]["seed"]) if args.seed is None else int(args.seed)
    if episodes <= 0:
        raise ValueError("--episodes 必须是正整数")
    policy, metadata = load_policy(
        args.algorithm,
        config,
        args.checkpoint,
        args.device,
    )
    result = evaluate_policy(policy, config, episodes, seed)
    print(f"模型信息：{json.dumps(metadata, ensure_ascii=False)}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
