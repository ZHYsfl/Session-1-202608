"""训练 DQN 或表格 Q-Learning，并保存模型、指标和 TensorBoard 日志。"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    DEFAULT_CONFIG,
    load_config,
    resolve_project_path,
    seed_everything,
    select_device,
)
from dqn import DQNAgent, ReplayBuffer, epsilon_by_step, save_dqn
from environment import CartPoleEnv
from q_learning import QLearningAgent, epsilon_by_episode, save_q_table
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 B3 Cart-Pole 强化学习智能体")
    parser.add_argument(
        "--algorithm",
        choices=("dqn", "q_learning"),
        default="dqn",
        help="训练算法，默认 dqn",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="覆盖 config.yaml 中的训练回合数",
    )
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument(
        "--device",
        default="auto",
        help="DQN 设备：auto、cpu 或 cuda；Q-Learning 忽略此项",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML 配置文件路径",
    )
    return parser.parse_args()


def moving_mean(values: deque[int]) -> float:
    """计算最近若干回合的平均坚持步数。"""

    return float(np.mean(values)) if values else 0.0


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8 JSON 保存可复查的训练曲线。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_dqn(
    config: dict[str, Any],
    episodes: int,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    """使用经验回放和目标网络训练 DQN。"""

    device = select_device(device_name)
    environment = CartPoleEnv(config, seed)
    agent = DQNAgent(config, device, seed)
    replay = ReplayBuffer(int(config["dqn"]["replay_capacity"]), seed)
    log_directory = (
        resolve_project_path(config["paths"]["tensorboard_root"])
        / f"dqn_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
    )
    writer = SummaryWriter(log_directory)
    window_size = int(config["training"]["moving_average_window"])
    recent_steps: deque[int] = deque(maxlen=window_size)
    records: list[dict[str, float | int | bool]] = []
    global_step = 0
    losses: deque[float] = deque(maxlen=100)
    best_moving_steps = 0.0

    try:
        for episode in range(1, episodes + 1):
            state = environment.reset(seed + episode)
            episode_return = 0.0
            terminated = truncated = False
            epsilon = 1.0

            while not (terminated or truncated):
                epsilon = epsilon_by_step(config, global_step)
                action = agent.act(state, epsilon)
                next_state, reward, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                replay.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                global_step += 1

                dqn_config = config["dqn"]
                if (
                    global_step >= int(dqn_config["learning_starts"])
                    and global_step % int(dqn_config["train_frequency"]) == 0
                ):
                    loss = agent.optimize(replay)
                    if loss is not None:
                        losses.append(loss)
                        writer.add_scalar("train/td_loss", loss, global_step)
                if global_step % int(dqn_config["target_update_frequency"]) == 0:
                    agent.update_target()

            recent_steps.append(environment.step_count)
            average_steps = moving_mean(recent_steps)
            success = bool(truncated)
            best_moving_steps = max(best_moving_steps, average_steps)
            record = {
                "episode": episode,
                "steps": environment.step_count,
                "return": episode_return,
                "success": success,
                "epsilon": epsilon,
                "moving_mean_steps": average_steps,
                "mean_recent_loss": float(np.mean(losses)) if losses else 0.0,
                "global_step": global_step,
            }
            records.append(record)
            writer.add_scalar("episode/steps", environment.step_count, episode)
            writer.add_scalar("episode/return", episode_return, episode)
            writer.add_scalar("episode/success", float(success), episode)
            writer.add_scalar("episode/moving_mean_steps", average_steps, episode)
            writer.add_scalar("exploration/epsilon", epsilon, episode)

            if (
                episode == 1
                or episode % int(config["training"]["log_interval"]) == 0
                or episode == episodes
            ):
                print(
                    f"[DQN] 回合 {episode:4d}/{episodes} | "
                    f"步数 {environment.step_count:3d} | "
                    f"近 {len(recent_steps):3d} 回合均值 {average_steps:6.1f} | "
                    f"epsilon {epsilon:.3f} | "
                    f"loss {record['mean_recent_loss']:.4f}"
                )
    finally:
        writer.close()

    metadata = {
        "experiment_version": config["experiment"]["version"],
        "algorithm": "dqn",
        "seed": seed,
        "episodes": episodes,
        "global_step": global_step,
        "best_moving_mean_steps": best_moving_steps,
        "device": str(device),
    }
    save_dqn(
        resolve_project_path(config["paths"]["dqn_checkpoint"]),
        agent,
        metadata,
    )
    result = {"metadata": metadata, "episodes": records}
    write_metrics(
        resolve_project_path(config["paths"]["dqn_metrics"]),
        result,
    )
    print(f"DQN 训练完成，模型已保存；TensorBoard 日志：{log_directory}")
    return result


def train_q_learning(
    config: dict[str, Any],
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """在离散化状态空间上训练表格 Q-Learning。"""

    environment = CartPoleEnv(config, seed)
    agent = QLearningAgent(config, seed)
    log_directory = (
        resolve_project_path(config["paths"]["tensorboard_root"])
        / f"q_learning_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
    )
    writer = SummaryWriter(log_directory)
    window_size = int(config["training"]["moving_average_window"])
    recent_steps: deque[int] = deque(maxlen=window_size)
    records: list[dict[str, float | int | bool]] = []
    best_moving_steps = 0.0

    try:
        for episode in range(1, episodes + 1):
            state = environment.reset(seed + episode)
            episode_return = 0.0
            td_errors: list[float] = []
            terminated = truncated = False
            epsilon = epsilon_by_episode(config, episode - 1)

            while not (terminated or truncated):
                action = agent.act(state, epsilon)
                next_state, reward, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                td_errors.append(
                    abs(agent.update(state, action, reward, next_state, done))
                )
                state = next_state
                episode_return += reward

            recent_steps.append(environment.step_count)
            average_steps = moving_mean(recent_steps)
            success = bool(truncated)
            best_moving_steps = max(best_moving_steps, average_steps)
            record = {
                "episode": episode,
                "steps": environment.step_count,
                "return": episode_return,
                "success": success,
                "epsilon": epsilon,
                "moving_mean_steps": average_steps,
                "mean_absolute_td_error": float(np.mean(td_errors)),
            }
            records.append(record)
            writer.add_scalar("episode/steps", environment.step_count, episode)
            writer.add_scalar("episode/return", episode_return, episode)
            writer.add_scalar("episode/success", float(success), episode)
            writer.add_scalar("episode/moving_mean_steps", average_steps, episode)
            writer.add_scalar(
                "train/mean_absolute_td_error",
                record["mean_absolute_td_error"],
                episode,
            )
            writer.add_scalar("exploration/epsilon", epsilon, episode)

            if (
                episode == 1
                or episode % int(config["training"]["log_interval"]) == 0
                or episode == episodes
            ):
                print(
                    f"[Q-Learning] 回合 {episode:4d}/{episodes} | "
                    f"步数 {environment.step_count:3d} | "
                    f"近 {len(recent_steps):3d} 回合均值 {average_steps:6.1f} | "
                    f"epsilon {epsilon:.3f} | "
                    f"|TD| {record['mean_absolute_td_error']:.4f}"
                )
    finally:
        writer.close()

    metadata = {
        "experiment_version": config["experiment"]["version"],
        "algorithm": "q_learning",
        "seed": seed,
        "episodes": episodes,
        "best_moving_mean_steps": best_moving_steps,
        "q_table_shape": list(agent.q_table.shape),
    }
    save_q_table(
        resolve_project_path(config["paths"]["q_learning_checkpoint"]),
        agent,
        metadata,
    )
    result = {"metadata": metadata, "episodes": records}
    write_metrics(
        resolve_project_path(config["paths"]["q_learning_metrics"]),
        result,
    )
    print(f"Q-Learning 训练完成，Q 表已保存；TensorBoard 日志：{log_directory}")
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["training"]["seed"]) if args.seed is None else int(args.seed)
    default_episodes = int(config[args.algorithm]["episodes"])
    episodes = default_episodes if args.episodes is None else int(args.episodes)
    if episodes <= 0:
        raise ValueError("--episodes 必须是正整数")
    seed_everything(seed)

    print(
        f"算法：{args.algorithm} | 回合：{episodes} | 种子：{seed}\n"
        "状态空间：[小车位置, 小车速度, 杆角度, 杆角速度]\n"
        "动作空间：0=向左施力，1=向右施力"
    )
    if args.algorithm == "dqn":
        train_dqn(config, episodes, seed, args.device)
    else:
        train_q_learning(config, episodes, seed)


if __name__ == "__main__":
    main()
