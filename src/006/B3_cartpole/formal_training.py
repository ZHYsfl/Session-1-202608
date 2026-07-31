"""正式实验使用的固定交互步数训练循环。"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dqn import DQNAgent, ReplayBuffer, save_dqn
from dqn import epsilon_by_step as dqn_epsilon
from environment import CartPoleEnv
from q_learning import (
    QLearningAgent,
    save_q_table,
)
from q_learning import (
    epsilon_by_step as q_epsilon,
)

Policy = Callable[[np.ndarray], int]


def evaluate_for_validation(
    policy: Policy,
    config: dict[str, Any],
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """只在验证初始状态上计算 checkpoint 选择指标。"""

    environment = CartPoleEnv(config, seed)
    lengths: list[int] = []
    successes = 0
    for episode in range(episodes):
        state = environment.reset(seed + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            state, _, terminated, truncated, _ = environment.step(policy(state))
        lengths.append(environment.step_count)
        successes += int(truncated)
    return {
        "mean_steps": float(np.mean(lengths)),
        "success_rate": successes / episodes,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """先写临时文件再替换，避免中断留下半截 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def checkpoint_metadata(
    config: dict[str, Any],
    *,
    algorithm: str,
    variant: str,
    seed: int,
    global_step: int,
    validation: dict[str, float] | None,
    checkpoint_kind: str,
    device: str,
) -> dict[str, Any]:
    """生成 best/final checkpoint 共用的可追溯元数据。"""

    return {
        "experiment_version": config["experiment"]["version"],
        "formal_config_sha256": config["formal"]["config_sha256"],
        "algorithm": algorithm,
        "variant": variant,
        "seed": seed,
        "global_step": global_step,
        "checkpoint_kind": checkpoint_kind,
        "validation": validation,
        "device": device,
    }


def train_dqn_fixed_steps(
    config: dict[str, Any],
    *,
    variant: str,
    label: str,
    seed: int,
    environment_steps: int,
    validation_interval: int,
    validation_episodes: int,
    validation_seed: int,
    output_directory: Path,
    device: Any,
) -> dict[str, Any]:
    """按统一环境交互步数训练一个 DQN 变体。"""

    environment = CartPoleEnv(config, seed)
    agent = DQNAgent(config, device, seed)
    replay = ReplayBuffer(int(config["dqn"]["replay_capacity"]), seed)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(output_directory / "tensorboard")
    best_path = output_directory / "best.pt"
    final_path = output_directory / "final.pt"
    episode_records: list[dict[str, float | int | bool]] = []
    validation_records: list[dict[str, float | int]] = []
    recent_losses: deque[float] = deque(maxlen=100)
    global_step = 0
    episode = 0
    next_validation = validation_interval
    best_validation = -1.0
    best_validation_record: dict[str, float | int] | None = None
    started = time.perf_counter()
    dqn_config = config["dqn"]

    try:
        while global_step < environment_steps:
            episode += 1
            state = environment.reset(seed + episode)
            episode_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated) and global_step < environment_steps:
                epsilon = dqn_epsilon(config, global_step)
                action = agent.act(state, epsilon)
                next_state, reward, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                replay.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                global_step += 1

                if (
                    global_step >= int(dqn_config["learning_starts"])
                    and global_step % int(dqn_config["train_frequency"]) == 0
                ):
                    loss = agent.optimize(replay)
                    if loss is not None:
                        recent_losses.append(loss)
                        writer.add_scalar("train/td_loss", loss, global_step)
                if global_step % int(dqn_config["target_update_frequency"]) == 0:
                    agent.update_target()

                if global_step >= next_validation:
                    validation = evaluate_for_validation(
                        lambda value: agent.act(value, epsilon=0.0),
                        config,
                        validation_episodes,
                        validation_seed,
                    )
                    record: dict[str, float | int] = {
                        "global_step": global_step,
                        **validation,
                    }
                    validation_records.append(record)
                    writer.add_scalar(
                        "validation/mean_steps",
                        validation["mean_steps"],
                        global_step,
                    )
                    writer.add_scalar(
                        "validation/success_rate",
                        validation["success_rate"],
                        global_step,
                    )
                    if validation["mean_steps"] > best_validation:
                        best_validation = validation["mean_steps"]
                        best_validation_record = record
                        save_dqn(
                            best_path,
                            agent,
                            checkpoint_metadata(
                                config,
                                algorithm="dqn",
                                variant=variant,
                                seed=seed,
                                global_step=global_step,
                                validation=validation,
                                checkpoint_kind="best",
                                device=str(device),
                            ),
                        )
                    print(
                        f"[{variant} seed={seed}] step {global_step}/{environment_steps} "
                        f"| validation {validation['mean_steps']:.1f} "
                        f"| success {validation['success_rate']:.0%}"
                    )
                    next_validation += validation_interval

            if terminated or truncated:
                episode_record = {
                    "episode": episode,
                    "global_step": global_step,
                    "steps": environment.step_count,
                    "return": episode_return,
                    "success": bool(truncated),
                    "epsilon": dqn_epsilon(config, global_step),
                    "mean_recent_loss": (
                        float(np.mean(recent_losses)) if recent_losses else 0.0
                    ),
                }
                episode_records.append(episode_record)
                writer.add_scalar("episode/steps", environment.step_count, global_step)
                writer.add_scalar("episode/return", episode_return, global_step)
                writer.add_scalar("episode/success", float(truncated), global_step)
                writer.add_scalar(
                    "exploration/epsilon",
                    episode_record["epsilon"],
                    global_step,
                )
    finally:
        writer.close()

    # 烟雾测试预算可能小于验证间隔，因此结束时补一次验证。
    if not validation_records or validation_records[-1]["global_step"] != global_step:
        validation = evaluate_for_validation(
            lambda value: agent.act(value, epsilon=0.0),
            config,
            validation_episodes,
            validation_seed,
        )
        record = {"global_step": global_step, **validation}
        validation_records.append(record)
        if validation["mean_steps"] > best_validation:
            best_validation_record = record
            save_dqn(
                best_path,
                agent,
                checkpoint_metadata(
                    config,
                    algorithm="dqn",
                    variant=variant,
                    seed=seed,
                    global_step=global_step,
                    validation=validation,
                    checkpoint_kind="best",
                    device=str(device),
                ),
            )

    save_dqn(
        final_path,
        agent,
        checkpoint_metadata(
            config,
            algorithm="dqn",
            variant=variant,
            seed=seed,
            global_step=global_step,
            validation=validation_records[-1],
            checkpoint_kind="final",
            device=str(device),
        ),
    )
    result = {
        "metadata": {
            "completed": True,
            "completed_at": datetime.now().astimezone().isoformat(),
            "formal_config_sha256": config["formal"]["config_sha256"],
            "algorithm": "dqn",
            "variant": variant,
            "label": label,
            "seed": seed,
            "environment_steps": global_step,
            "episodes_completed": len(episode_records),
            "wall_clock_seconds": time.perf_counter() - started,
            "best_validation": best_validation_record,
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "device": str(device),
        },
        "validation": validation_records,
        "episodes": episode_records,
    }
    write_json(output_directory / "training_metrics.json", result)
    return result


def train_q_learning_fixed_steps(
    config: dict[str, Any],
    *,
    variant: str,
    label: str,
    seed: int,
    environment_steps: int,
    validation_interval: int,
    validation_episodes: int,
    validation_seed: int,
    output_directory: Path,
) -> dict[str, Any]:
    """按统一环境交互步数训练一个 Q-Learning 分箱变体。"""

    environment = CartPoleEnv(config, seed)
    agent = QLearningAgent(config, seed)
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(output_directory / "tensorboard")
    best_path = output_directory / "best.npz"
    final_path = output_directory / "final.npz"
    episode_records: list[dict[str, float | int | bool]] = []
    validation_records: list[dict[str, float | int]] = []
    global_step = 0
    episode = 0
    next_validation = validation_interval
    best_validation = -1.0
    best_validation_record: dict[str, float | int] | None = None
    started = time.perf_counter()

    try:
        while global_step < environment_steps:
            episode += 1
            state = environment.reset(seed + episode)
            episode_return = 0.0
            td_errors: list[float] = []
            terminated = truncated = False

            while not (terminated or truncated) and global_step < environment_steps:
                epsilon = q_epsilon(config, global_step)
                action = agent.act(state, epsilon)
                next_state, reward, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                td_errors.append(
                    abs(agent.update(state, action, reward, next_state, done))
                )
                state = next_state
                episode_return += reward
                global_step += 1

                if global_step >= next_validation:
                    validation = evaluate_for_validation(
                        lambda value: agent.act(
                            value,
                            epsilon=0.0,
                            deterministic=True,
                        ),
                        config,
                        validation_episodes,
                        validation_seed,
                    )
                    record: dict[str, float | int] = {
                        "global_step": global_step,
                        **validation,
                    }
                    validation_records.append(record)
                    writer.add_scalar(
                        "validation/mean_steps",
                        validation["mean_steps"],
                        global_step,
                    )
                    writer.add_scalar(
                        "validation/success_rate",
                        validation["success_rate"],
                        global_step,
                    )
                    if validation["mean_steps"] > best_validation:
                        best_validation = validation["mean_steps"]
                        best_validation_record = record
                        save_q_table(
                            best_path,
                            agent,
                            checkpoint_metadata(
                                config,
                                algorithm="q_learning",
                                variant=variant,
                                seed=seed,
                                global_step=global_step,
                                validation=validation,
                                checkpoint_kind="best",
                                device="cpu",
                            ),
                        )
                    print(
                        f"[{variant} seed={seed}] step {global_step}/{environment_steps} "
                        f"| validation {validation['mean_steps']:.1f} "
                        f"| success {validation['success_rate']:.0%}"
                    )
                    next_validation += validation_interval

            if terminated or truncated:
                episode_record = {
                    "episode": episode,
                    "global_step": global_step,
                    "steps": environment.step_count,
                    "return": episode_return,
                    "success": bool(truncated),
                    "epsilon": q_epsilon(config, global_step),
                    "mean_absolute_td_error": (
                        float(np.mean(td_errors)) if td_errors else 0.0
                    ),
                }
                episode_records.append(episode_record)
                writer.add_scalar("episode/steps", environment.step_count, global_step)
                writer.add_scalar("episode/return", episode_return, global_step)
                writer.add_scalar("episode/success", float(truncated), global_step)
                writer.add_scalar(
                    "exploration/epsilon",
                    episode_record["epsilon"],
                    global_step,
                )
                writer.add_scalar(
                    "train/mean_absolute_td_error",
                    episode_record["mean_absolute_td_error"],
                    global_step,
                )
    finally:
        writer.close()

    if not validation_records or validation_records[-1]["global_step"] != global_step:
        validation = evaluate_for_validation(
            lambda value: agent.act(
                value,
                epsilon=0.0,
                deterministic=True,
            ),
            config,
            validation_episodes,
            validation_seed,
        )
        record = {"global_step": global_step, **validation}
        validation_records.append(record)
        if validation["mean_steps"] > best_validation:
            best_validation_record = record
            save_q_table(
                best_path,
                agent,
                checkpoint_metadata(
                    config,
                    algorithm="q_learning",
                    variant=variant,
                    seed=seed,
                    global_step=global_step,
                    validation=validation,
                    checkpoint_kind="best",
                    device="cpu",
                ),
            )

    save_q_table(
        final_path,
        agent,
        checkpoint_metadata(
            config,
            algorithm="q_learning",
            variant=variant,
            seed=seed,
            global_step=global_step,
            validation=validation_records[-1],
            checkpoint_kind="final",
            device="cpu",
        ),
    )
    result = {
        "metadata": {
            "completed": True,
            "completed_at": datetime.now().astimezone().isoformat(),
            "formal_config_sha256": config["formal"]["config_sha256"],
            "algorithm": "q_learning",
            "variant": variant,
            "label": label,
            "seed": seed,
            "environment_steps": global_step,
            "episodes_completed": len(episode_records),
            "wall_clock_seconds": time.perf_counter() - started,
            "best_validation": best_validation_record,
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "device": "cpu",
        },
        "validation": validation_records,
        "episodes": episode_records,
    }
    write_json(output_directory / "training_metrics.json", result)
    return result
