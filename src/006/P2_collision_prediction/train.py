"""在 GPU 批量环境中正式训练局部避障 CNN-PPO 策略。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from common import (
    DEFAULT_CONFIG,
    load_config,
    resolve_project_path,
    seed_everything,
    select_device,
)
from planner import astar_path_length
from policy import ActorCritic, policy_payload, save_policy
from ppo import compute_gae, ppo_update
from rl_env import BatchedLocalAvoidanceEnv

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # TensorBoard 是可选的训练可视化依赖。
    SummaryWriter = None  # type: ignore[assignment,misc]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--total-steps", type=int, help="覆盖最大总训练步数")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument(
        "--resume", action="store_true", help="从 latest checkpoint 续训"
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="关闭第三阶段训练测距噪声",
    )
    return parser


def _path(config: dict[str, Any], key: str) -> Path:
    return resolve_project_path(str(config["paths"][key]))


def _stage_best_path(latest_path: Path, stage: int) -> Path:
    name = latest_path.name.replace("_latest", f"_stage{stage}_best")
    return latest_path.with_name(name)


def _mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def validate_policy(
    model: ActorCritic,
    config: dict[str, Any],
    device: torch.device,
    stage: int,
    episodes: int,
    num_envs: int,
) -> dict[str, float | int]:
    """在固定、无噪声、从不参与训练的任务上评估当前策略。"""

    was_training = model.training
    model.eval()
    completed = success_count = collision_count = timeout_count = 0
    episode_rewards: list[float] = []
    path_ratios: list[float] = []
    seed = int(config["validation"]["seed"]) + stage * 1000

    while completed < episodes:
        batch_size = min(num_envs, episodes - completed)
        env = BatchedLocalAvoidanceEnv(
            config,
            batch_size,
            device,
            seed + completed,
            training_noise=False,
            auto_reset=False,
        )
        env.set_curriculum_stage(stage)
        observation = env.reset()
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        returns = torch.zeros(batch_size, device=device)
        references: list[float | None] = []
        for index in range(batch_size):
            snapshot = env.snapshot(index)
            references.append(
                astar_path_length(
                    env.width,
                    env.height,
                    snapshot["obstacles"],  # type: ignore[arg-type]
                    snapshot["position"],  # type: ignore[arg-type]
                    snapshot["goal"],  # type: ignore[arg-type]
                    env.radius + 2.0,
                )
            )

        while bool(active.any()):
            action, _, _, _, _ = model.get_action_and_value(
                observation, deterministic=True
            )
            action[~active, 0] = -1.0
            action[~active, 1] = 0.0
            observation, reward, done, info = env.step(action)
            returns += torch.where(active, reward, torch.zeros_like(reward))
            finished = done & active
            if not bool(finished.any()):
                continue
            indices = torch.nonzero(finished, as_tuple=False).flatten()
            success_count += int(info["success"][indices].sum())
            collision_count += int(info["collision"][indices].sum())
            timeout_count += int(info["timeout"][indices].sum())
            for index in indices.detach().cpu().tolist():
                episode_rewards.append(float(returns[index].detach().cpu()))
                reference = references[index]
                if bool(info["success"][index]) and reference:
                    path = float(info["episode_path_length"][index].detach().cpu())
                    path_ratios.append(path / reference)
            active[indices] = False
        completed += batch_size

    if was_training:
        model.train()
    return {
        "stage": stage,
        "episodes": completed,
        "success_rate": success_count / completed,
        "collision_rate": collision_count / completed,
        "timeout_rate": timeout_count / completed,
        "mean_episode_reward": _mean_or_zero(episode_rewards),
        "mean_path_ratio": (
            _mean_or_zero(path_ratios) if path_ratios else float("nan")
        ),
    }


def _validation_passed(
    metrics: dict[str, float | int],
    config: dict[str, Any],
    stage: int,
) -> bool:
    validation = config["validation"]
    if stage < len(config["environment"]["stages"]) - 1:
        return float(metrics["success_rate"]) >= float(
            validation["promotion_success_rate"]
        ) and float(metrics["collision_rate"]) <= float(
            validation["promotion_collision_rate"]
        )
    return (
        float(metrics["success_rate"]) >= float(validation["final_success_rate"])
        and float(metrics["collision_rate"])
        <= float(validation["final_collision_rate"])
        and float(metrics["timeout_rate"]) <= float(validation["final_timeout_rate"])
    )


def _checkpoint_metadata(
    config: dict[str, Any],
    model_type: str,
    global_step: int,
    stage: int,
    validation_metrics: dict[str, float | int] | None,
) -> dict[str, Any]:
    return {
        "experiment_version": str(config["experiment"]["version"]),
        "model": model_type,
        "global_step": global_step,
        "curriculum_stage": stage,
        "validation": validation_metrics,
    }


def _save_training_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    *,
    stage_steps: int,
    consecutive_passes: int,
    best_scores: list[tuple[float, float]],
    history: list[dict[str, Any]],
    environment_state: torch.Tensor,
) -> None:
    """保存可继续训练的完整状态。"""

    payload = policy_payload(model, metadata)
    payload.update(
        {
            "optimizer_state": optimizer.state_dict(),
            "stage_steps": stage_steps,
            "consecutive_passes": consecutive_passes,
            "best_scores": best_scores,
            "history": history,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "environment_random_state": environment_state,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _restore_training_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    env: BatchedLocalAvoidanceEnv,
    expected_version: str,
) -> tuple[int, int, int, int, list[tuple[float, float]], list[dict[str, Any]]]:
    """恢复完整训练状态并返回进度字段。"""

    checkpoint = torch.load(path, map_location=env.device, weights_only=False)
    metadata = dict(checkpoint.get("metadata", {}))
    actual_version = metadata.get("experiment_version")
    if actual_version != expected_version:
        raise RuntimeError(
            f"无法续训版本 {actual_version!r}；当前版本为 {expected_version}"
        )
    if str(checkpoint.get("model_type")) != model.model_type:
        raise RuntimeError("checkpoint 模型类型与 --model 不一致")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    cuda_states = checkpoint.get("cuda_random_states")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
    env.set_generator_state(checkpoint["environment_random_state"])
    stage = int(metadata["curriculum_stage"])
    env.set_curriculum_stage(stage)
    scores = [
        (float(success), float(collision))
        for success, collision in checkpoint.get("best_scores", [])
    ]
    while len(scores) < len(env.stage_configs):
        scores.append((-1.0, 1.0))
    return (
        int(metadata["global_step"]),
        stage,
        int(checkpoint.get("stage_steps", 0)),
        int(checkpoint.get("consecutive_passes", 0)),
        scores,
        list(checkpoint.get("history", [])),
    )


def _write_metrics(
    path: Path,
    model_type: str,
    version: str,
    completed: bool,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_version": version,
                "model": model_type,
                "formal_training_complete": completed,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    ppo_config = config["ppo"]
    validation_config = config["validation"]
    seed = int(ppo_config["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    num_envs = args.num_envs or int(ppo_config["num_envs"])
    rollout_steps = args.rollout_steps or int(ppo_config["rollout_steps"])
    max_total_steps = args.total_steps or int(ppo_config["max_total_steps"])
    validation_episodes = args.validation_episodes or int(validation_config["episodes"])
    validation_num_envs = min(int(validation_config["num_envs"]), validation_episodes)
    validation_interval = int(validation_config["interval_updates"])
    required_passes = int(validation_config["consecutive_passes"])
    transitions_per_update = num_envs * rollout_steps
    estimated_updates = math.ceil(max_total_steps / transitions_per_update)

    env = BatchedLocalAvoidanceEnv(
        config,
        num_envs,
        device,
        seed,
        training_noise=not args.no_noise,
        mix_previous_stages=True,
    )
    model = ActorCritic(
        env.observation_dim,
        env.sensor_count,
        args.model,
        config["model"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(ppo_config["learning_rate"])
    )
    latest_path = _path(config, f"{args.model}_latest_checkpoint")
    publish_path = _path(config, f"{args.model}_checkpoint")
    metrics_path = _path(config, f"{args.model}_training_metrics")
    run_dir = latest_path.parent / f"tensorboard_{args.model}_v2"
    writer = SummaryWriter(run_dir) if SummaryWriter is not None else None

    global_step = 0
    stage = int(config["environment"]["initial_curriculum_stage"])
    stage_steps = 0
    consecutive_passes = 0
    best_scores = [(-1.0, 1.0) for _ in env.stage_configs]
    history: list[dict[str, Any]] = []
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"找不到续训 checkpoint：{latest_path}")
        (
            global_step,
            stage,
            stage_steps,
            consecutive_passes,
            best_scores,
            history,
        ) = _restore_training_checkpoint(
            latest_path,
            model,
            optimizer,
            env,
            str(config["experiment"]["version"]),
        )
        print(f"已恢复：step={global_step} stage={stage} stage_steps={stage_steps}")
    else:
        env.set_curriculum_stage(stage)

    observation = env.reset()
    recent_results: deque[float] = deque(maxlen=500)
    training_complete = False
    update = global_step // transitions_per_update
    last_validation: dict[str, float | int] | None = None

    print(
        f"设备={device} 模型={args.model} 环境数={num_envs} "
        f"每次采样={rollout_steps} 最大更新≈{estimated_updates}"
    )
    while global_step < max_total_steps and not training_complete:
        update += 1
        observations = torch.zeros(
            (rollout_steps, num_envs, env.observation_dim), device=device
        )
        raw_actions = torch.zeros((rollout_steps, num_envs, 2), device=device)
        log_probabilities = torch.zeros((rollout_steps, num_envs), device=device)
        rewards = torch.zeros((rollout_steps, num_envs), device=device)
        dones = torch.zeros((rollout_steps, num_envs), device=device)
        values = torch.zeros((rollout_steps, num_envs), device=device)

        for rollout_step in range(rollout_steps):
            observations[rollout_step] = observation
            with torch.no_grad():
                action, raw_action, log_probability, _, value = (
                    model.get_action_and_value(observation)
                )
            observation, reward, done, info = env.step(action)
            raw_actions[rollout_step] = raw_action
            log_probabilities[rollout_step] = log_probability
            rewards[rollout_step] = reward
            dones[rollout_step] = done.float()
            values[rollout_step] = value

            finished = torch.nonzero(done, as_tuple=False).flatten()
            if finished.numel():
                recent_results.extend(
                    info["success"][finished].float().detach().cpu().tolist()
                )

        global_step += transitions_per_update
        stage_steps += transitions_per_update
        with torch.no_grad():
            next_value = model.get_value(observation)
        advantages, returns = compute_gae(
            rewards,
            dones,
            values,
            next_value,
            float(ppo_config["gamma"]),
            float(ppo_config["gae_lambda"]),
        )
        update_metrics = ppo_update(
            model,
            optimizer,
            observations.flatten(0, 1),
            raw_actions.flatten(0, 1),
            log_probabilities.flatten(0, 1),
            advantages.flatten(),
            returns.flatten(),
            update_epochs=int(ppo_config["update_epochs"]),
            minibatch_size=int(ppo_config["minibatch_size"]),
            clip_coef=float(ppo_config["clip_coef"]),
            value_coef=float(ppo_config["value_coef"]),
            entropy_coef=float(ppo_config["entropy_coef"]),
            max_grad_norm=float(ppo_config["max_grad_norm"]),
        )
        training_success = (
            sum(recent_results) / len(recent_results) if recent_results else 0.0
        )
        row: dict[str, Any] = {
            "update": update,
            "global_step": global_step,
            "curriculum_stage": stage,
            "stage_steps": stage_steps,
            "training_success_rate": training_success,
            "mean_step_reward": float(rewards.mean()),
            "policy_loss": update_metrics.policy_loss,
            "value_loss": update_metrics.value_loss,
            "entropy": update_metrics.entropy,
            "approximate_kl": update_metrics.approximate_kl,
            "device": str(device),
            "model": args.model,
        }

        should_validate = update % validation_interval == 0
        if should_validate:
            last_validation = validate_policy(
                model,
                config,
                device,
                stage,
                validation_episodes,
                validation_num_envs,
            )
            row["validation"] = last_validation
            minimum_steps = int(env.stage_configs[stage]["minimum_steps"])
            eligible = stage_steps >= minimum_steps
            passed = eligible and _validation_passed(last_validation, config, stage)
            consecutive_passes = consecutive_passes + 1 if passed else 0

            score = (
                float(last_validation["success_rate"]),
                -float(last_validation["collision_rate"]),
            )
            if score > best_scores[stage]:
                best_scores[stage] = score
                metadata = _checkpoint_metadata(
                    config, args.model, global_step, stage, last_validation
                )
                save_policy(_stage_best_path(latest_path, stage), model, metadata)

            if consecutive_passes >= required_passes:
                final_stage = len(env.stage_configs) - 1
                if stage == final_stage:
                    training_complete = True
                    metadata = _checkpoint_metadata(
                        config, args.model, global_step, stage, last_validation
                    )
                    save_policy(publish_path, model, metadata)
                else:
                    stage += 1
                    env.set_curriculum_stage(stage)
                    observation = env.reset()
                    stage_steps = 0
                    consecutive_passes = 0
                    recent_results.clear()

        history.append(row)
        if writer is not None:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar(f"train/{key}", value, global_step)
            if last_validation is not None and should_validate:
                for key, value in last_validation.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"validation/{key}", value, global_step)

        metadata = _checkpoint_metadata(
            config, args.model, global_step, stage, last_validation
        )
        _save_training_checkpoint(
            latest_path,
            model,
            optimizer,
            metadata,
            stage_steps=stage_steps,
            consecutive_passes=consecutive_passes,
            best_scores=best_scores,
            history=history,
            environment_state=env.get_generator_state(),
        )
        _write_metrics(
            metrics_path,
            args.model,
            str(config["experiment"]["version"]),
            training_complete,
            history,
        )

        validation_text = ""
        if should_validate and last_validation is not None:
            validation_text = (
                f" | val_success={float(last_validation['success_rate']):.3f}"
                f" val_collision={float(last_validation['collision_rate']):.3f}"
                f" pass={consecutive_passes}/{required_passes}"
            )
        print(
            f"更新 {update:04d} | step={global_step} | stage={stage} "
            f"| train_success={training_success:.3f} "
            f"| reward={row['mean_step_reward']:.3f}{validation_text}"
        )

    if writer is not None:
        writer.close()
    if training_complete:
        print(f"正式训练通过验证，发布策略：{publish_path}")
    else:
        print(
            f"达到训练上限 {max_total_steps}，尚未通过正式标准；"
            f"可使用 --resume 延长训练。latest={latest_path}"
        )
    print(f"训练记录：{metrics_path}")


if __name__ == "__main__":
    main()
