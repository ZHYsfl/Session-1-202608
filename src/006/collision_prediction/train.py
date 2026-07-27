"""在 GPU 批量环境中训练局部避障 PPO 策略。"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
from common import (
    DEFAULT_CONFIG,
    load_config,
    resolve_project_path,
    seed_everything,
    select_device,
)
from policy import ActorCritic, save_policy
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
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="关闭课程第三阶段的测距噪声",
    )
    return parser


def _checkpoint_path(config: dict[str, Any], model_type: str) -> Path:
    key = f"{model_type}_checkpoint"
    return resolve_project_path(str(config["paths"][key]))


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    ppo_config = config["ppo"]
    seed = int(ppo_config["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    num_envs = args.num_envs or int(ppo_config["num_envs"])
    rollout_steps = args.rollout_steps or int(ppo_config["rollout_steps"])
    total_steps = args.total_steps or int(ppo_config["total_steps"])
    transitions_per_update = num_envs * rollout_steps
    update_count = max(1, total_steps // transitions_per_update)

    env = BatchedLocalAvoidanceEnv(
        config,
        num_envs,
        device,
        seed,
        training_noise=not args.no_noise,
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
    observation = env.observe()
    checkpoint_path = _checkpoint_path(config, args.model)
    metrics_path = resolve_project_path(
        str(config["paths"][f"{args.model}_training_metrics"])
    )
    run_dir = checkpoint_path.parent / f"tensorboard_{args.model}"
    writer = SummaryWriter(run_dir) if SummaryWriter is not None else None

    recent_results: deque[float] = deque(maxlen=500)
    history: list[dict[str, float | int | str]] = []
    best_success_rate = -1.0
    global_step = 0

    print(
        f"设备={device} 模型={args.model} 环境数={num_envs} "
        f"每次采样={rollout_steps} 总更新={update_count}"
    )
    for update in range(1, update_count + 1):
        observations = torch.zeros(
            (rollout_steps, num_envs, env.observation_dim), device=device
        )
        raw_actions = torch.zeros((rollout_steps, num_envs, 2), device=device)
        log_probabilities = torch.zeros((rollout_steps, num_envs), device=device)
        rewards = torch.zeros((rollout_steps, num_envs), device=device)
        dones = torch.zeros((rollout_steps, num_envs), device=device)
        values = torch.zeros((rollout_steps, num_envs), device=device)

        for step in range(rollout_steps):
            observations[step] = observation
            with torch.no_grad():
                action, raw_action, log_probability, _, value = (
                    model.get_action_and_value(observation)
                )
            observation, reward, done, info = env.step(action)
            raw_actions[step] = raw_action
            log_probabilities[step] = log_probability
            rewards[step] = reward
            dones[step] = done.float()
            values[step] = value
            global_step += num_envs

            finished = torch.nonzero(done, as_tuple=False).flatten()
            if finished.numel():
                recent_results.extend(
                    info["success"][finished].float().detach().cpu().tolist()
                )

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
        success_rate = (
            sum(recent_results) / len(recent_results) if recent_results else 0.0
        )
        promotion_rate = float(config["environment"]["promotion_success_rate"])
        if (
            len(recent_results) >= 100
            and success_rate >= promotion_rate
            and env.curriculum_stage < 2
        ):
            env.set_curriculum_stage(env.curriculum_stage + 1)
            recent_results.clear()

        row = {
            "update": update,
            "global_step": global_step,
            "curriculum_stage": env.curriculum_stage,
            "success_rate": success_rate,
            "mean_reward": float(rewards.mean()),
            "policy_loss": update_metrics.policy_loss,
            "value_loss": update_metrics.value_loss,
            "entropy": update_metrics.entropy,
            "approximate_kl": update_metrics.approximate_kl,
            "device": str(device),
            "model": args.model,
        }
        history.append(row)
        if writer is not None:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, global_step)

        metadata = {
            "seed": seed,
            "global_step": global_step,
            "curriculum_stage": env.curriculum_stage,
            "success_rate": success_rate,
        }
        best_success_rate = max(best_success_rate, success_rate)
        # 每次更新保存最新策略，避免简单课程阶段的高成功率锁死旧模型。
        save_policy(checkpoint_path, model, metadata)
        print(
            f"更新 {update:04d}/{update_count} | step={global_step} "
            f"| stage={env.curriculum_stage} | success={success_rate:.3f} "
            f"| reward={row['mean_reward']:.3f}"
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "best_success_rate": best_success_rate,
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if writer is not None:
        writer.close()
    print(f"策略已保存：{checkpoint_path}")
    print(f"训练记录：{metrics_path}")


if __name__ == "__main__":
    main()
