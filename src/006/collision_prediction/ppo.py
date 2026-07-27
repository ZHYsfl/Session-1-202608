"""PPO 的 GAE 与裁剪更新实现。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from policy import ActorCritic


@dataclass(frozen=True)
class UpdateMetrics:
    """一次 PPO 更新的标量诊断。"""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按时间倒序计算广义优势估计与回报。"""

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    raw_actions: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    update_epochs: int,
    minibatch_size: int,
    clip_coef: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> UpdateMetrics:
    """在一个 rollout 上执行多轮随机小批量 PPO 更新。"""

    batch_size = observations.shape[0]
    minibatch_size = max(1, min(minibatch_size, batch_size))
    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approximate_kls: list[float] = []

    model.train()
    for _ in range(update_epochs):
        permutation = torch.randperm(batch_size, device=observations.device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            _, _, new_log_probability, entropy, value = model.get_action_and_value(
                observations[indices],
                raw_actions[indices],
            )
            log_ratio = new_log_probability - old_log_probabilities[indices]
            ratio = log_ratio.exp()
            unclipped = -normalized_advantages[indices] * ratio
            clipped = -normalized_advantages[indices] * torch.clamp(
                ratio, 1.0 - clip_coef, 1.0 + clip_coef
            )
            policy_loss = torch.maximum(unclipped, clipped).mean()
            value_loss = 0.5 * (value - returns[indices]).square().mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy_mean.detach()))
            approximate_kls.append(float(approximate_kl.detach()))

    return UpdateMetrics(
        policy_loss=sum(policy_losses) / max(len(policy_losses), 1),
        value_loss=sum(value_losses) / max(len(value_losses), 1),
        entropy=sum(entropies) / max(len(entropies), 1),
        approximate_kl=sum(approximate_kls) / max(len(approximate_kls), 1),
    )
