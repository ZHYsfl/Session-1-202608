"""Actor-Critic 与 PPO 数值工具测试。"""

from __future__ import annotations

import torch
from common import load_config
from policy import ActorCritic
from ppo import compute_gae


def test_cnn_and_mlp_action_shapes() -> None:
    config = load_config()
    observation = torch.zeros((5, 43))
    for model_type in ("cnn", "mlp"):
        model = ActorCritic(43, 36, model_type, config["model"])
        action, raw, log_probability, entropy, value = model.get_action_and_value(
            observation
        )
        assert action.shape == (5, 2)
        assert raw.shape == (5, 2)
        assert log_probability.shape == (5,)
        assert entropy.shape == (5,)
        assert value.shape == (5,)
        assert torch.all(action.abs() <= 1.0)


def test_gae_stops_bootstrap_at_terminal() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    dones = torch.tensor([[0.0], [1.0]])
    values = torch.zeros_like(rewards)
    advantages, returns = compute_gae(
        rewards,
        dones,
        values,
        next_value=torch.tensor([100.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(advantages[:, 0], torch.tensor([3.0, 2.0]))
    assert torch.allclose(returns, advantages)
