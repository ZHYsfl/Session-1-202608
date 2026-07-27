"""PPO 使用的 Actor-Critic 策略网络。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


def _initialize_layer(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    """使用 PPO 常见的正交初始化。"""

    if isinstance(layer, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """支持射线 CNN 主模型与扁平 MLP 对照模型。"""

    def __init__(
        self,
        observation_dim: int,
        sensor_count: int,
        model_type: str,
        model_config: dict[str, Any],
    ) -> None:
        super().__init__()
        if model_type not in {"cnn", "mlp"}:
            raise ValueError("model_type 必须是 cnn 或 mlp")
        if observation_dim != sensor_count + 7:
            raise ValueError("观测维度必须等于射线数加 7")
        self.observation_dim = observation_dim
        self.sensor_count = sensor_count
        self.model_type = model_type

        if model_type == "cnn":
            channels = [int(value) for value in model_config["ray_channels"]]
            self.ray_encoder = nn.Sequential(
                _initialize_layer(
                    nn.Conv1d(1, channels[0], kernel_size=5, padding=2),
                    gain=nn.init.calculate_gain("relu"),
                ),
                nn.ReLU(),
                _initialize_layer(
                    nn.Conv1d(
                        channels[0],
                        channels[1],
                        kernel_size=5,
                        dilation=2,
                        padding=4,
                    ),
                    gain=nn.init.calculate_gain("relu"),
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            auxiliary_hidden = int(model_config["auxiliary_hidden"])
            self.auxiliary_encoder = nn.Sequential(
                _initialize_layer(
                    nn.Linear(7, auxiliary_hidden),
                    gain=nn.init.calculate_gain("tanh"),
                ),
                nn.Tanh(),
            )
            fused_hidden = int(model_config["fused_hidden"])
            self.trunk = nn.Sequential(
                _initialize_layer(
                    nn.Linear(channels[1] + auxiliary_hidden, fused_hidden),
                    gain=nn.init.calculate_gain("tanh"),
                ),
                nn.Tanh(),
            )
            feature_dim = fused_hidden
        else:
            hidden = [int(value) for value in model_config["mlp_hidden"]]
            layers: list[nn.Module] = []
            current = observation_dim
            for width in hidden:
                layers.extend(
                    (
                        _initialize_layer(
                            nn.Linear(current, width),
                            gain=nn.init.calculate_gain("tanh"),
                        ),
                        nn.Tanh(),
                    )
                )
                current = width
            self.trunk = nn.Sequential(*layers)
            feature_dim = current

        self.actor_mean = _initialize_layer(nn.Linear(feature_dim, 2), 0.01)
        self.critic = _initialize_layer(nn.Linear(feature_dim, 1), 1.0)
        self.actor_log_std = nn.Parameter(torch.full((2,), -0.5))

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """把局部观测编码为共享策略特征。"""

        if self.model_type == "cnn":
            rays = observation[:, : self.sensor_count].unsqueeze(1)
            auxiliary = observation[:, self.sensor_count :]
            features = torch.cat(
                (
                    self.ray_encoder(rays),
                    self.auxiliary_encoder(auxiliary),
                ),
                dim=1,
            )
            return self.trunk(features)
        return self.trunk(observation)

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        """估计状态价值。"""

        return self.critic(self.encode(observation)).squeeze(-1)

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """采样 tanh 高斯动作并返回 PPO 所需统计量。"""

        features = self.encode(observation)
        mean = self.actor_mean(features)
        standard_deviation = self.actor_log_std.exp().expand_as(mean)
        distribution = Normal(mean, standard_deviation)
        if raw_action is None:
            raw_action = mean if deterministic else distribution.sample()
        action = torch.tanh(raw_action)
        log_probability = (
            distribution.log_prob(raw_action) - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        value = self.critic(features).squeeze(-1)
        return action, raw_action, log_probability, entropy, value


def save_policy(
    path: Path,
    model: ActorCritic,
    metadata: dict[str, Any],
) -> None:
    """保存可恢复的策略 state_dict 和结构元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "observation_dim": model.observation_dim,
            "sensor_count": model.sensor_count,
            "model_type": model.model_type,
            "metadata": metadata,
        },
        path,
    )


def load_policy(
    path: Path,
    model_config: dict[str, Any],
    device: torch.device,
) -> tuple[ActorCritic, dict[str, Any]]:
    """从 checkpoint 恢复策略并切换为评估模式。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(
        int(checkpoint["observation_dim"]),
        int(checkpoint["sensor_count"]),
        str(checkpoint["model_type"]),
        model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, dict(checkpoint.get("metadata", {}))
