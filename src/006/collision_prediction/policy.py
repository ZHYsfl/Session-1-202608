"""PPO 使用的 Actor-Critic 策略网络。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal


def _initialize_layer(layer: nn.Module, gain: float = 1.0) -> nn.Module:
    """使用正交初始化稳定 PPO 训练。"""

    if isinstance(layer, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class CircularConvBlock(nn.Module):
    """在 360° 射线首尾处使用环形填充的一维卷积块。"""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        total_padding = dilation * (kernel_size - 1)
        self.left_padding = total_padding // 2
        self.right_padding = total_padding - self.left_padding
        self.conv = _initialize_layer(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            gain=nn.init.calculate_gain("relu"),
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padded = nn.functional.pad(
            values,
            (self.left_padding, self.right_padding),
            mode="circular",
        )
        return self.activation(self.conv(padded))


class ActorCritic(nn.Module):
    """支持保留角度结构的射线 CNN 与扁平 MLP 对照模型。"""

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
            sectors = int(model_config["ray_sectors"])
            if sectors <= 1 or sectors > sensor_count:
                raise ValueError("ray_sectors 必须位于 2 到 sensor_count 之间")
            self.ray_encoder = nn.Sequential(
                CircularConvBlock(1, channels[0], kernel_size=5),
                CircularConvBlock(
                    channels[0],
                    channels[1],
                    kernel_size=5,
                    dilation=2,
                ),
                nn.AdaptiveAvgPool1d(sectors),
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
                    nn.Linear(channels[1] * sectors + auxiliary_hidden, fused_hidden),
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


def policy_payload(
    model: ActorCritic,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """构造可由演示和评估脚本读取的策略数据。"""

    return {
        "model_state": model.state_dict(),
        "observation_dim": model.observation_dim,
        "sensor_count": model.sensor_count,
        "model_type": model.model_type,
        "metadata": metadata,
    }


def save_policy(
    path: Path,
    model: ActorCritic,
    metadata: dict[str, Any],
) -> None:
    """保存仅用于推理的策略 checkpoint。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_payload(model, metadata), path)


def load_policy(
    path: Path,
    model_config: dict[str, Any],
    device: torch.device,
    *,
    expected_version: str | None = None,
) -> tuple[ActorCritic, dict[str, Any]]:
    """恢复策略并拒绝不兼容的实验版本。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(checkpoint.get("metadata", {}))
    checkpoint_version = metadata.get("experiment_version")
    if expected_version is not None and checkpoint_version != expected_version:
        raise RuntimeError(
            f"checkpoint 版本不匹配：需要 {expected_version}，"
            f"实际为 {checkpoint_version!r}"
        )
    model = ActorCritic(
        int(checkpoint["observation_dim"]),
        int(checkpoint["sensor_count"]),
        str(checkpoint["model_type"]),
        model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, metadata
