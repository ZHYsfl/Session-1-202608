"""PyTorch 多层感知机及模型保存、加载工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


class CollisionMLP(nn.Module):
    """根据多路测距与速度特征输出单个碰撞 logit。"""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        dimensions = [input_dim, *(int(value) for value in hidden_dims)]
        if input_dim <= 0 or any(value <= 0 for value in dimensions):
            raise ValueError("网络层维度必须为正数")

        layers: list[nn.Module] = []
        for source_dim, target_dim in zip(
            dimensions[:-1],
            dimensions[1:],
            strict=True,
        ):
            layers.extend(
                [
                    nn.Linear(source_dim, target_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                ]
            )
        layers.append(nn.Linear(dimensions[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回未经过 Sigmoid 的 logits，供 BCEWithLogitsLoss 使用。"""

        return self.layers(features).squeeze(-1)


def save_checkpoint(
    path: Path,
    model: CollisionMLP,
    input_dim: int,
    hidden_dims: list[int],
    metadata: dict[str, Any],
) -> None:
    """保存 state_dict 和重建网络所需的最小元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[CollisionMLP, dict[str, Any]]:
    """从可信的本地检查点恢复模型。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CollisionMLP(
        int(checkpoint["input_dim"]),
        list(checkpoint["hidden_dims"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, dict(checkpoint.get("metadata", {}))


def predict_probabilities(
    model: CollisionMLP,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """批量计算碰撞概率。"""

    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        return torch.sigmoid(model(tensor)).cpu().numpy()
