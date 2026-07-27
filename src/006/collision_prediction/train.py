"""生成仿真数据并训练机器人碰撞预测多层感知机。"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import generate_dataset, split_dataset
from metrics import classification_metrics
from network import CollisionMLP, predict_probabilities, save_checkpoint

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置文件根节点必须是键值映射")
    return config


def resolve_project_path(value: str) -> Path:
    """将配置中的相对输出路径限制在当前实验目录。"""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def select_device(requested: str) -> torch.device:
    """根据命令行选项选择 CPU 或 CUDA。"""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """将 NumPy 数组转换为 PyTorch DataLoader。"""

    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def loader_loss(
    model: CollisionMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """计算一个数据划分上的平均损失。"""

    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            loss = criterion(model(features), labels)
            total_loss += float(loss.item()) * len(features)
            total_count += len(features)
    return total_loss / max(total_count, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, help="覆盖配置中的训练样本数")
    parser.add_argument("--epochs", type=int, help="覆盖配置中的训练轮数")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto、cpu、cuda 或 cuda:0",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="将本次训练数据保存到 outputs/dataset.npz",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    training = config["training"]
    seed = int(training["seed"])
    sample_count = args.samples or int(training["samples"])
    epochs = args.epochs or int(training["epochs"])
    batch_size = int(training["batch_size"])
    hidden_dims = [int(value) for value in training["hidden_dims"]]
    device = select_device(args.device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"正在生成 {sample_count} 个平衡仿真样本...")
    features, labels = generate_dataset(config, sample_count, seed)
    train_set, validation_set, test_set = split_dataset(
        features,
        labels,
        seed,
    )
    if args.save_dataset:
        dataset_path = PROJECT_DIR / "outputs" / "dataset.npz"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, features=features, labels=labels)

    train_loader = make_loader(*train_set, batch_size, shuffle=True)
    validation_loader = make_loader(
        *validation_set,
        batch_size,
        shuffle=False,
    )
    model = CollisionMLP(features.shape[1], hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
    )

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_features)
            seen += len(batch_features)

        train_loss = running_loss / max(seen, 1)
        validation_loss = loader_loss(
            model,
            validation_loader,
            criterion,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch:03d}/{epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={validation_loss:.4f}"
        )

    if best_state is None:
        raise RuntimeError("训练未产生有效模型")
    model.load_state_dict(best_state)

    test_features, test_labels = test_set
    test_probabilities = predict_probabilities(
        model,
        test_features,
        device,
    )
    test_metrics = classification_metrics(
        test_labels,
        test_probabilities,
        float(config["prediction"]["warning_threshold"]),
    )
    metadata = {
        "seed": seed,
        "samples": sample_count,
        "best_validation_loss": best_validation_loss,
        "test_metrics": test_metrics,
        "sensor_count": len(config["sensors"]["angles_deg"]),
    }

    checkpoint_path = resolve_project_path(config["paths"]["checkpoint"])
    save_checkpoint(
        checkpoint_path,
        model,
        features.shape[1],
        hidden_dims,
        metadata,
    )

    metrics_path = resolve_project_path(config["paths"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "device": str(device),
                "history": history,
                **metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"模型已保存：{checkpoint_path}")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
