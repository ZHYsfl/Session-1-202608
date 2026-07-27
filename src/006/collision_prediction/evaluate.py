"""在新随机场景上比较神经网络与最小距离阈值基线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import generate_dataset
from metrics import classification_metrics
from network import load_checkpoint, predict_probabilities
from train import load_config, resolve_project_path, select_device

PROJECT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "config.yaml",
    )
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--rule-threshold",
        type=float,
        default=0.25,
        help="规则基线的归一化最小距离阈值",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    seed = int(config["training"]["seed"]) + 1000
    features, labels = generate_dataset(config, args.samples, seed)

    checkpoint_path = resolve_project_path(config["paths"]["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"未找到模型 {checkpoint_path}，请先运行 train.py"
        )
    model, metadata = load_checkpoint(checkpoint_path, device)
    probabilities = predict_probabilities(model, features, device)
    network_metrics = classification_metrics(
        labels,
        probabilities,
        float(config["prediction"]["warning_threshold"]),
    )

    sensor_count = len(config["sensors"]["angles_deg"])
    minimum_sensor_ratio = np.min(features[:, :sensor_count], axis=1)
    rule_probabilities = (
        minimum_sensor_ratio < args.rule_threshold
    ).astype(np.float32)
    rule_metrics = classification_metrics(labels, rule_probabilities)

    report = {
        "samples": args.samples,
        "seed": seed,
        "checkpoint_metadata": metadata,
        "network": network_metrics,
        "minimum_distance_rule": {
            "threshold": args.rule_threshold,
            **rule_metrics,
        },
    }
    output_path = resolve_project_path(config["paths"]["evaluation"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"评估结果已保存：{output_path}")


if __name__ == "__main__":
    main()
