"""汇总正式实验 JSON，并生成 CSV 与论文候选图。"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """按所有行字段的并集写出 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_evaluation_rows(
    output_root: Path,
    split: str,
) -> list[dict[str, Any]]:
    """把每个种子、每种 checkpoint 的评估 JSON 展开成行。"""

    rows: list[dict[str, Any]] = []
    pattern = f"runs/*/seed_*/evaluation_{split}.json"
    for path in sorted(output_root.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        for checkpoint, result in payload["checkpoints"].items():
            rows.append(
                {
                    "variant": metadata["variant"],
                    "label": metadata["label"],
                    "algorithm": metadata["algorithm"],
                    "seed": metadata["seed"],
                    "split": split,
                    "checkpoint": checkpoint,
                    "environment_steps": metadata.get("environment_steps", 0),
                    "wall_clock_seconds": metadata.get("wall_clock_seconds", 0.0),
                    **result,
                }
            )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按方法和 checkpoint 汇总三个训练种子的均值与标准差。"""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), str(row["checkpoint"]))].append(row)

    aggregate: list[dict[str, Any]] = []
    for (variant, checkpoint), group in sorted(groups.items()):
        mean_steps = np.array([row["mean_steps"] for row in group], dtype=float)
        success_rates = np.array([row["success_rate"] for row in group], dtype=float)
        training_steps = np.array(
            [row["environment_steps"] for row in group], dtype=float
        )
        wall_time = np.array([row["wall_clock_seconds"] for row in group], dtype=float)
        ddof = 1 if len(group) > 1 else 0
        aggregate.append(
            {
                "variant": variant,
                "label": group[0]["label"],
                "algorithm": group[0]["algorithm"],
                "checkpoint": checkpoint,
                "training_seeds": len(group),
                "mean_steps_mean": float(mean_steps.mean()),
                "mean_steps_seed_std": float(mean_steps.std(ddof=ddof)),
                "success_rate_mean": float(success_rates.mean()),
                "success_rate_seed_std": float(success_rates.std(ddof=ddof)),
                "environment_steps_mean": float(training_steps.mean()),
                "wall_clock_seconds_mean": float(wall_time.mean()),
            }
        )
    return aggregate


def plot_method_comparison(
    aggregate: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """绘制 best checkpoint（Random 为 policy）的方法比较图。"""

    selected = [row for row in aggregate if row["checkpoint"] in {"best", "policy"}]
    labels = [str(row["label"]) for row in selected]
    mean_steps = [row["mean_steps_mean"] for row in selected]
    mean_steps_error = [row["mean_steps_seed_std"] for row in selected]
    success = [100.0 * row["success_rate_mean"] for row in selected]
    success_error = [100.0 * row["success_rate_seed_std"] for row in selected]
    positions = np.arange(len(selected))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(
        positions,
        mean_steps,
        yerr=mean_steps_error,
        capsize=4,
        color="#4c78a8",
    )
    axes[0].set_ylabel("Mean episode length")
    axes[0].set_ylim(0, 520)
    axes[1].bar(
        positions,
        success,
        yerr=success_error,
        capsize=4,
        color="#59a14f",
    )
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_ylim(0, 105)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_checkpoint_gap(
    aggregate: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """绘制每个学习方法 best 与 final 的平均步数差异。"""

    indexed = {(str(row["variant"]), str(row["checkpoint"])): row for row in aggregate}
    variants = sorted(
        {
            str(row["variant"])
            for row in aggregate
            if row["checkpoint"] == "best" and (str(row["variant"]), "final") in indexed
        }
    )
    labels = [str(indexed[(variant, "best")]["label"]) for variant in variants]
    best = [indexed[(variant, "best")]["mean_steps_mean"] for variant in variants]
    final = [indexed[(variant, "final")]["mean_steps_mean"] for variant in variants]
    positions = np.arange(len(variants))
    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions - width / 2, best, width, label="Best")
    axis.bar(positions + width / 2, final, width, label="Final")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("Mean episode length")
    axis.set_ylim(0, 520)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_sample_efficiency(
    output_root: Path,
    output_path: Path,
) -> None:
    """按验证步数聚合不同训练种子的学习曲线。"""

    curves: dict[str, list[tuple[str, dict[int, float]]]] = defaultdict(list)
    for path in sorted(output_root.glob("runs/*/seed_*/training_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        points = {
            int(item["global_step"]): float(item["mean_steps"])
            for item in payload["validation"]
        }
        curves[str(metadata["variant"])].append((str(metadata["label"]), points))

    figure, axis = plt.subplots(figsize=(10, 6))
    for variant, runs in sorted(curves.items()):
        common_steps = sorted(set.intersection(*(set(points) for _, points in runs)))
        if not common_steps:
            continue
        values = np.array(
            [[points[step] for step in common_steps] for _, points in runs]
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        label = runs[0][0]
        axis.plot(common_steps, mean, label=label)
        axis.fill_between(
            common_steps,
            mean - std,
            mean + std,
            alpha=0.15,
        )
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Validation mean episode length")
    axis.set_ylim(0, 520)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def generate_report(output_root: Path, split: str) -> dict[str, Path]:
    """生成原始结果表、跨种子汇总表和三张图。"""

    rows = collect_evaluation_rows(output_root, split)
    if not rows:
        raise FileNotFoundError(f"没有找到 {split} 评估 JSON")
    aggregate = aggregate_rows(rows)
    report_directory = output_root / "reports" / split
    report_directory.mkdir(parents=True, exist_ok=True)
    raw_csv = report_directory / "evaluation_rows.csv"
    aggregate_csv = report_directory / "aggregate_results.csv"
    comparison = report_directory / "method_comparison.png"
    checkpoint_gap = report_directory / "best_vs_final.png"
    sample_efficiency = report_directory / "sample_efficiency.png"
    write_csv(raw_csv, rows)
    write_csv(aggregate_csv, aggregate)
    plot_method_comparison(aggregate, comparison)
    plot_checkpoint_gap(aggregate, checkpoint_gap)
    plot_sample_efficiency(output_root, sample_efficiency)
    return {
        "raw_csv": raw_csv,
        "aggregate_csv": aggregate_csv,
        "method_comparison": comparison,
        "best_vs_final": checkpoint_gap,
        "sample_efficiency": sample_efficiency,
    }
