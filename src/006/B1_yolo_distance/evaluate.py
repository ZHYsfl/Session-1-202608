"""根据真实距离与预测距离 CSV 计算测距评价指标。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def calculate_metrics(
    pairs: Iterable[tuple[float, float]],
) -> dict[str, float | int]:
    """计算样本数、MAE、RMSE、MAPE 和有符号系统偏差。"""

    rows = list(pairs)
    if not rows:
        raise ValueError("at least one result row is required")
    if any(truth <= 0 for truth, _ in rows):
        raise ValueError("ground-truth distances must be positive")

    # 误差为“预测值 - 真实值”；正偏差表示整体高估距离。
    errors = [prediction - truth for truth, prediction in rows]
    absolute_errors = [abs(error) for error in errors]
    count = len(rows)
    return {
        "count": count,
        "mae_m": sum(absolute_errors) / count,
        "rmse_m": math.sqrt(sum(error**2 for error in errors) / count),
        "mape_percent": (
            sum(abs(error) / truth for (truth, _), error in zip(rows, errors))
            / count
            * 100
        ),
        "bias_m": sum(errors) / count,
    }


def load_pairs(csv_path: Path) -> list[tuple[float, float]]:
    """从 CSV 中读取（真实距离、预测距离）数据对。"""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ground_truth_m", "predicted_m"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain ground_truth_m and predicted_m")
        return [
            (float(row["ground_truth_m"]), float(row["predicted_m"]))
            for row in reader
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    metrics = calculate_metrics(load_pairs(args.csv))
    rendered = json.dumps(
        {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in metrics.items()
        },
        indent=2,
    )
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
