"""碰撞二分类评价指标。"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def classification_metrics(
    labels: Iterable[float],
    probabilities: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """计算准确率、精确率、召回率、F1 和混淆矩阵计数。"""

    truth = np.asarray(list(labels), dtype=np.int64)
    probability = np.asarray(list(probabilities), dtype=np.float64)
    if truth.size == 0 or truth.size != probability.size:
        raise ValueError("标签与预测必须非空且长度一致")

    prediction = (probability >= threshold).astype(np.int64)
    true_positive = int(np.sum((prediction == 1) & (truth == 1)))
    true_negative = int(np.sum((prediction == 0) & (truth == 0)))
    false_positive = int(np.sum((prediction == 1) & (truth == 0)))
    false_negative = int(np.sum((prediction == 0) & (truth == 1)))

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
