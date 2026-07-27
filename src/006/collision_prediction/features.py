"""神经网络输入特征构造。"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from simulation import RobotState


def sensor_angles_from_degrees(values: Iterable[float]) -> tuple[float, ...]:
    """将配置中的角度值转换为弧度。"""

    return tuple(math.radians(float(value)) for value in values)


def build_feature_vector(
    sensor_distances: Iterable[float],
    state: RobotState,
    max_range: float,
    max_speed: float,
    max_angular_speed: float,
) -> np.ndarray:
    """构造归一化特征：[多路距离, 线速度, 角速度]。"""

    if max_range <= 0 or max_speed <= 0 or max_angular_speed <= 0:
        raise ValueError("归一化尺度必须为正数")
    normalized_sensors = [
        min(max(float(distance) / max_range, 0.0), 1.0)
        for distance in sensor_distances
    ]
    speed = min(max(state.speed / max_speed, -1.0), 1.0)
    angular_speed = min(
        max(state.angular_speed / max_angular_speed, -1.0),
        1.0,
    )
    return np.asarray(
        [*normalized_sensors, speed, angular_speed],
        dtype=np.float32,
    )
