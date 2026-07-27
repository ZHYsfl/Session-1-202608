"""通过二维仿真自动生成碰撞预测数据集。"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from features import build_feature_vector, sensor_angles_from_degrees
from simulation import World, random_free_state, random_obstacles


def generate_dataset(
    config: dict[str, Any],
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """生成类别近似均衡的特征和二分类标签。"""

    if sample_count < 2:
        raise ValueError("样本数必须至少为 2")

    rng = random.Random(seed)
    window = config["window"]
    robot = config["robot"]
    sensors = config["sensors"]
    prediction = config["prediction"]
    training = config["training"]
    angles = sensor_angles_from_degrees(sensors["angles_deg"])

    positive_target = sample_count // 2
    negative_target = sample_count - positive_target
    positive_count = 0
    negative_count = 0
    features: list[np.ndarray] = []
    labels: list[float] = []
    attempts = 0
    world: World | None = None

    # 使用多个随机世界，防止网络只记住一套障碍物布局。
    while len(features) < sample_count:
        attempts += 1
        if attempts > sample_count * 100:
            raise RuntimeError("无法生成足够的正负碰撞样本")
        if world is None or attempts % 250 == 1:
            world = World(
                width=float(window["width"]),
                height=float(window["height"]),
                obstacles=random_obstacles(
                    rng,
                    float(window["width"]),
                    float(window["height"]),
                    int(training["random_obstacle_count"]),
                ),
            )

        try:
            state = random_free_state(
                rng,
                world,
                float(robot["radius"]),
                float(robot["max_speed"]),
                float(robot["max_angular_speed"]),
            )
        except RuntimeError:
            world = None
            continue

        collision = world.will_collide(
            state,
            float(robot["radius"]),
            float(prediction["horizon_sec"]),
            float(prediction["simulation_step_sec"]),
        )
        if collision and positive_count >= positive_target:
            continue
        if not collision and negative_count >= negative_target:
            continue

        distances = world.sensor_distances(
            state,
            angles,
            float(sensors["max_range"]),
        )
        features.append(
            build_feature_vector(
                distances,
                state,
                float(sensors["max_range"]),
                float(robot["max_speed"]),
                float(robot["max_angular_speed"]),
            )
        )
        labels.append(float(collision))
        if collision:
            positive_count += 1
        else:
            negative_count += 1

    return np.stack(features), np.asarray(labels, dtype=np.float32)


def split_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """按照 70%/15%/15% 划分训练、验证和测试集。"""

    if len(features) != len(labels):
        raise ValueError("特征和标签样本数不一致")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    train_end = int(len(indices) * 0.70)
    validation_end = int(len(indices) * 0.85)

    def subset(selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return features[selected], labels[selected]

    return (
        subset(indices[:train_end]),
        subset(indices[train_end:validation_end]),
        subset(indices[validation_end:]),
    )
