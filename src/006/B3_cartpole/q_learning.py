"""把连续 Cart-Pole 状态离散化后的表格 Q-Learning。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class StateDiscretizer:
    """把 4 维连续状态映射为有限表格索引。"""

    def __init__(self, config: dict[str, Any]) -> None:
        environment = config["environment"]
        q_config = config["q_learning"]
        position = float(environment["position_threshold"])
        angle = math.radians(float(environment["angle_threshold_degrees"]))
        velocity = tuple(float(value) for value in q_config["velocity_limits"])
        angular_velocity = tuple(
            float(value) for value in q_config["angular_velocity_limits"]
        )
        self.bins = tuple(int(value) for value in q_config["bins"])
        limits = (
            (-position, position),
            velocity,
            (-angle, angle),
            angular_velocity,
        )
        self.edges = tuple(
            np.linspace(low, high, count + 1, dtype=np.float32)[1:-1]
            for (low, high), count in zip(limits, self.bins, strict=True)
        )

    def encode(self, state: np.ndarray) -> tuple[int, int, int, int]:
        """裁剪极端速度，并返回每一维所在区间。"""

        return tuple(
            int(np.digitize(float(value), edges))
            for value, edges in zip(state, self.edges, strict=True)
        )  # type: ignore[return-value]


class QLearningAgent:
    """使用显式 Q 表和 epsilon-greedy 决策。"""

    def __init__(
        self,
        config: dict[str, Any],
        seed: int,
        q_table: np.ndarray | None = None,
    ) -> None:
        self.discretizer = StateDiscretizer(config)
        expected_shape = (*self.discretizer.bins, 2)
        self.q_table = (
            np.zeros(expected_shape, dtype=np.float32)
            if q_table is None
            else np.asarray(q_table, dtype=np.float32)
        )
        if self.q_table.shape != expected_shape:
            raise ValueError(
                f"Q 表形状应为 {expected_shape}，实际为 {self.q_table.shape}"
            )
        q_config = config["q_learning"]
        self.learning_rate = float(q_config["learning_rate"])
        self.gamma = float(q_config["gamma"])
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """按 Q 表最大值决策；训练时以 epsilon 概率随机探索。"""

        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, 2))
        values = self.q_table[self.discretizer.encode(state)]
        maximum = values.max()
        candidates = np.flatnonzero(values == maximum)
        return int(self.rng.choice(candidates))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """执行一次 Bellman Q-Learning 更新并返回 TD 误差。"""

        state_index = self.discretizer.encode(state)
        next_index = self.discretizer.encode(next_state)
        current = float(self.q_table[state_index][action])
        bootstrap = 0.0 if done else float(self.q_table[next_index].max())
        target = reward + self.gamma * bootstrap
        td_error = target - current
        self.q_table[state_index][action] += self.learning_rate * td_error
        return td_error


def epsilon_by_episode(config: dict[str, Any], episode: int) -> float:
    """按回合数指数降低表格 Q-Learning 探索率。"""

    q_config = config["q_learning"]
    start = float(q_config["epsilon_start"])
    end = float(q_config["epsilon_end"])
    decay = float(q_config["epsilon_decay_episodes"])
    return end + (start - end) * math.exp(-episode / decay)


def save_q_table(
    path: Path,
    agent: QLearningAgent,
    metadata: dict[str, Any],
) -> None:
    """保存 Q 表和 JSON 元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        q_table=agent.q_table,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def load_q_table(
    path: Path,
    config: dict[str, Any],
) -> tuple[QLearningAgent, dict[str, Any]]:
    """加载并校验表格 Q-Learning checkpoint。"""

    with np.load(path, allow_pickle=False) as data:
        q_table = data["q_table"]
        metadata = json.loads(str(data["metadata"].item()))
    expected = str(config["experiment"]["version"])
    if metadata.get("experiment_version") != expected:
        raise RuntimeError(
            f"checkpoint 版本不匹配：需要 {expected}，"
            f"实际为 {metadata.get('experiment_version')!r}"
        )
    return QLearningAgent(config, int(metadata.get("seed", 0)), q_table), metadata
