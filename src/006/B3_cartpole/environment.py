"""不依赖 Gym 的经典 Cart-Pole 动力学环境。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class CartPoleEnv:
    """使用离散左右动作控制连续状态 Cart-Pole。"""

    observation_size = 4
    action_size = 2

    def __init__(self, config: dict[str, Any], seed: int = 0) -> None:
        environment = config["environment"]
        self.gravity = float(environment["gravity"])
        self.cart_mass = float(environment["cart_mass"])
        self.pole_mass = float(environment["pole_mass"])
        self.total_mass = self.cart_mass + self.pole_mass
        self.half_pole_length = float(environment["half_pole_length"])
        self.pole_mass_length = self.pole_mass * self.half_pole_length
        self.force_magnitude = float(environment["force_magnitude"])
        self.time_step = float(environment["time_step"])
        self.position_threshold = float(environment["position_threshold"])
        self.angle_threshold = math.radians(
            float(environment["angle_threshold_degrees"])
        )
        self.max_steps = int(environment["max_steps"])
        self.alive_reward = float(environment["alive_reward"])
        self.failure_penalty = float(environment["failure_penalty"])
        self.success_bonus = float(environment["success_bonus"])
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(self.observation_size, dtype=np.float32)
        self.step_count = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """把状态重置为直立点附近的小随机扰动。"""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float | int]]:
        """执行左右施力，并返回 next_state、reward、terminated、truncated。"""

        if action not in (0, 1):
            raise ValueError("action 必须为 0（左）或 1（右）")
        position, velocity, angle, angular_velocity = (
            float(value) for value in self.state
        )
        force = self.force_magnitude if action == 1 else -self.force_magnitude
        cosine = math.cos(angle)
        sine = math.sin(angle)
        temporary = (
            force + self.pole_mass_length * angular_velocity**2 * sine
        ) / self.total_mass
        angular_acceleration = (self.gravity * sine - cosine * temporary) / (
            self.half_pole_length
            * (4.0 / 3.0 - self.pole_mass * cosine**2 / self.total_mass)
        )
        acceleration = (
            temporary
            - self.pole_mass_length * angular_acceleration * cosine / self.total_mass
        )

        # 半隐式欧拉积分比直接更新位置更稳定，仍保持实现足够直观。
        velocity += self.time_step * acceleration
        position += self.time_step * velocity
        angular_velocity += self.time_step * angular_acceleration
        angle += self.time_step * angular_velocity
        self.state = np.array(
            (position, velocity, angle, angular_velocity),
            dtype=np.float32,
        )
        self.step_count += 1

        terminated = (
            abs(position) > self.position_threshold or abs(angle) > self.angle_threshold
        )
        truncated = self.step_count >= self.max_steps and not terminated
        reward = self.alive_reward
        if terminated:
            reward += self.failure_penalty
        elif truncated:
            reward += self.success_bonus
        info: dict[str, float | int] = {
            "step": self.step_count,
            "position": position,
            "angle_degrees": math.degrees(angle),
        }
        return self.state.copy(), reward, terminated, truncated, info
