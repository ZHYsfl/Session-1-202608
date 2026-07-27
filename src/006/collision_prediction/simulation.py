"""扫地机器人二维运动、传感器和碰撞标签仿真。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Iterable

from geometry import (
    Rect,
    circle_intersects_rect,
    circle_outside_bounds,
    expand_rect,
    raycast_distance,
    rects_overlap,
    segment_intersects_rect,
)


@dataclass(frozen=True)
class RobotState:
    """机器人位姿与运动状态，角度和角速度均使用弧度。"""

    x: float
    y: float
    heading: float
    speed: float
    angular_speed: float


@dataclass(frozen=True)
class World:
    """包含场地边界和静态矩形障碍物的二维世界。"""

    width: float
    height: float
    obstacles: tuple[Rect, ...]

    def collides(self, state: RobotState, radius: float) -> bool:
        """判断给定机器人状态是否发生碰撞。"""

        if circle_outside_bounds(
            state.x,
            state.y,
            radius,
            self.width,
            self.height,
        ):
            return True
        return any(
            circle_intersects_rect(state.x, state.y, radius, obstacle)
            for obstacle in self.obstacles
        )

    def sensor_distances(
        self,
        state: RobotState,
        sensor_angles_rad: Iterable[float],
        max_range: float,
    ) -> list[float]:
        """计算所有测距射线到最近障碍物的距离。"""

        return [
            raycast_distance(
                state.x,
                state.y,
                state.heading + relative_angle,
                self.obstacles,
                self.width,
                self.height,
                max_range,
            )
            for relative_angle in sensor_angles_rad
        ]

    def will_collide(
        self,
        state: RobotState,
        radius: float,
        horizon_sec: float,
        step_sec: float,
    ) -> bool:
        """保持当前速度，预测指定时间窗口内是否会发生碰撞。"""

        if horizon_sec <= 0 or step_sec <= 0:
            raise ValueError("预测时间和仿真步长必须为正数")
        simulated = state
        steps = math.ceil(horizon_sec / step_sec)
        for _ in range(steps):
            simulated = advance_state(simulated, step_sec)
            if self.collides(simulated, radius):
                return True
        return False


def advance_state(state: RobotState, delta_time: float) -> RobotState:
    """使用一阶运动学模型推进机器人状态。"""

    heading = normalize_angle(
        state.heading + state.angular_speed * delta_time
    )
    return replace(
        state,
        x=state.x + math.cos(heading) * state.speed * delta_time,
        y=state.y + math.sin(heading) * state.speed * delta_time,
        heading=heading,
    )


def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi)。"""

    return (angle + math.pi) % (2 * math.pi) - math.pi


def distance_to_goal(
    state: RobotState,
    goal: tuple[float, float],
) -> float:
    """计算机器人圆心到目标点的欧氏距离。"""

    return math.hypot(state.x - goal[0], state.y - goal[1])


def reached_goal(
    state: RobotState,
    goal: tuple[float, float],
    goal_radius: float,
) -> bool:
    """判断机器人圆心是否进入目标区域。"""

    if goal_radius <= 0:
        raise ValueError("目标区域半径必须为正数")
    return distance_to_goal(state, goal) <= goal_radius


def random_obstacles(
    rng: random.Random,
    width: float,
    height: float,
    count: int,
) -> tuple[Rect, ...]:
    """生成用于训练的随机矩形障碍物。"""

    obstacles: list[Rect] = []
    for _ in range(count):
        obstacle_width = rng.uniform(55, 155)
        obstacle_height = rng.uniform(45, 145)
        x = rng.uniform(25, width - obstacle_width - 25)
        y = rng.uniform(25, height - obstacle_height - 25)
        obstacles.append((x, y, obstacle_width, obstacle_height))
    return tuple(obstacles)


def reasonable_random_obstacles(
    rng: random.Random,
    width: float,
    height: float,
    count: int,
    route: tuple[tuple[float, float], ...],
    min_size: tuple[float, float],
    max_size: tuple[float, float],
    boundary_margin: float,
    obstacle_spacing: float,
    route_clearance: float,
    max_attempts: int,
) -> tuple[Rect, ...]:
    """生成不重叠且不侵入既定路线安全走廊的随机障碍物。"""

    if count < 0 or len(route) < 2:
        raise ValueError("障碍物数量不能为负，路线至少需要两个点")
    if any(value <= 0 for value in (*min_size, *max_size)):
        raise ValueError("障碍物尺寸必须为正数")
    if min_size[0] > max_size[0] or min_size[1] > max_size[1]:
        raise ValueError("最小障碍物尺寸不能大于最大尺寸")

    obstacles: list[Rect] = []
    attempts = 0
    while len(obstacles) < count and attempts < max_attempts:
        attempts += 1
        obstacle_width = rng.uniform(min_size[0], max_size[0])
        obstacle_height = rng.uniform(min_size[1], max_size[1])
        available_width = width - 2 * boundary_margin - obstacle_width
        available_height = height - 2 * boundary_margin - obstacle_height
        if available_width <= 0 or available_height <= 0:
            raise ValueError("场地尺寸不足以放置配置的障碍物")

        candidate: Rect = (
            rng.uniform(boundary_margin, boundary_margin + available_width),
            rng.uniform(boundary_margin, boundary_margin + available_height),
            obstacle_width,
            obstacle_height,
        )
        if any(
            rects_overlap(candidate, existing, obstacle_spacing)
            for existing in obstacles
        ):
            continue

        protected_candidate = expand_rect(candidate, route_clearance)
        if any(
            segment_intersects_rect(start, end, protected_candidate)
            for start, end in zip(route[:-1], route[1:], strict=True)
        ):
            continue
        obstacles.append(candidate)

    if len(obstacles) != count:
        raise RuntimeError(
            f"在 {max_attempts} 次尝试内只能生成 "
            f"{len(obstacles)}/{count} 个合理障碍物"
        )
    return tuple(obstacles)


def random_free_state(
    rng: random.Random,
    world: World,
    radius: float,
    max_speed: float,
    max_angular_speed: float,
    maximum_attempts: int = 500,
) -> RobotState:
    """随机采样一个不与障碍物重叠的运动状态。"""

    for _ in range(maximum_attempts):
        state = RobotState(
            x=rng.uniform(radius + 2, world.width - radius - 2),
            y=rng.uniform(radius + 2, world.height - radius - 2),
            heading=rng.uniform(-math.pi, math.pi),
            speed=rng.uniform(0.2 * max_speed, max_speed),
            angular_speed=rng.uniform(
                -max_angular_speed,
                max_angular_speed,
            ),
        )
        if not world.collides(state, radius):
            return state
    raise RuntimeError("无法在随机世界中采样无碰撞机器人状态")
