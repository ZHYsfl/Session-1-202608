"""简单的途经点导航控制器。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from simulation import RobotState, distance_to_goal, normalize_angle


@dataclass
class WaypointNavigator:
    """记录自动模式当前正在前往的途经点序号。"""

    index: int = 0
    emergency_stop: bool = False


def goal_navigation_control(
    state: RobotState,
    target: tuple[float, float],
    robot_config: dict[str, Any],
    navigation_config: dict[str, Any],
) -> RobotState:
    """使用比例转向控制机器人朝单个目标点行驶。"""

    max_speed = float(robot_config["max_speed"])
    max_angular_speed = float(robot_config["max_angular_speed"])
    target_heading = math.atan2(target[1] - state.y, target[0] - state.x)
    heading_error = normalize_angle(target_heading - state.heading)
    angular_speed = min(
        max(heading_error * 2.0, -max_angular_speed),
        max_angular_speed,
    )

    # 转角较大时降低速度，转向稳定后再恢复巡航速度。
    turn_ratio = min(abs(heading_error) / math.pi, 1.0)
    cruise_ratio = float(navigation_config["cruise_speed_ratio"])
    sharp_turn_ratio = float(navigation_config["sharp_turn_speed_ratio"])
    speed_ratio = cruise_ratio - (
        cruise_ratio - sharp_turn_ratio
    ) * turn_ratio
    return replace(
        state,
        speed=max_speed * speed_ratio,
        angular_speed=angular_speed,
    )


def waypoint_navigation_control(
    state: RobotState,
    route: tuple[tuple[float, float], ...],
    robot_config: dict[str, Any],
    navigation_config: dict[str, Any],
    navigator: WaypointNavigator,
) -> tuple[RobotState, WaypointNavigator]:
    """依次跟随途经点，最后一个点应为实验终点。"""

    if not route:
        raise ValueError("自动导航路线不能为空")
    navigator.index = min(navigator.index, len(route) - 1)
    waypoint_radius = float(navigation_config["waypoint_radius"])

    # 进入当前途经点范围后立即切换到下一个点。
    while (
        navigator.index < len(route) - 1
        and distance_to_goal(state, route[navigator.index])
        <= waypoint_radius
    ):
        navigator.index += 1

    navigator.emergency_stop = False
    controlled = goal_navigation_control(
        state,
        route[navigator.index],
        robot_config,
        navigation_config,
    )
    return controlled, navigator
