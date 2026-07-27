"""显示扫地机器人、多角度测距和神经网络碰撞概率。"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import torch

from controller import (
    WaypointNavigator,
    waypoint_navigation_control,
)
from features import build_feature_vector, sensor_angles_from_degrees
from network import CollisionMLP, load_checkpoint, predict_probabilities
from planner import generate_partitioned_layout
from simulation import (
    RobotState,
    World,
    advance_state,
    distance_to_goal,
    reached_goal,
)
from train import load_config, resolve_project_path, select_device

PROJECT_DIR = Path(__file__).resolve().parent

BACKGROUND = (18, 23, 31)
GRID = (35, 43, 55)
OBSTACLE = (105, 116, 132)
ROBOT_SAFE = (45, 205, 125)
ROBOT_WARNING = (239, 68, 68)
SENSOR_SAFE = (60, 150, 220)
SENSOR_NEAR = (255, 190, 60)
TEXT = (235, 240, 248)
START_COLOR = (100, 210, 255)
GOAL_COLOR = (255, 205, 70)
PATH_COLOR = (125, 105, 210)
PLANNED_PATH_COLOR = (65, 90, 125)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "config.yaml",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def build_world_and_route(
    config: dict[str, Any],
    start: tuple[float, float],
    goal: tuple[float, float],
    rng: random.Random,
) -> tuple[World, tuple[tuple[float, float], ...]]:
    """生成挑战地形及 A* 路线，或使用固定地形作为回退。"""

    window = config["window"]
    terrain = config["terrain"]
    if bool(terrain["randomize"]):
        gap_size = terrain["gap_size"]
        barrier_thickness = terrain["barrier_thickness"]
        layout = generate_partitioned_layout(
            rng=rng,
            width=float(window["width"]),
            height=float(window["height"]),
            barrier_count=int(terrain["barrier_count"]),
            start=start,
            goal=goal,
            robot_radius=float(config["robot"]["radius"]),
            gap_size=(float(gap_size[0]), float(gap_size[1])),
            barrier_thickness=(
                float(barrier_thickness[0]),
                float(barrier_thickness[1]),
            ),
            x_jitter=float(terrain["x_jitter"]),
            layout_attempts=int(terrain["layout_attempts"]),
            grid_size=float(terrain["grid_size"]),
            extra_clearance=float(terrain["extra_clearance"]),
            minimum_path_ratio=float(terrain["minimum_path_ratio"]),
            minimum_route_points=int(terrain["minimum_route_points"]),
        )
        print(
            "Generated terrain: "
            f"{len(layout.world.obstacles)} obstacles, "
            f"path ratio={layout.path_ratio:.2f}, "
            f"direct blockers={layout.direct_blockers}"
        )
        # A* 路线包含起点；导航器只需要后续途经点和终点。
        return layout.world, layout.route[1:]

    episode = config["episode"]
    route = tuple(
        (float(point[0]), float(point[1]))
        for point in episode.get("waypoints", [])
    ) + (goal,)
    world = World(
        width=float(window["width"]),
        height=float(window["height"]),
        obstacles=tuple(
            tuple(float(value) for value in obstacle)
            for obstacle in config["obstacles"]
        ),
    )
    return world, route


def initial_state(episode_config: dict[str, Any]) -> RobotState:
    """根据配置文件返回一轮实验的初始状态。"""

    start = episode_config["start"]
    return RobotState(
        x=float(start[0]),
        y=float(start[1]),
        heading=math.radians(float(start[2])),
        speed=55.0,
        angular_speed=0.0,
    )


def model_probability(
    model: CollisionMLP | None,
    feature_vector: np.ndarray,
    device: torch.device,
) -> float | None:
    """模型存在时计算单帧碰撞概率，否则返回 None。"""

    if model is None:
        return None
    return float(
        predict_probabilities(model, feature_vector[None, :], device)[0]
    )


def update_manual_controls(
    state: RobotState,
    keys: Any,
    delta_time: float,
    robot_config: dict[str, Any],
) -> RobotState:
    """根据方向键或 WASD 更新线速度和角速度。"""

    acceleration = float(robot_config["acceleration"])
    max_speed = float(robot_config["max_speed"])
    max_angular_speed = float(robot_config["max_angular_speed"])
    speed = state.speed
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        speed = min(max_speed, speed + acceleration * delta_time)
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        speed = max(0.0, speed - acceleration * delta_time)

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        angular_speed = -max_angular_speed
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        angular_speed = max_angular_speed
    else:
        angular_speed = 0.0
    if keys[pygame.K_SPACE]:
        speed = 0.0
        angular_speed = 0.0
    return replace(state, speed=speed, angular_speed=angular_speed)


def draw_grid(screen: pygame.Surface, width: int, height: int) -> None:
    """绘制便于观察运动的背景网格。"""

    for x in range(0, width, 40):
        pygame.draw.line(screen, GRID, (x, 0), (x, height), 1)
    for y in range(0, height, 40):
        pygame.draw.line(screen, GRID, (0, y), (width, y), 1)


def draw_world(
    screen: pygame.Surface,
    world: World,
    state: RobotState,
    distances: list[float],
    sensor_angles: tuple[float, ...],
    max_range: float,
    radius: float,
    warning: bool,
    start: tuple[float, float],
    goal: tuple[float, float],
    goal_radius: float,
    path_trace: list[tuple[float, float]],
    planned_route: tuple[tuple[float, float], ...],
) -> None:
    """绘制起终点、路径、障碍物、传感器和机器人。"""

    if len(planned_route) >= 2:
        pygame.draw.lines(
            screen,
            PLANNED_PATH_COLOR,
            False,
            planned_route,
            2,
        )
        for waypoint in planned_route[1:-1]:
            pygame.draw.circle(
                screen,
                PLANNED_PATH_COLOR,
                (round(waypoint[0]), round(waypoint[1])),
                5,
            )
    if len(path_trace) >= 2:
        pygame.draw.lines(screen, PATH_COLOR, False, path_trace, 2)
    pygame.draw.circle(
        screen,
        START_COLOR,
        (round(start[0]), round(start[1])),
        10,
        2,
    )
    pygame.draw.circle(
        screen,
        GOAL_COLOR,
        (round(goal[0]), round(goal[1])),
        round(goal_radius),
        3,
    )
    pygame.draw.circle(
        screen,
        GOAL_COLOR,
        (round(goal[0]), round(goal[1])),
        5,
    )

    for obstacle in world.obstacles:
        pygame.draw.rect(
            screen,
            OBSTACLE,
            pygame.Rect(*(round(value) for value in obstacle)),
            border_radius=4,
        )

    for distance, relative_angle in zip(
        distances,
        sensor_angles,
        strict=True,
    ):
        angle = state.heading + relative_angle
        end = (
            state.x + math.cos(angle) * distance,
            state.y + math.sin(angle) * distance,
        )
        color = SENSOR_NEAR if distance < max_range * 0.35 else SENSOR_SAFE
        pygame.draw.aaline(screen, color, (state.x, state.y), end)
        pygame.draw.circle(screen, color, end, 3)

    robot_color = ROBOT_WARNING if warning else ROBOT_SAFE
    center = (round(state.x), round(state.y))
    pygame.draw.circle(screen, robot_color, center, round(radius))
    heading_end = (
        state.x + math.cos(state.heading) * radius,
        state.y + math.sin(state.heading) * radius,
    )
    pygame.draw.line(screen, TEXT, center, heading_end, 3)


def draw_status(
    screen: pygame.Surface,
    font: pygame.font.Font,
    probability: float | None,
    state: RobotState,
    automatic: bool,
    model_loaded: bool,
    collision_flash: float,
    episode_status: str,
    elapsed_sec: float,
    path_length: float,
    goal_distance: float,
    waypoint_index: int,
    waypoint_count: int,
    emergency_stop: bool,
    navigation_mode: str,
) -> None:
    """在窗口左上角显示模型和控制状态。"""

    probability_text = (
        "N/A" if probability is None else f"{probability * 100:5.1f}%"
    )
    lines = [
        f"Episode: {episode_status}",
        f"Collision probability: {probability_text}",
        f"Goal distance: {goal_distance:6.1f} px",
        f"Elapsed / path: {elapsed_sec:5.1f} s / {path_length:6.1f} px",
        f"Speed: {state.speed:5.1f} px/s",
        f"Mode: {'AUTO' if automatic else 'MANUAL'}",
        f"Navigation: {navigation_mode}",
        "Prediction: DISPLAY ONLY",
        f"Waypoint: {waypoint_index + 1}/{waypoint_count}",
        f"Emergency stop: {'YES' if emergency_stop else 'NO'}",
        f"Model: {'LOADED' if model_loaded else 'NOT TRAINED'}",
        "Controls: TAB auto, P planner, R new terrain, ESC quit",
    ]
    if collision_flash > 0 and episode_status == "COLLISION":
        lines.insert(0, "COLLISION!")
    for index, text in enumerate(lines):
        color = ROBOT_WARNING if index == 0 and collision_flash > 0 else TEXT
        surface = font.render(text, True, color)
        screen.blit(surface, (14, 12 + index * 24))


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    window = config["window"]
    robot_config = config["robot"]
    sensor_config = config["sensors"]
    prediction_config = config["prediction"]
    navigation_config = config["navigation"]
    episode_config = config["episode"]
    sensor_angles = sensor_angles_from_degrees(sensor_config["angles_deg"])
    max_range = float(sensor_config["max_range"])
    radius = float(robot_config["radius"])
    warning_threshold = float(prediction_config["warning_threshold"])
    start = (
        float(episode_config["start"][0]),
        float(episode_config["start"][1]),
    )
    goal = (
        float(episode_config["goal"][0]),
        float(episode_config["goal"][1]),
    )
    goal_radius = float(episode_config["goal_radius"])
    max_duration_sec = float(episode_config["max_duration_sec"])

    terrain_seed = config["terrain"].get("seed")
    terrain_rng = random.Random(
        None if terrain_seed is None else int(terrain_seed)
    )
    world, route = build_world_and_route(
        config,
        start,
        goal,
        terrain_rng,
    )
    configured_initial_state = initial_state(episode_config)
    if world.collides(configured_initial_state, radius):
        raise ValueError("配置的起点与边界或障碍物发生重叠")
    if not (
        goal_radius <= goal[0] <= world.width - goal_radius
        and goal_radius <= goal[1] <= world.height - goal_radius
    ):
        raise ValueError("配置的终点超出场地有效区域")
    device = select_device(args.device)
    checkpoint_path = resolve_project_path(config["paths"]["checkpoint"])
    model: CollisionMLP | None = None
    if checkpoint_path.exists():
        model, _ = load_checkpoint(checkpoint_path, device)

    pygame.init()
    screen = pygame.display.set_mode(
        (int(window["width"]), int(window["height"]))
    )
    pygame.display.set_caption("Robot Collision Prediction")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    state = configured_initial_state
    automatic = False
    use_astar = True
    collision_flash = 0.0
    episode_status = "RUNNING"
    elapsed_sec = 0.0
    path_length = 0.0
    path_trace: list[tuple[float, float]] = [(state.x, state.y)]
    navigator = WaypointNavigator()
    running = True
    while running:
        delta_time = min(clock.tick(int(window["fps"])) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    automatic = not automatic
                    navigator = WaypointNavigator()
                elif event.key == pygame.K_p:
                    use_astar = not use_astar
                    navigator = WaypointNavigator()
                    state = initial_state(episode_config)
                    episode_status = "RUNNING"
                    elapsed_sec = 0.0
                    path_length = 0.0
                    path_trace = [(state.x, state.y)]
                elif event.key == pygame.K_r:
                    world, route = build_world_and_route(
                        config,
                        start,
                        goal,
                        terrain_rng,
                    )
                    state = initial_state(episode_config)
                    collision_flash = 0.0
                    episode_status = "RUNNING"
                    elapsed_sec = 0.0
                    path_length = 0.0
                    path_trace = [(state.x, state.y)]
                    navigator = WaypointNavigator()

        distances = world.sensor_distances(
            state,
            sensor_angles,
            max_range,
        )
        probability: float | None = None

        if episode_status == "RUNNING":
            if automatic:
                active_route = route if use_astar else (goal,)
                state, navigator = waypoint_navigation_control(
                    state,
                    active_route,
                    robot_config,
                    navigation_config,
                    navigator,
                )
            else:
                state = update_manual_controls(
                    state,
                    pygame.key.get_pressed(),
                    delta_time,
                    robot_config,
                )

            # 神经网络只负责预测和显示，不再参与自动控制状态切换。
            feature_vector = build_feature_vector(
                distances,
                state,
                max_range,
                float(robot_config["max_speed"]),
                float(robot_config["max_angular_speed"]),
            )
            probability = model_probability(
                model,
                feature_vector,
                device,
            )

            previous_position = (state.x, state.y)
            candidate = advance_state(state, delta_time)
            if world.collides(candidate, radius):
                if automatic:
                    # 默认路线经过离线验证；如果配置被改坏，则拒绝危险
                    # 位移并停止，而不是让机器人穿入障碍物。
                    navigator.emergency_stop = True
                    state = replace(
                        state,
                        speed=0.0,
                        angular_speed=0.0,
                    )
                else:
                    state = replace(
                        state,
                        speed=0.0,
                        angular_speed=0.0,
                    )
                    collision_flash = 0.45
                    episode_status = "COLLISION"
            else:
                state = candidate
                movement = math.hypot(
                    state.x - previous_position[0],
                    state.y - previous_position[1],
                )
                path_length += movement
                if not path_trace or math.hypot(
                    state.x - path_trace[-1][0],
                    state.y - path_trace[-1][1],
                ) >= 4:
                    path_trace.append((state.x, state.y))

            elapsed_sec += delta_time
            if reached_goal(state, goal, goal_radius):
                state = replace(state, speed=0.0, angular_speed=0.0)
                episode_status = "GOAL REACHED"
            elif elapsed_sec >= max_duration_sec:
                state = replace(state, speed=0.0, angular_speed=0.0)
                episode_status = "TIMEOUT"
        collision_flash = max(0.0, collision_flash - delta_time)

        # 移动后重新计算传感器，使绘制内容与画面中的位姿一致。
        distances = world.sensor_distances(
            state,
            sensor_angles,
            max_range,
        )
        feature_vector = build_feature_vector(
            distances,
            state,
            max_range,
            float(robot_config["max_speed"]),
            float(robot_config["max_angular_speed"]),
        )
        probability = model_probability(model, feature_vector, device)
        warning = probability is not None and probability >= warning_threshold

        screen.fill(BACKGROUND)
        draw_grid(screen, int(world.width), int(world.height))
        draw_world(
            screen,
            world,
            state,
            distances,
            sensor_angles,
            max_range,
            radius,
            warning,
            start,
            goal,
            goal_radius,
            path_trace,
            (start, *(route if use_astar else (goal,))),
        )
        active_route = route if use_astar else (goal,)
        draw_status(
            screen,
            font,
            probability,
            state,
            automatic,
            model is not None,
            collision_flash,
            episode_status,
            elapsed_sec,
            path_length,
            distance_to_goal(state, goal),
            navigator.index,
            len(active_route),
            navigator.emergency_stop,
            "ASTAR" if use_astar else "DIRECT",
        )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
