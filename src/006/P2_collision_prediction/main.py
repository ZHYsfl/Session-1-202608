"""使用 Pygame 展示局部避障策略的实时轨迹。"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pygame
import torch
from common import (
    DEFAULT_CONFIG,
    load_config,
    resolve_project_path,
    select_device,
)
from controller import direct_action, policy_action, reactive_action
from policy import ActorCritic, load_policy
from rl_env import BatchedLocalAvoidanceEnv

Controller = Callable[[torch.Tensor], torch.Tensor]
BACKGROUND = (18, 22, 30)
GRID = (31, 38, 49)
OBSTACLE = (210, 92, 74)
ROBOT = (75, 180, 245)
GOAL = (75, 210, 130)
RAY = (84, 105, 125)
PATH = (190, 115, 235)
TEXT = (230, 235, 240)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--controller",
        choices=("direct", "reactive", "mlp", "cnn"),
        default="reactive",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--noise", action="store_true", help="在演示中启用测距噪声")
    return parser


def build_controller(
    name: str,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Controller, ActorCritic | None]:
    """创建指定控制器；学习控制器需要已有 checkpoint。"""

    sensor_count = int(config["sensors"]["count"])
    if name == "direct":
        return lambda obs: direct_action(obs, sensor_count), None
    if name == "reactive":
        return lambda obs: reactive_action(obs, sensor_count), None
    checkpoint = resolve_project_path(str(config["paths"][f"{name}_checkpoint"]))
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"找不到 {checkpoint}；请先运行 train.py --model {name}"
        )
    model, _ = load_policy(
        checkpoint,
        config["model"],
        device,
        expected_version=str(config["experiment"]["version"]),
    )
    return lambda obs: policy_action(model, obs), model


def draw_scene(
    screen: pygame.Surface,
    snapshot: dict[str, object],
    env: BatchedLocalAvoidanceEnv,
    path_trace: list[tuple[float, float]],
) -> None:
    """绘制障碍物、射线、目标、轨迹和机器人。"""

    screen.fill(BACKGROUND)
    for x in range(0, int(env.width), 40):
        pygame.draw.line(screen, GRID, (x, 0), (x, int(env.height)), 1)
    for y in range(0, int(env.height), 40):
        pygame.draw.line(screen, GRID, (0, y), (int(env.width), y), 1)

    for rect in snapshot["obstacles"]:  # type: ignore[union-attr]
        pygame.draw.rect(screen, OBSTACLE, pygame.Rect(*rect))

    position = snapshot["position"]  # type: ignore[assignment]
    heading = float(snapshot["heading"])
    goal = snapshot["goal"]  # type: ignore[assignment]
    rays = snapshot["rays"]  # type: ignore[assignment]
    for index, distance in enumerate(rays):
        angle = heading + index * 2.0 * math.pi / env.sensor_count
        end = (
            position[0] + math.cos(angle) * distance,
            position[1] + math.sin(angle) * distance,
        )
        pygame.draw.line(screen, RAY, position, end, 1)

    if len(path_trace) >= 2:
        pygame.draw.lines(screen, PATH, False, path_trace, 3)
    pygame.draw.circle(
        screen, GOAL, (round(goal[0]), round(goal[1])), round(env.goal_radius), 3
    )
    pygame.draw.circle(
        screen,
        ROBOT,
        (round(position[0]), round(position[1])),
        round(env.radius),
    )
    nose = (
        position[0] + math.cos(heading) * env.radius,
        position[1] + math.sin(heading) * env.radius,
    )
    pygame.draw.line(screen, BACKGROUND, position, nose, 3)


def draw_status(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: BatchedLocalAvoidanceEnv,
    controller_name: str,
    status: str,
    paused: bool,
    noisy: bool,
    snapshot: dict[str, object],
) -> None:
    """在场地右侧绘制运行状态和快捷键。"""

    x = int(env.width) + 18
    lines = (
        "LOCAL PPO PLANNER",
        f"Controller: {controller_name.upper()}",
        f"Status: {status}",
        f"Paused: {'YES' if paused else 'NO'}",
        f"Noise: {'ON' if noisy else 'OFF'}",
        f"Stage: {snapshot['stage']}",
        f"Gates/Bends: {snapshot['gate_count']}/{snapshot['bend_count']}",
        f"Step: {snapshot['step_count']}/{env.max_steps}",
        f"Path: {float(snapshot['path_length']):.1f} px",
        f"Device: {env.device}",
        "",
        "R: new local task",
        "SPACE: pause",
        "ESC: quit",
    )
    for index, line in enumerate(lines):
        surface = font.render(line, True, TEXT)
        screen.blit(surface, (x, 20 + index * 25))


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    controller, model = build_controller(args.controller, config, device)
    _ = model
    env = BatchedLocalAvoidanceEnv(
        config,
        1,
        device,
        int(config["evaluation"]["seed"]),
        training_noise=args.noise,
        auto_reset=False,
    )
    env.set_curriculum_stage(2)
    observation = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((int(env.width) + 300, int(env.height)))
    pygame.display.set_caption("PyTorch PPO Local Obstacle Avoidance")
    font = pygame.font.Font(None, 23)
    clock = pygame.time.Clock()
    accumulator = 0.0
    running = True
    paused = False
    status = "RUNNING"
    snapshot = env.snapshot()
    path_trace = [snapshot["position"]]  # type: ignore[list-item]

    while running:
        frame_dt = min(clock.tick(int(config["window"]["fps"])) / 1000.0, 0.05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    observation = env.reset()
                    status = "RUNNING"
                    accumulator = 0.0
                    snapshot = env.snapshot()
                    path_trace = [snapshot["position"]]  # type: ignore[list-item]

        if not paused and status == "RUNNING":
            accumulator += frame_dt
            while accumulator >= env.dt and status == "RUNNING":
                with torch.no_grad():
                    action = controller(observation)
                observation, _, done, info = env.step(action)
                accumulator -= env.dt
                snapshot = env.snapshot()
                path_trace.append(snapshot["position"])  # type: ignore[arg-type]
                if bool(done[0]):
                    if bool(info["success"][0]):
                        status = "GOAL REACHED"
                    elif bool(info["collision"][0]):
                        status = "COLLISION"
                    else:
                        status = "TIMEOUT"

        snapshot = env.snapshot()
        draw_scene(screen, snapshot, env, path_trace)
        draw_status(
            screen,
            font,
            env,
            args.controller,
            status,
            paused,
            args.noise,
            snapshot,
        )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
