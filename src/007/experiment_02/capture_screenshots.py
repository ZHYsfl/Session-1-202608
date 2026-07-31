"""Capture screenshots of the robot environment for the paper."""
import math
import os
import sys
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame
import torch
from stable_baselines3 import DQN

from robot_env_v3 import (
    EMERGENCY_DISTANCE,
    ROBOT_RADIUS,
    SENSOR_ANGLES,
    SENSOR_RANGE,
    WARNING_DISTANCE,
    WALL_THICKNESS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    RobotReflectionEnv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_PATH = (
    PROJECT_ROOT
    / "models"
    / "dqn_reflection_v3_safe"
    / "best_policy.pt"
)


def load_v3_policy(env):
    """Load the V3 DQN policy."""
    if not POLICY_PATH.exists():
        # Try experiment-local path
        alt = Path(__file__).resolve().parent / "models" / "dqn_reflection_v3_safe" / "best_policy.pt"
        if alt.exists():
            payload = torch.load(alt, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"Cannot find policy at {POLICY_PATH} or {alt}")
    else:
        payload = torch.load(POLICY_PATH, map_location="cpu", weights_only=True)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={"net_arch": [64, 64]},
        device="cpu",
        verbose=0,
        seed=2026,
    )
    model.policy.load_state_dict(payload["policy_state_dict"], strict=True)
    model.policy.set_training_mode(False)
    return model


def draw_scene_to_surface(env, action, info, fonts, reflection_flash=0, emergency_flash=0):
    """Render the environment to a pygame Surface."""
    surf = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    surf.fill((238, 240, 244))

    small_font, normal_font, large_font = fonts

    # Walls
    pygame.draw.rect(surf, (30, 33, 38),
                     pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
                     width=WALL_THICKNESS)

    # Obstacles
    for obstacle in env.obstacles:
        pygame.draw.rect(surf, (105, 110, 118), obstacle)
        pygame.draw.rect(surf, (35, 38, 43), obstacle, width=3)

    center = (round(env.robot_x), round(env.robot_y))

    # Warning and emergency circles
    pygame.draw.circle(surf, (230, 150, 45), center, round(WARNING_DISTANCE), width=1)
    pygame.draw.circle(surf, (220, 55, 55), center, round(EMERGENCY_DISTANCE), width=1)

    # Read sensors
    distances, endpoints, _ = env._read_sensors()

    # Draw sensors
    for angle_offset, distance, endpoint in zip(SENSOR_ANGLES, distances, endpoints):
        ray_angle = env.robot_angle + math.radians(angle_offset)
        start = (
            round(env.robot_x + math.cos(ray_angle) * ROBOT_RADIUS),
            round(env.robot_y - math.sin(ray_angle) * ROBOT_RADIUS),
        )
        end = (round(endpoint[0]), round(endpoint[1]))

        if distance <= EMERGENCY_DISTANCE:
            color = (220, 45, 45)
        elif distance <= WARNING_DISTANCE:
            color = (235, 140, 35)
        else:
            color = (40, 165, 90)

        pygame.draw.line(surf, color, start, end, width=2)
        pygame.draw.circle(surf, color, end, 3)

    # Robot body
    if env.motion_phase == "braking":
        robot_color = (240, 150, 35)
    elif env.motion_phase == "accelerating":
        robot_color = (75, 155, 225)
    else:
        robot_color = (65, 115, 205)

    pygame.draw.circle(surf, robot_color, center, round(ROBOT_RADIUS))
    pygame.draw.circle(surf, (25, 28, 33), center, round(ROBOT_RADIUS), width=3)

    # Heading arrow
    heading_end = (
        round(env.robot_x + math.cos(env.robot_angle) * (ROBOT_RADIUS + 24)),
        round(env.robot_y - math.sin(env.robot_angle) * (ROBOT_RADIUS + 24)),
    )
    # Draw arrow
    pygame.draw.line(surf, (190, 35, 35), center, heading_end, width=4)

    # Info panel (top-left)
    panel = pygame.Surface((380, 140), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 215))
    surf.blit(panel, (15, 15))

    action_name = "CONTINUE" if action == 0 else "BRAKE / REFLECT"
    phase = env.motion_phase.upper()
    speed = env.speed
    step = env.step_count
    reflections = info.get("episode_reflections", 0)
    emergencies = info.get("episode_emergency_triggers", 0)

    lines = [
        f"Action: {action} ({action_name})",
        f"Phase: {phase}    Speed: {speed:.1f} px/s",
        f"Step: {step}    Reflections: {reflections}    Emergency: {emergencies}",
    ]
    for i, text in enumerate(lines):
        surf.blit(small_font.render(text, True, (25, 28, 33)), (28, 25 + i * 24))

    # Sensor panel (top-right)
    sensor_panel = pygame.Surface((230, 240), pygame.SRCALPHA)
    sensor_panel.fill((255, 255, 255, 210))
    surf.blit(sensor_panel, (WINDOW_WIDTH - 248, 15))
    surf.blit(normal_font.render("Sensor distances", True, (25, 28, 33)),
              (WINDOW_WIDTH - 235, 22))

    for i, (angle, dist) in enumerate(zip(SENSOR_ANGLES, distances)):
        text = f"S{i} {angle:+4d} deg: {float(dist):6.1f}"
        if float(dist) <= EMERGENCY_DISTANCE:
            c = (220, 45, 45)
        elif float(dist) <= WARNING_DISTANCE:
            c = (235, 140, 35)
        else:
            c = (40, 165, 90)
        surf.blit(small_font.render(text, True, c),
                  (WINDOW_WIDTH - 235, 55 + i * 20))

    return surf


def main():
    pygame.init()
    fonts = (
        pygame.font.Font(None, 18),
        pygame.font.Font(None, 24),
        pygame.font.Font(None, 36),
    )

    env = RobotReflectionEnv()

    # Try to load the V3 model
    try:
        model = load_v3_policy(env)
        use_model = True
        print("V3 policy loaded successfully.")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Running without model (emergency-only baseline).")
        use_model = False

    seeds = [2026, 15000, 35000, 55000, 75000]
    captured = 0
    max_captures = 8

    for seed in seeds:
        if captured >= max_captures:
            break

        observation, info = env.reset(seed=seed)

        for step in range(600):
            if captured >= max_captures:
                break

            if use_model:
                action, _ = model.predict(observation, deterministic=True)
                action = int(np.asarray(action).item())
            else:
                action = 0

            # Capture at interesting moments
            front_dists = [float(env._read_sensors()[0][i]) for i in (3, 4, 5)]
            min_front = min(front_dists)

            should_capture = False
            label = ""

            if step == 0:
                should_capture = True
                label = "episode_start"
            elif env.motion_phase == "cruise" and min_front < WARNING_DISTANCE and min_front > EMERGENCY_DISTANCE and action == 1:
                should_capture = True
                label = "initiating_braking"
            elif env.motion_phase == "braking":
                should_capture = True
                label = "braking_phase"
            elif info.get("reflection", False):
                should_capture = True
                label = "reflection_moment"

            if should_capture and captured < max_captures:
                surf = draw_scene_to_surface(env, action, info, fonts)
                path = OUTPUT_DIR / f"{label}_seed{seed}_step{step}.png"
                pygame.image.save(surf, str(path))
                print(f"Captured: {path}")
                captured += 1

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    env.close()
    pygame.quit()
    print(f"\nDone. {captured} screenshots saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
