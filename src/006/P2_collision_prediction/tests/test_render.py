"""Pygame 绘制函数的无窗口冒烟测试。"""

from __future__ import annotations

import os

import pygame
import torch
from common import load_config
from main import draw_scene, draw_status
from rl_env import BatchedLocalAvoidanceEnv


def test_pygame_scene_renders_to_surface() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    config = load_config()
    env = BatchedLocalAvoidanceEnv(config, 1, torch.device("cpu"), seed=5)
    env.set_curriculum_stage(2)
    env.reset()
    snapshot = env.snapshot()
    pygame.init()
    surface = pygame.Surface((int(env.width) + 300, int(env.height)))
    font = pygame.font.Font(None, 23)
    draw_scene(surface, snapshot, env, [snapshot["position"]])
    draw_status(
        surface,
        font,
        env,
        "reactive",
        "RUNNING",
        False,
        False,
        snapshot,
    )
    pygame.quit()
