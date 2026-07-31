from __future__ import annotations

import numpy as np
import pygame
from common import load_config
from renderer import draw_scene


def test_draw_scene_on_headless_surface() -> None:
    pygame.font.init()
    config = load_config()
    surface = pygame.Surface(
        (int(config["render"]["width"]), int(config["render"]["height"]))
    )
    draw_scene(
        surface,
        np.zeros(4, dtype=np.float32),
        config,
        algorithm="dqn",
        episode=1,
        step=0,
        episode_return=0.0,
        action=0,
        status="RUNNING",
        paused=False,
        font=pygame.font.Font(None, 24),
    )
    assert surface.get_at((0, 0))[:3] == (18, 22, 30)
