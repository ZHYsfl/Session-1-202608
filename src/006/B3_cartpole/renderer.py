"""Cart-Pole 的 Pygame 绘制与键盘事件处理。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

BACKGROUND = (18, 22, 30)
TRACK = (115, 130, 148)
CART = (72, 166, 235)
POLE = (245, 190, 70)
AXLE = (235, 238, 243)
TEXT = (230, 235, 240)
GOOD = (80, 205, 130)
BAD = (225, 92, 78)


@dataclass(frozen=True)
class RenderEvents:
    """一次事件轮询的用户操作。"""

    quit: bool = False
    reset: bool = False
    pause_toggle: bool = False
    manual_action: int | None = None


def draw_scene(
    surface: pygame.Surface,
    state: np.ndarray,
    config: dict[str, Any],
    *,
    algorithm: str,
    episode: int,
    step: int,
    episode_return: float,
    action: int,
    status: str,
    paused: bool,
    font: pygame.font.Font,
) -> None:
    """在指定 Surface 上绘制环境和教学状态栏。"""

    render = config["render"]
    width = int(render["width"])
    height = int(render["height"])
    scale = float(render["world_scale"])
    surface.fill(BACKGROUND)
    track_y = int(height * 0.70)
    pygame.draw.line(surface, TRACK, (40, track_y + 24), (width - 300, track_y + 24), 4)

    position, velocity, angle, angular_velocity = (float(value) for value in state)
    cart_x = int((width - 300) / 2 + position * scale)
    cart_rect = pygame.Rect(cart_x - 40, track_y - 20, 80, 40)
    pygame.draw.rect(surface, CART, cart_rect, border_radius=7)
    pygame.draw.circle(surface, TRACK, (cart_x - 25, track_y + 24), 9)
    pygame.draw.circle(surface, TRACK, (cart_x + 25, track_y + 24), 9)

    axle = (cart_x, track_y - 18)
    pole_pixels = 210
    tip = (
        axle[0] + int(math.sin(angle) * pole_pixels),
        axle[1] - int(math.cos(angle) * pole_pixels),
    )
    pygame.draw.line(surface, POLE, axle, tip, 12)
    pygame.draw.circle(surface, AXLE, axle, 10)

    panel_x = width - 270
    pygame.draw.line(surface, TRACK, (panel_x - 15, 0), (panel_x - 15, height), 2)
    lines = (
        "B3 CART-POLE",
        f"Algorithm: {algorithm.upper()}",
        f"Episode: {episode}",
        f"Step: {step}/{config['environment']['max_steps']}",
        f"Return: {episode_return:.1f}",
        f"Action: {'RIGHT' if action == 1 else 'LEFT'}",
        f"Status: {status}",
        f"Paused: {'YES' if paused else 'NO'}",
        "",
        "STATE SPACE",
        f"x: {position:+.3f} m",
        f"x velocity: {velocity:+.3f}",
        f"angle: {math.degrees(angle):+.2f} deg",
        f"angular vel: {angular_velocity:+.3f}",
        "",
        "R: reset",
        "SPACE: pause",
        "LEFT/RIGHT: manual",
        "ESC: quit",
    )
    for index, line in enumerate(lines):
        color = TEXT
        if line.startswith("Status:"):
            color = GOOD if status in {"RUNNING", "SUCCESS"} else BAD
        surface.blit(font.render(line, True, color), (panel_x, 20 + index * 28))


class CartPoleRenderer:
    """持有 Pygame 窗口并提供事件和绘制接口。"""

    def __init__(self, config: dict[str, Any]) -> None:
        pygame.init()
        render = config["render"]
        self.config = config
        self.screen = pygame.display.set_mode(
            (int(render["width"]), int(render["height"]))
        )
        pygame.display.set_caption("B3 Cart-Pole Reinforcement Learning")
        self.font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()

    def poll_events(self) -> RenderEvents:
        """读取退出、重置、暂停和人工动作。"""

        quit_requested = reset = pause_toggle = False
        manual_action: int | None = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_requested = True
                elif event.key == pygame.K_r:
                    reset = True
                elif event.key == pygame.K_SPACE:
                    pause_toggle = True
                elif event.key == pygame.K_LEFT:
                    manual_action = 0
                elif event.key == pygame.K_RIGHT:
                    manual_action = 1
        return RenderEvents(quit_requested, reset, pause_toggle, manual_action)

    def render(self, **values: Any) -> None:
        """绘制一帧并按配置限制刷新率。"""

        draw_scene(self.screen, config=self.config, font=self.font, **values)
        pygame.display.flip()
        self.clock.tick(int(self.config["render"]["fps"]))

    @staticmethod
    def close() -> None:
        pygame.quit()
