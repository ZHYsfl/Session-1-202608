import math
import sys
from pathlib import Path

import numpy as np
import pygame
import torch
from stable_baselines3 import DQN

from robot_env_v3 import (
    EMERGENCY_DISTANCE,
    FPS,
    ROBOT_RADIUS,
    SENSOR_ANGLES,
    WARNING_DISTANCE,
    WALL_THICKNESS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    RobotReflectionEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLICY_PATH = (
    PROJECT_ROOT
    / "models"
    / "dqn_reflection_v3_safe"
    / "best_policy.pt"
)

RESET_DELAY = 0.75
FLASH_FRAMES = 45


def load_policy(env):
    """读取V3最佳DQN策略。"""

    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"找不到V3最佳模型：{POLICY_PATH}"
        )

    payload = torch.load(
        POLICY_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if (
        payload.get("format")
        != "safe_policy_state_dict_v3"
    ):
        raise ValueError(
            "模型文件不是V3安全格式。"
        )

    if (
        payload.get("observation_shape")
        != [14]
    ):
        raise ValueError(
            "模型状态维数不是14。"
        )

    if payload.get("action_count") != 2:
        raise ValueError(
            "模型动作数量不是2。"
        )

    model = DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs={
            "net_arch": [64, 64],
        },
        device="cpu",
        verbose=0,
        seed=2026,
    )

    model.policy.load_state_dict(
        payload["policy_state_dict"],
        strict=True,
    )

    model.policy.set_training_mode(
        False
    )

    return model, payload


def predict_action(
    model,
    observation,
):
    """计算两个动作的Q值并选择较大者。"""

    tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = (
            model.q_net(tensor)
            .cpu()
            .numpy()[0]
        )

    action = int(
        np.argmax(q_values)
    )

    return action, q_values


def point_from_angle(
    x,
    y,
    angle,
    length,
):
    """根据角度计算Pygame坐标点。"""

    return (
        round(
            x
            + math.cos(angle)
            * length
        ),
        round(
            y
            - math.sin(angle)
            * length
        ),
    )


def sensor_color(
    distance,
):
    """根据距离选择传感器颜色。"""

    if distance <= EMERGENCY_DISTANCE:
        return (
            220,
            45,
            45,
        )

    if distance <= WARNING_DISTANCE:
        return (
            235,
            140,
            35,
        )

    return (
        40,
        165,
        90,
    )


def draw_arrow(
    screen,
    start,
    end,
    color,
    width=4,
):
    """绘制带箭头的直线。"""

    pygame.draw.line(
        screen,
        color,
        start,
        end,
        width,
    )

    angle = math.atan2(
        start[1] - end[1],
        end[0] - start[0],
    )

    size = 11

    left = (
        round(
            end[0]
            - math.cos(
                angle - 0.5
            )
            * size
        ),
        round(
            end[1]
            + math.sin(
                angle - 0.5
            )
            * size
        ),
    )

    right = (
        round(
            end[0]
            - math.cos(
                angle + 0.5
            )
            * size
        ),
        round(
            end[1]
            + math.sin(
                angle + 0.5
            )
            * size
        ),
    )

    pygame.draw.polygon(
        screen,
        color,
        [
            end,
            left,
            right,
        ],
    )


def draw_scene(
    screen,
    env,
    action,
    q_values,
    info,
    payload,
    episode,
    total_collisions,
    paused,
    end_message,
    reflection_flash,
    emergency_flash,
    incident_angle,
    reflected_angle,
    fonts,
):
    """绘制环境、机器人、传感器和状态信息。"""

    (
        small_font,
        normal_font,
        large_font,
    ) = fonts

    screen.fill(
        (
            238,
            240,
            244,
        )
    )

    # 四周墙体
    pygame.draw.rect(
        screen,
        (
            30,
            33,
            38,
        ),
        pygame.Rect(
            0,
            0,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        ),
        width=WALL_THICKNESS,
    )

    # 内部障碍物
    for obstacle in env.obstacles:
        pygame.draw.rect(
            screen,
            (
                105,
                110,
                118,
            ),
            obstacle,
        )

        pygame.draw.rect(
            screen,
            (
                35,
                38,
                43,
            ),
            obstacle,
            width=3,
        )

    center = (
        round(env.robot_x),
        round(env.robot_y),
    )

    # 黄色为预警距离
    pygame.draw.circle(
        screen,
        (
            230,
            150,
            45,
        ),
        center,
        round(
            WARNING_DISTANCE
        ),
        width=1,
    )

    # 红色为紧急保护距离
    pygame.draw.circle(
        screen,
        (
            220,
            55,
            55,
        ),
        center,
        round(
            EMERGENCY_DISTANCE
        ),
        width=1,
    )

    # 读取并绘制9个传感器
    (
        distances,
        endpoints,
        _,
    ) = env._read_sensors()

    for (
        angle_offset,
        distance,
        endpoint,
    ) in zip(
        SENSOR_ANGLES,
        distances,
        endpoints,
    ):
        ray_angle = (
            env.robot_angle
            + math.radians(
                angle_offset
            )
        )

        start = point_from_angle(
            env.robot_x,
            env.robot_y,
            ray_angle,
            ROBOT_RADIUS,
        )

        end = (
            round(endpoint[0]),
            round(endpoint[1]),
        )

        color = sensor_color(
            float(distance)
        )

        pygame.draw.line(
            screen,
            color,
            start,
            end,
            width=2,
        )

        pygame.draw.circle(
            screen,
            color,
            end,
            3,
        )

    # 根据运动阶段显示机器人颜色
    if end_message == "COLLISION":
        robot_color = (
            215,
            55,
            55,
        )

    elif env.motion_phase == "braking":
        robot_color = (
            240,
            150,
            35,
        )

    elif env.motion_phase == "accelerating":
        robot_color = (
            75,
            155,
            225,
        )

    else:
        robot_color = (
            65,
            115,
            205,
        )

    pygame.draw.circle(
        screen,
        robot_color,
        center,
        round(
            ROBOT_RADIUS
        ),
    )

    pygame.draw.circle(
        screen,
        (
            25,
            28,
            33,
        ),
        center,
        round(
            ROBOT_RADIUS
        ),
        width=3,
    )

    # 当前运动方向
    heading_end = point_from_angle(
        env.robot_x,
        env.robot_y,
        env.robot_angle,
        ROBOT_RADIUS + 24,
    )

    draw_arrow(
        screen,
        center,
        heading_end,
        (
            190,
            35,
            35,
        ),
    )

    # 显示入射方向和反射方向
    if (
        reflection_flash > 0
        and incident_angle
        is not None
        and reflected_angle
        is not None
    ):
        incoming_start = (
            point_from_angle(
                env.robot_x,
                env.robot_y,
                incident_angle
                + math.pi,
                75,
            )
        )

        outgoing_end = (
            point_from_angle(
                env.robot_x,
                env.robot_y,
                reflected_angle,
                90,
            )
        )

        # 紫色：入射方向
        draw_arrow(
            screen,
            incoming_start,
            center,
            (
                150,
                70,
                200,
            ),
        )

        # 蓝色：反射方向
        draw_arrow(
            screen,
            center,
            outgoing_end,
            (
                40,
                175,
                220,
            ),
        )

    # 左上角状态面板
    panel = pygame.Surface(
        (
            510,
            245,
        ),
        pygame.SRCALPHA,
    )

    panel.fill(
        (
            255,
            255,
            255,
            220,
        )
    )

    screen.blit(
        panel,
        (
            18,
            18,
        ),
    )

    if action == 0:
        action_name = "CONTINUE"
    else:
        action_name = (
            "BRAKE / REFLECT"
        )

    lines = [
        (
            "DQN V3 - Specular Reflection Control"
        ),
        (
            f"Episode: {episode}    "
            f"Step: {env.step_count}    "
            f"Collisions: {total_collisions}"
        ),
        (
            f"Action: {action} "
            f"({action_name})"
        ),
        (
            f"Q0: {q_values[0]:.3f}    "
            f"Q1: {q_values[1]:.3f}"
        ),
        (
            f"Phase: {env.motion_phase}    "
            f"Speed: {env.speed:.1f}"
        ),
        (
            f"Reflections: "
            f"{info.get('episode_reflections', 0)}    "
            f"Emergency: "
            f"{info.get('episode_emergency_triggers', 0)}"
        ),
        (
            f"Model: {payload.get('label')}    "
            f"Train steps: "
            f"{payload.get('num_timesteps')}"
        ),
        (
            "SPACE: Pause    "
            "R: Reset    "
            "ESC: Exit"
        ),
    ]

    for index, text in enumerate(
        lines
    ):
        if index == 0:
            font = normal_font
        else:
            font = small_font

        screen.blit(
            font.render(
                text,
                True,
                (
                    25,
                    28,
                    33,
                ),
            ),
            (
                32,
                30
                + index * 27,
            ),
        )

    # 右上角传感器读数
    sensor_panel = pygame.Surface(
        (
            265,
            270,
        ),
        pygame.SRCALPHA,
    )

    sensor_panel.fill(
        (
            255,
            255,
            255,
            215,
        )
    )

    screen.blit(
        sensor_panel,
        (
            WINDOW_WIDTH - 285,
            18,
        ),
    )

    screen.blit(
        normal_font.render(
            "Sensor distances",
            True,
            (
                25,
                28,
                33,
            ),
        ),
        (
            WINDOW_WIDTH - 270,
            30,
        ),
    )

    for (
        index,
        (
            angle,
            distance,
        ),
    ) in enumerate(
        zip(
            SENSOR_ANGLES,
            distances,
        )
    ):
        text = (
            f"S{index} "
            f"{angle:+4d} deg: "
            f"{float(distance):6.1f}"
        )

        screen.blit(
            small_font.render(
                text,
                True,
                sensor_color(
                    float(distance)
                ),
            ),
            (
                WINDOW_WIDTH - 270,
                62
                + index * 23,
            ),
        )

    # 底部事件提示
    if reflection_flash > 0:
        banner = (
            "SPECULAR REFLECTION"
        )

        color = (
            40,
            145,
            215,
        )

    elif emergency_flash > 0:
        banner = (
            "EMERGENCY PROTECTION"
        )

        color = (
            220,
            120,
            25,
        )

    elif paused:
        banner = "PAUSED"

        color = (
            85,
            85,
            90,
        )

    elif end_message:
        banner = end_message

        color = (
            210,
            45,
            45,
        )

    else:
        banner = ""

        color = (
            0,
            0,
            0,
        )

    if banner:
        surface = large_font.render(
            banner,
            True,
            color,
        )

        rectangle = (
            surface.get_rect(
                center=(
                    WINDOW_WIDTH // 2,
                    WINDOW_HEIGHT - 45,
                )
            )
        )

        screen.blit(
            surface,
            rectangle,
        )


def main():
    pygame.init()

    screen = pygame.display.set_mode(
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )
    )

    pygame.display.set_caption(
        "DQN V3 Robot Collision Avoidance"
    )

    clock = pygame.time.Clock()

    fonts = (
        pygame.font.Font(
            None,
            23,
        ),
        pygame.font.Font(
            None,
            29,
        ),
        pygame.font.Font(
            None,
            42,
        ),
    )

    env = RobotReflectionEnv()

    model, payload = load_policy(
        env
    )

    observation, info = (
        env.reset(
            seed=15000
        )
    )

    episode = 1
    total_collisions = 0

    action = 0

    q_values = np.zeros(
        2,
        dtype=np.float32,
    )

    paused = False
    running = True

    reset_timer = 0.0
    end_message = ""

    reflection_flash = 0
    emergency_flash = 0

    incident_angle = None
    reflected_angle = None

    while running:
        elapsed = (
            clock.tick(FPS)
            / 1000.0
        )

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    (
                        observation,
                        info,
                    ) = env.reset()

                    episode += 1

                    reset_timer = 0.0
                    end_message = ""

                    reflection_flash = 0
                    emergency_flash = 0

        reflection_flash = max(
            0,
            reflection_flash - 1,
        )

        emergency_flash = max(
            0,
            emergency_flash - 1,
        )

        if not paused:

            if reset_timer > 0.0:

                reset_timer -= (
                    elapsed
                )

                if reset_timer <= 0.0:

                    (
                        observation,
                        info,
                    ) = env.reset()

                    episode += 1
                    end_message = ""

            else:

                (
                    action,
                    q_values,
                ) = predict_action(
                    model,
                    observation,
                )

                angle_before = (
                    env.robot_angle
                )

                (
                    observation,
                    _reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(
                    action
                )

                if info.get(
                    "reflection",
                    False,
                ):
                    incident_angle = (
                        angle_before
                    )

                    reflected_angle = (
                        env.robot_angle
                    )

                    reflection_flash = (
                        FLASH_FRAMES
                    )

                if info.get(
                    "emergency",
                    False,
                ):
                    emergency_flash = (
                        FLASH_FRAMES
                    )

                if terminated:

                    total_collisions += 1

                    end_message = (
                        "COLLISION"
                    )

                    reset_timer = (
                        RESET_DELAY
                    )

                elif truncated:

                    end_message = (
                        "EPISODE COMPLETE"
                    )

                    reset_timer = (
                        RESET_DELAY
                    )

        draw_scene(
            screen,
            env,
            action,
            q_values,
            info,
            payload,
            episode,
            total_collisions,
            paused,
            end_message,
            reflection_flash,
            emergency_flash,
            incident_angle,
            reflected_angle,
            fonts,
        )

        pygame.display.flip()

    env.close()
    pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())