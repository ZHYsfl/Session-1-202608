import csv
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import pygame


# ==================== 基本设置 ====================
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
FPS = 60
WALL_THICKNESS = 5

ROBOT_RADIUS = 22
MOVE_SPEED = 180.0
TURN_SPEED = 2.5

# 9个传感器相对于机器人正前方的角度
SENSOR_ANGLES = [
    -120,
    -90,
    -60,
    -30,
    0,
    30,
    60,
    90,
    120,
]

# 测距射线最大长度
SENSOR_RANGE = 160.0

RANDOM_SEED = int(
    datetime.now().strftime("%Y%m%d%H%M%S")
)
SCENE_ID = "scene_01"

# main.py位于src文件夹中，因此parents[1]是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "robot_raw.csv"
)


def circle_rect_collision(
    circle_x,
    circle_y,
    radius,
    rectangle,
):
    """判断圆形机器人是否与矩形相交。"""

    nearest_x = max(
        rectangle.left,
        min(circle_x, rectangle.right),
    )

    nearest_y = max(
        rectangle.top,
        min(circle_y, rectangle.bottom),
    )

    distance_x = circle_x - nearest_x
    distance_y = circle_y - nearest_y

    return (
        distance_x * distance_x
        + distance_y * distance_y
        <= radius * radius
    )


def collides_with_world(
    robot_x,
    robot_y,
    obstacles,
):
    """判断机器人是否碰到边界或障碍物。"""

    if robot_x - ROBOT_RADIUS <= WALL_THICKNESS:
        return True

    if (
        robot_x + ROBOT_RADIUS
        >= WINDOW_WIDTH - WALL_THICKNESS
    ):
        return True

    if robot_y - ROBOT_RADIUS <= WALL_THICKNESS:
        return True

    if (
        robot_y + ROBOT_RADIUS
        >= WINDOW_HEIGHT - WALL_THICKNESS
    ):
        return True

    for obstacle in obstacles:
        if circle_rect_collision(
            robot_x,
            robot_y,
            ROBOT_RADIUS,
            obstacle,
        ):
            return True

    return False


def random_valid_pose(
    obstacles,
    random_generator,
):
    """生成不与障碍物重叠的随机位置和朝向。"""

    margin = (
        ROBOT_RADIUS
        + WALL_THICKNESS
        + 5
    )

    for _ in range(2000):
        robot_x = random_generator.uniform(
            margin,
            WINDOW_WIDTH - margin,
        )

        robot_y = random_generator.uniform(
            margin,
            WINDOW_HEIGHT - margin,
        )

        robot_angle = random_generator.uniform(
            0.0,
            2.0 * math.pi,
        )

        if not collides_with_world(
            robot_x,
            robot_y,
            obstacles,
        ):
            return (
                robot_x,
                robot_y,
                robot_angle,
            )

    raise RuntimeError(
        "无法找到安全出生位置，请减少障碍物数量。"
    )


def cast_sensor_ray(
    robot_x,
    robot_y,
    robot_angle,
    relative_angle,
    detection_rectangles,
):
    """计算一条测距射线到最近障碍物的距离。"""

    sensor_angle = (
        robot_angle
        + math.radians(relative_angle)
    )

    direction_x = math.cos(sensor_angle)
    direction_y = -math.sin(sensor_angle)

    # 从机器人圆形外边缘开始测量
    start_x = (
        robot_x
        + direction_x * ROBOT_RADIUS
    )

    start_y = (
        robot_y
        + direction_y * ROBOT_RADIUS
    )

    maximum_end_x = (
        start_x
        + direction_x * SENSOR_RANGE
    )

    maximum_end_y = (
        start_y
        + direction_y * SENSOR_RANGE
    )

    nearest_distance = SENSOR_RANGE

    nearest_point = (
        maximum_end_x,
        maximum_end_y,
    )

    line_start = (
        round(start_x),
        round(start_y),
    )

    line_end = (
        round(maximum_end_x),
        round(maximum_end_y),
    )

    for rectangle in detection_rectangles:
        clipped_line = rectangle.clipline(
            line_start,
            line_end,
        )

        if not clipped_line:
            continue

        point_a, point_b = clipped_line

        for point in (point_a, point_b):
            distance = math.hypot(
                point[0] - start_x,
                point[1] - start_y,
            )

            if distance < nearest_distance:
                nearest_distance = distance

                nearest_point = (
                    float(point[0]),
                    float(point[1]),
                )

    return nearest_distance, nearest_point


def read_all_sensors(
    robot_x,
    robot_y,
    robot_angle,
    detection_rectangles,
):
    """读取9个方向的测距传感器。"""

    distances = []
    endpoints = []

    for relative_angle in SENSOR_ANGLES:
        distance, endpoint = cast_sensor_ray(
            robot_x,
            robot_y,
            robot_angle,
            relative_angle,
            detection_rectangles,
        )

        distances.append(distance)
        endpoints.append(endpoint)

    return distances, endpoints


def get_sensor_color(distance):
    """根据距离设置射线颜色。"""

    if distance < 45.0:
        return 220, 50, 50

    if distance < 90.0:
        return 230, 150, 40

    return 50, 170, 80


# 自动运动的动作和速度
AUTO_ACTIONS = {
    "forward": (
        MOVE_SPEED,
        0.0,
    ),
    "forward_left": (
        MOVE_SPEED * 0.85,
        TURN_SPEED * 0.65,
    ),
    "forward_right": (
        MOVE_SPEED * 0.85,
        -TURN_SPEED * 0.65,
    ),
    "turn_left": (
        0.0,
        TURN_SPEED,
    ),
    "turn_right": (
        0.0,
        -TURN_SPEED,
    ),
    "backward": (
        -MOVE_SPEED * 0.55,
        0.0,
    ),
}


def choose_auto_action(random_generator):
    """随机选择一个自动动作。"""

    action = random_generator.choices(
        population=list(AUTO_ACTIONS.keys()),
        weights=[
            34,
            22,
            22,
            8,
            8,
            6,
        ],
        k=1,
    )[0]

    duration = random_generator.uniform(
        0.35,
        1.10,
    )

    return action, duration


def create_csv_writer():
    """建立CSV文件和写入器。"""

    CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_is_empty = (
        not CSV_PATH.exists()
        or CSV_PATH.stat().st_size == 0
    )

    csv_file = CSV_PATH.open(
        mode="a",
        newline="",
        encoding="utf-8-sig",
    )

    fieldnames = [
        "run_id",
        "scene_id",
        "seed",
        "episode_id",
        "step",
        "x",
        "y",
        "angle_rad",
        "linear_speed",
        "angular_speed",
        "sensor_0",
        "sensor_1",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_5",
        "sensor_6",
        "sensor_7",
        "sensor_8",
        "collision_event",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    if file_is_empty:
        writer.writeheader()
        csv_file.flush()

    return csv_file, writer


def main():
    pygame.init()

    screen = pygame.display.set_mode(
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
        )
    )

    pygame.display.set_caption(
        "Robot Collision Prediction - Data Collection"
    )

    clock = pygame.time.Clock()

    # 使用Pygame默认字体，避免系统字体错误
    font = pygame.font.Font(
        None,
        25,
    )

    small_font = pygame.font.Font(
        None,
        22,
    )

    large_font = pygame.font.Font(
        None,
        38,
    )

    random_generator = random.Random(
        RANDOM_SEED
    )

    run_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # 矩形障碍物
    obstacles = [
        pygame.Rect(
            300,
            100,
            70,
            250,
        ),
        pygame.Rect(
            500,
            400,
            300,
            60,
        ),
        pygame.Rect(
            700,
            120,
            80,
            180,
        ),
        pygame.Rect(
            180,
            500,
            180,
            70,
        ),
    ]

    # 用于传感器检测的四周边界
    boundary_walls = [
        pygame.Rect(
            0,
            0,
            WINDOW_WIDTH,
            WALL_THICKNESS,
        ),
        pygame.Rect(
            0,
            WINDOW_HEIGHT - WALL_THICKNESS,
            WINDOW_WIDTH,
            WALL_THICKNESS,
        ),
        pygame.Rect(
            0,
            0,
            WALL_THICKNESS,
            WINDOW_HEIGHT,
        ),
        pygame.Rect(
            WINDOW_WIDTH - WALL_THICKNESS,
            0,
            WALL_THICKNESS,
            WINDOW_HEIGHT,
        ),
    ]

    detection_rectangles = (
        obstacles
        + boundary_walls
    )

    (
        robot_x,
        robot_y,
        robot_angle,
    ) = random_valid_pose(
        obstacles,
        random_generator,
    )

    mode = "manual"
    recording = False
    collision = False

    episode_id = 1
    step_in_episode = 0
    recorded_rows = 0

    current_action = "manual"
    action_time_remaining = 0.0
    collision_wait_time = 0.0

    csv_file, csv_writer = (
        create_csv_writer()
    )

    running = True

    try:
        while running:
            delta_time = (
                clock.tick(FPS)
                / 1000.0
            )

            # =========================
            # 处理键盘和窗口事件
            # =========================
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_m:
                        if mode == "manual":
                            mode = "auto"
                        else:
                            mode = "manual"

                        collision = False
                        collision_wait_time = 0.0
                        action_time_remaining = 0.0

                    elif event.key == pygame.K_c:
                        recording = not recording

                    elif event.key == pygame.K_r:
                        (
                            robot_x,
                            robot_y,
                            robot_angle,
                        ) = random_valid_pose(
                            obstacles,
                            random_generator,
                        )

                        episode_id += 1
                        step_in_episode = 0
                        collision = False
                        collision_wait_time = 0.0
                        action_time_remaining = 0.0

            linear_speed = 0.0
            angular_speed = 0.0
            movement_allowed = False

            # =========================
            # 手动模式
            # =========================
            if mode == "manual":
                keys = pygame.key.get_pressed()

                if (
                    keys[pygame.K_w]
                    or keys[pygame.K_UP]
                ):
                    linear_speed = MOVE_SPEED

                if (
                    keys[pygame.K_s]
                    or keys[pygame.K_DOWN]
                ):
                    linear_speed = -MOVE_SPEED

                if (
                    keys[pygame.K_a]
                    or keys[pygame.K_LEFT]
                ):
                    angular_speed = TURN_SPEED

                if (
                    keys[pygame.K_d]
                    or keys[pygame.K_RIGHT]
                ):
                    angular_speed = -TURN_SPEED

                current_action = "manual"
                movement_allowed = True

            # =========================
            # 自动模式：碰撞后等待复位
            # =========================
            elif collision:
                current_action = "collision_wait"

                collision_wait_time += (
                    delta_time
                )

                if collision_wait_time >= 0.45:
                    (
                        robot_x,
                        robot_y,
                        robot_angle,
                    ) = random_valid_pose(
                        obstacles,
                        random_generator,
                    )

                    episode_id += 1
                    step_in_episode = 0
                    collision = False
                    collision_wait_time = 0.0
                    action_time_remaining = 0.0

            # =========================
            # 自动模式：随机运动
            # =========================
            else:
                movement_allowed = True

                action_time_remaining -= (
                    delta_time
                )

                if action_time_remaining <= 0.0:
                    (
                        current_action,
                        action_time_remaining,
                    ) = choose_auto_action(
                        random_generator
                    )

                (
                    linear_speed,
                    angular_speed,
                ) = AUTO_ACTIONS[
                    current_action
                ]

            # 移动前读取传感器
            # 这些数据是以后神经网络的输入
            (
                sensor_values_before_move,
                _,
            ) = read_all_sensors(
                robot_x,
                robot_y,
                robot_angle,
                detection_rectangles,
            )

            collision_event = 0

            # =========================
            # 更新机器人状态
            # =========================
            if movement_allowed:
                old_x = robot_x
                old_y = robot_y
                old_angle = robot_angle

                candidate_angle = (
                    old_angle
                    + angular_speed
                    * delta_time
                ) % (2.0 * math.pi)

                candidate_x = (
                    old_x
                    + math.cos(candidate_angle)
                    * linear_speed
                    * delta_time
                )

                candidate_y = (
                    old_y
                    - math.sin(candidate_angle)
                    * linear_speed
                    * delta_time
                )

                collision_event = int(
                    collides_with_world(
                        candidate_x,
                        candidate_y,
                        obstacles,
                    )
                )

                collision = bool(
                    collision_event
                )

                if not collision:
                    robot_x = candidate_x
                    robot_y = candidate_y
                    robot_angle = candidate_angle
                else:
                    collision_wait_time = 0.0

                # 只记录自动模式数据
                if (
                    mode == "auto"
                    and recording
                ):
                    row = {
                        "run_id": run_id,
                        "scene_id": SCENE_ID,
                        "seed": RANDOM_SEED,
                        "episode_id": episode_id,
                        "step": step_in_episode,
                        "x": round(
                            old_x,
                            6,
                        ),
                        "y": round(
                            old_y,
                            6,
                        ),
                        "angle_rad": round(
                            old_angle,
                            8,
                        ),
                        "linear_speed": round(
                            linear_speed,
                            6,
                        ),
                        "angular_speed": round(
                            angular_speed,
                            8,
                        ),
                        "collision_event": (
                            collision_event
                        ),
                    }

                    for (
                        sensor_index,
                        distance,
                    ) in enumerate(
                        sensor_values_before_move
                    ):
                        row[
                            f"sensor_{sensor_index}"
                        ] = round(
                            distance,
                            6,
                        )

                    csv_writer.writerow(row)
                    recorded_rows += 1

                    # 每60行保存一次
                    if recorded_rows % 60 == 0:
                        csv_file.flush()

                step_in_episode += 1

            # =========================
            # 重新计算显示用传感器
            # =========================
            (
                sensor_distances,
                sensor_endpoints,
            ) = read_all_sensors(
                robot_x,
                robot_y,
                robot_angle,
                detection_rectangles,
            )

            # =========================
            # 绘制背景与障碍物
            # =========================
            screen.fill(
                (
                    240,
                    240,
                    240,
                )
            )

            pygame.draw.rect(
                screen,
                (
                    30,
                    30,
                    30,
                ),
                pygame.Rect(
                    0,
                    0,
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT,
                ),
                width=WALL_THICKNESS,
            )

            for obstacle in obstacles:
                pygame.draw.rect(
                    screen,
                    (
                        110,
                        110,
                        110,
                    ),
                    obstacle,
                )

                pygame.draw.rect(
                    screen,
                    (
                        30,
                        30,
                        30,
                    ),
                    obstacle,
                    width=3,
                )

            # =========================
            # 绘制9条测距射线
            # =========================
            for (
                relative_angle,
                distance,
                endpoint,
            ) in zip(
                SENSOR_ANGLES,
                sensor_distances,
                sensor_endpoints,
            ):
                sensor_angle = (
                    robot_angle
                    + math.radians(
                        relative_angle
                    )
                )

                start_x = (
                    robot_x
                    + math.cos(sensor_angle)
                    * ROBOT_RADIUS
                )

                start_y = (
                    robot_y
                    - math.sin(sensor_angle)
                    * ROBOT_RADIUS
                )

                color = get_sensor_color(
                    distance
                )

                pygame.draw.line(
                    screen,
                    color,
                    (
                        round(start_x),
                        round(start_y),
                    ),
                    (
                        round(endpoint[0]),
                        round(endpoint[1]),
                    ),
                    width=2,
                )

                pygame.draw.circle(
                    screen,
                    color,
                    (
                        round(endpoint[0]),
                        round(endpoint[1]),
                    ),
                    4,
                )

            # =========================
            # 绘制机器人
            # =========================
            robot_center = (
                round(robot_x),
                round(robot_y),
            )

            if collision:
                robot_color = (
                    210,
                    60,
                    60,
                )
            else:
                robot_color = (
                    60,
                    120,
                    200,
                )

            pygame.draw.circle(
                screen,
                robot_color,
                robot_center,
                ROBOT_RADIUS,
            )

            pygame.draw.circle(
                screen,
                (
                    20,
                    20,
                    20,
                ),
                robot_center,
                ROBOT_RADIUS,
                width=2,
            )

            # 机器人朝向线
            direction_length = (
                ROBOT_RADIUS
                + 18
            )

            direction_end = (
                round(
                    robot_x
                    + math.cos(robot_angle)
                    * direction_length
                ),
                round(
                    robot_y
                    - math.sin(robot_angle)
                    * direction_length
                ),
            )

            pygame.draw.line(
                screen,
                (
                    200,
                    40,
                    40,
                ),
                robot_center,
                direction_end,
                width=4,
            )

            # =========================
            # 显示状态信息
            # =========================
            if recording:
                recording_text = "ON"
            else:
                recording_text = "OFF"

            status_lines = [
                (
                    "M: Manual/Auto    "
                    "C: Record    "
                    "R: Reset    "
                    "ESC: Exit"
                ),
                (
                    "Manual: "
                    "W/S move, "
                    "A/D turn"
                ),
                (
                    f"Mode: {mode.upper()}    "
                    f"Recording: {recording_text}    "
                    f"Rows: {recorded_rows}"
                ),
                (
                    f"Episode: {episode_id}    "
                    f"Step: {step_in_episode}    "
                    f"Action: {current_action}"
                ),
            ]

            for (
                line_index,
                text,
            ) in enumerate(
                status_lines
            ):
                screen.blit(
                    font.render(
                        text,
                        True,
                        (
                            25,
                            25,
                            25,
                        ),
                    ),
                    (
                        20,
                        20
                        + line_index * 28,
                    ),
                )

            screen.blit(
                font.render(
                    "Sensor distances",
                    True,
                    (
                        20,
                        20,
                        20,
                    ),
                ),
                (
                    800,
                    145,
                ),
            )

            for (
                sensor_index,
                (
                    angle_value,
                    distance,
                ),
            ) in enumerate(
                zip(
                    SENSOR_ANGLES,
                    sensor_distances,
                )
            ):
                sensor_text = (
                    f"S{sensor_index} "
                    f"({angle_value:+4d} deg): "
                    f"{distance:6.1f} px"
                )

                screen.blit(
                    small_font.render(
                        sensor_text,
                        True,
                        get_sensor_color(
                            distance
                        ),
                    ),
                    (
                        790,
                        180
                        + sensor_index * 25,
                    ),
                )

            if collision:
                screen.blit(
                    large_font.render(
                        "COLLISION",
                        True,
                        (
                            210,
                            30,
                            30,
                        ),
                    ),
                    (
                        WINDOW_WIDTH - 190,
                        25,
                    ),
                )

            pygame.display.flip()

    finally:
        csv_file.flush()
        csv_file.close()
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())