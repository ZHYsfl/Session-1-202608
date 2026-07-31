import math
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


# =========================================================
# 场景参数
# =========================================================
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
WALL_THICKNESS = 15

ROBOT_RADIUS = 20.0

SENSOR_RANGE = 160.0

SENSOR_ANGLES = (
    -120,
    -90,
    -60,
    -30,
    0,
    30,
    60,
    90,
    120,
)

# 使用前方左、中、右三个传感器判断碰撞风险
FRONT_SENSOR_INDICES = (3, 4, 5)

FPS = 60
TIME_STEP = 1.0 / FPS

MAX_EPISODE_STEPS = 600


# =========================================================
# 速度控制参数
# =========================================================
NORMAL_SPEED = 150.0
SLOW_SPEED = 45.0

DECELERATION = 360.0
ACCELERATION = 220.0

WARNING_DISTANCE = 90.0
EMERGENCY_DISTANCE = 42.0


class RobotReflectionEnv(gym.Env):
    """
    两动作DQN强化学习环境。

    动作0：保持当前方向运动
    动作1：启动“减速—镜面反射—重新加速”流程
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(self) -> None:
        super().__init__()

        # =================================================
        # 动作空间
        # 0：保持运动
        # 1：启动减速反射
        # =================================================
        self.action_space = spaces.Discrete(2)

        # =================================================
        # 状态空间，共14维
        #
        # 0～8：9个归一化传感器距离
        # 9：当前速度
        # 10：sin(angle)
        # 11：cos(angle)
        # 12：是否处于减速阶段
        # 13：是否处于加速阶段
        # =================================================
        observation_low = np.asarray(
            [0.0] * 10
            + [
                -1.0,
                -1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        observation_high = np.asarray(
            [1.0] * 14,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )

        # =================================================
        # 障碍物
        # 与之前的Pygame场景保持一致
        # =================================================
        self.obstacles = [
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

        # 四周墙体，用于传感器射线检测
        self.boundary_walls = [
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

        self.sensor_rectangles = (
            self.obstacles
            + self.boundary_walls
        )

        # 机器人状态
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_angle = 0.0

        self.speed = NORMAL_SPEED

        # cruise、braking、accelerating
        self.motion_phase = "cruise"

        self.step_count = 0

        # Episode统计量
        self.episode_reflections = 0
        self.episode_emergency_triggers = 0
        self.episode_collisions = 0

    # =====================================================
    # Gymnasium重置接口
    # =====================================================
    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        (
            self.robot_x,
            self.robot_y,
            self.robot_angle,
        ) = self._random_valid_pose()

        self.speed = NORMAL_SPEED
        self.motion_phase = "cruise"

        self.step_count = 0

        self.episode_reflections = 0
        self.episode_emergency_triggers = 0
        self.episode_collisions = 0

        observation = self._get_observation()

        info = self._build_info(
            collision=False,
            reflection=False,
            emergency=False,
        )

        return observation, info

    # =====================================================
    # Gymnasium单步接口
    # =====================================================
    def step(
        self,
        action,
    ):
        action = int(action)

        if action not in (0, 1):
            raise ValueError(
                f"无效动作：{action}"
            )

        self.step_count += 1

        # 移动前读取传感器
        (
            sensors_before,
            _,
            normals_before,
        ) = self._read_sensors()

        (
            front_distance_before,
            front_normal,
        ) = self._front_hit(
            sensors_before,
            normals_before,
        )

        reward = 0.02

        reflection_happened = False
        emergency_triggered = False
        agent_started_braking = False
        useless_reflection_request = False

        # =================================================
        # 1. 正常巡航阶段
        # =================================================
        if self.motion_phase == "cruise":

            # DQN主动要求减速反射
            if action == 1:

                if (
                    front_distance_before
                    <= WARNING_DISTANCE
                    and front_normal is not None
                ):
                    self.motion_phase = "braking"

                    agent_started_braking = True

                else:
                    # 周围没有风险却启动反射
                    useless_reflection_request = True

            # DQN没有及时反应时的紧急保护
            elif (
                front_distance_before
                <= EMERGENCY_DISTANCE
                and front_normal is not None
            ):
                self.motion_phase = "braking"

                emergency_triggered = True

                self.episode_emergency_triggers += 1

        # =================================================
        # 2. 减速阶段
        # 方向不变，只减小速度
        # =================================================
        if self.motion_phase == "braking":

            self.speed = max(
                SLOW_SPEED,
                self.speed
                - DECELERATION
                * TIME_STEP,
            )

            # 正常减速完成，或者距离已经非常近
            if (
                self.speed
                <= SLOW_SPEED + 1e-6
                or front_distance_before
                <= EMERGENCY_DISTANCE
            ):
                reflection_happened = (
                    self._apply_specular_reflection(
                        front_normal
                    )
                )

                if reflection_happened:
                    self.episode_reflections += 1

                self.motion_phase = "accelerating"

        # =================================================
        # 3. 重新加速阶段
        # 沿反射后的方向恢复速度
        # =================================================
        elif self.motion_phase == "accelerating":

            self.speed = min(
                NORMAL_SPEED,
                self.speed
                + ACCELERATION
                * TIME_STEP,
            )

            if (
                self.speed
                >= NORMAL_SPEED - 1e-6
            ):
                self.speed = NORMAL_SPEED

                self.motion_phase = "cruise"

        # =================================================
        # 根据当前方向和速度计算下一位置
        # =================================================
        candidate_x = (
            self.robot_x
            + math.cos(
                self.robot_angle
            )
            * self.speed
            * TIME_STEP
        )

        candidate_y = (
            self.robot_y
            - math.sin(
                self.robot_angle
            )
            * self.speed
            * TIME_STEP
        )

        collision = self._collides(
            candidate_x,
            candidate_y,
        )

        terminated = bool(
            collision
        )

        if collision:
            self.episode_collisions += 1

        else:
            self.robot_x = candidate_x
            self.robot_y = candidate_y

        # 移动后重新读取传感器
        (
            sensors_after,
            _,
            _,
        ) = self._read_sensors()

        front_distance_after = min(
            float(
                sensors_after[index]
            )
            for index
            in FRONT_SENSOR_INDICES
        )

        # =================================================
        # 奖励函数
        # =================================================

        # 鼓励保持有效运动速度
        reward += (
            0.03
            * self.speed
            / NORMAL_SPEED
        )

        # 接近障碍物时给予连续危险惩罚
        if (
            front_distance_after
            < WARNING_DISTANCE
        ):
            danger_ratio = (
                1.0
                - front_distance_after
                / WARNING_DISTANCE
            )

            reward -= (
                0.15
                * danger_ratio
            )

        # DQN在危险区主动开始减速
        if agent_started_braking:
            reward += 0.0

        # 空旷区域无意义地要求反射
        if useless_reflection_request:
            reward -= 0.20

        # 依靠紧急保护说明DQN反应太晚
        if emergency_triggered:
            reward -= 2.00

        # 完成一次镜面反射
        if reflection_happened:

            distance_improvement = (
                front_distance_after
                - front_distance_before
            )

            # 只有反射后前方空间增大才奖励
            if distance_improvement > 0.0:
                reward += (
                    0.30
                    * distance_improvement
                    / SENSOR_RANGE
                )
            else:
                reward -= 0.20

        # 碰撞给予最大惩罚
        if collision:
            reward = -25.0

        # 达到最大步数属于截断，不是真实碰撞
        truncated = bool(
            self.step_count
            >= MAX_EPISODE_STEPS
            and not terminated
        )

        observation = (
            self._get_observation()
        )

        info = self._build_info(
            collision=collision,
            reflection=reflection_happened,
            emergency=emergency_triggered,
        )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # =====================================================
    # 构造14维状态
    # =====================================================
    def _get_observation(
        self,
    ) -> np.ndarray:

        (
            sensor_distances,
            _,
            _,
        ) = self._read_sensors()

        normalized_sensors = np.clip(
            sensor_distances
            / SENSOR_RANGE,
            0.0,
            1.0,
        )

        observation = np.asarray(
            normalized_sensors.tolist()
            + [
                self.speed
                / NORMAL_SPEED,

                math.sin(
                    self.robot_angle
                ),

                math.cos(
                    self.robot_angle
                ),

                float(
                    self.motion_phase
                    == "braking"
                ),

                float(
                    self.motion_phase
                    == "accelerating"
                ),
            ],
            dtype=np.float32,
        )

        return observation

    # =====================================================
    # 建立info信息
    # =====================================================
    def _build_info(
        self,
        collision: bool,
        reflection: bool,
        emergency: bool,
    ) -> dict:

        return {
            "collision": bool(
                collision
            ),

            "reflection": bool(
                reflection
            ),

            "emergency": bool(
                emergency
            ),

            "speed": float(
                self.speed
            ),

            "motion_phase": (
                self.motion_phase
            ),

            "step_count": int(
                self.step_count
            ),

            "episode_reflections": int(
                self.episode_reflections
            ),

            "episode_emergency_triggers": int(
                self.episode_emergency_triggers
            ),

            "episode_collisions": int(
                self.episode_collisions
            ),
        }

    # =====================================================
    # 随机生成初始位置
    # =====================================================
    def _random_valid_pose(
        self,
    ):
        margin = (
            WALL_THICKNESS
            + ROBOT_RADIUS
            + 10.0
        )

        for _ in range(5000):

            x = float(
                self.np_random.uniform(
                    margin,
                    WINDOW_WIDTH - margin,
                )
            )

            y = float(
                self.np_random.uniform(
                    margin,
                    WINDOW_HEIGHT - margin,
                )
            )

            angle = float(
                self.np_random.uniform(
                    0.0,
                    2.0 * math.pi,
                )
            )

            if self._collides(
                x,
                y,
            ):
                continue

            old_x = self.robot_x
            old_y = self.robot_y
            old_angle = self.robot_angle

            self.robot_x = x
            self.robot_y = y
            self.robot_angle = angle

            (
                sensor_distances,
                _,
                _,
            ) = self._read_sensors()

            self.robot_x = old_x
            self.robot_y = old_y
            self.robot_angle = old_angle

            # 防止机器人出生在紧贴障碍物的位置
            if (
                float(
                    sensor_distances.min()
                )
                >= 35.0
            ):
                return (
                    x,
                    y,
                    angle,
                )

        raise RuntimeError(
            "无法找到有效的机器人初始位置。"
        )

    # =====================================================
    # 圆形机器人与场景碰撞判断
    # =====================================================
    def _collides(
        self,
        x: float,
        y: float,
    ) -> bool:

        # 四周边界
        if (
            x - ROBOT_RADIUS
            <= WALL_THICKNESS
        ):
            return True

        if (
            x + ROBOT_RADIUS
            >= WINDOW_WIDTH
            - WALL_THICKNESS
        ):
            return True

        if (
            y - ROBOT_RADIUS
            <= WALL_THICKNESS
        ):
            return True

        if (
            y + ROBOT_RADIUS
            >= WINDOW_HEIGHT
            - WALL_THICKNESS
        ):
            return True

        # 内部矩形障碍物
        for rectangle in self.obstacles:

            nearest_x = min(
                max(
                    x,
                    rectangle.left,
                ),
                rectangle.right,
            )

            nearest_y = min(
                max(
                    y,
                    rectangle.top,
                ),
                rectangle.bottom,
            )

            difference_x = (
                x - nearest_x
            )

            difference_y = (
                y - nearest_y
            )

            if (
                difference_x
                * difference_x
                + difference_y
                * difference_y
                <= ROBOT_RADIUS
                * ROBOT_RADIUS
            ):
                return True

        return False

    # =====================================================
    # 读取9个传感器
    # 同时返回射线命中的表面法向量
    # =====================================================
    def _read_sensors(
        self,
    ):
        distances = []
        endpoints = []
        normals = []

        for (
            relative_angle_degrees
        ) in SENSOR_ANGLES:

            ray_angle = (
                self.robot_angle
                + math.radians(
                    relative_angle_degrees
                )
            )

            direction_x = math.cos(
                ray_angle
            )

            direction_y = -math.sin(
                ray_angle
            )

            (
                distance,
                endpoint,
                normal,
            ) = self._cast_ray(
                self.robot_x,
                self.robot_y,
                direction_x,
                direction_y,
            )

            distances.append(
                distance
            )

            endpoints.append(
                endpoint
            )

            normals.append(
                normal
            )

        return (
            np.asarray(
                distances,
                dtype=np.float32,
            ),
            endpoints,
            normals,
        )

    # =====================================================
    # 发射一条传感器射线
    # =====================================================
    def _cast_ray(
        self,
        origin_x,
        origin_y,
        direction_x,
        direction_y,
    ):
        nearest_distance = (
            SENSOR_RANGE
        )

        nearest_normal: Optional[
            np.ndarray
        ] = None

        for (
            rectangle
        ) in self.sensor_rectangles:

            result = (
                self._ray_rectangle_intersection(
                    origin_x,
                    origin_y,
                    direction_x,
                    direction_y,
                    rectangle,
                )
            )

            if result is None:
                continue

            (
                distance,
                normal,
            ) = result

            if (
                0.0
                <= distance
                < nearest_distance
            ):
                nearest_distance = (
                    distance
                )

                nearest_normal = normal

        endpoint = (
            origin_x
            + direction_x
            * nearest_distance,

            origin_y
            + direction_y
            * nearest_distance,
        )

        return (
            float(
                nearest_distance
            ),
            endpoint,
            nearest_normal,
        )

    # =====================================================
    # 计算射线和矩形各边的交点
    # 返回距离和表面单位法向量
    # =====================================================
    @staticmethod
    def _ray_rectangle_intersection(
        origin_x,
        origin_y,
        direction_x,
        direction_y,
        rectangle,
    ):
        epsilon = 1e-9

        candidates = []

        # 检查矩形左边和右边
        if abs(
            direction_x
        ) > epsilon:

            vertical_sides = (
                (
                    float(
                        rectangle.left
                    ),
                    np.asarray(
                        [-1.0, 0.0],
                        dtype=np.float32,
                    ),
                ),
                (
                    float(
                        rectangle.right
                    ),
                    np.asarray(
                        [1.0, 0.0],
                        dtype=np.float32,
                    ),
                ),
            )

            for (
                side_x,
                normal,
            ) in vertical_sides:

                distance = (
                    side_x - origin_x
                ) / direction_x

                hit_y = (
                    origin_y
                    + distance
                    * direction_y
                )

                if (
                    distance >= 0.0
                    and rectangle.top
                    - epsilon
                    <= hit_y
                    <= rectangle.bottom
                    + epsilon
                ):
                    candidates.append(
                        (
                            float(
                                distance
                            ),
                            normal,
                        )
                    )

        # 检查矩形上边和下边
        if abs(
            direction_y
        ) > epsilon:

            horizontal_sides = (
                (
                    float(
                        rectangle.top
                    ),
                    np.asarray(
                        [0.0, -1.0],
                        dtype=np.float32,
                    ),
                ),
                (
                    float(
                        rectangle.bottom
                    ),
                    np.asarray(
                        [0.0, 1.0],
                        dtype=np.float32,
                    ),
                ),
            )

            for (
                side_y,
                normal,
            ) in horizontal_sides:

                distance = (
                    side_y - origin_y
                ) / direction_y

                hit_x = (
                    origin_x
                    + distance
                    * direction_x
                )

                if (
                    distance >= 0.0
                    and rectangle.left
                    - epsilon
                    <= hit_x
                    <= rectangle.right
                    + epsilon
                ):
                    candidates.append(
                        (
                            float(
                                distance
                            ),
                            normal,
                        )
                    )

        if not candidates:
            return None

        minimum_distance = min(
            item[0]
            for item in candidates
        )

        # 射线正好命中角点时，合成两个法向量
        closest_normals = [
            normal
            for (
                distance,
                normal,
            ) in candidates
            if abs(
                distance
                - minimum_distance
            )
            <= 1e-6
        ]

        combined_normal = np.sum(
            closest_normals,
            axis=0,
        )

        normal_length = float(
            np.linalg.norm(
                combined_normal
            )
        )

        if normal_length <= epsilon:
            return None

        combined_normal = (
            combined_normal
            / normal_length
        )

        return (
            minimum_distance,
            combined_normal.astype(
                np.float32
            ),
        )

    # =====================================================
    # 找出前方三个传感器中最近的障碍物
    # =====================================================
    @staticmethod
    def _front_hit(
        sensor_distances,
        sensor_normals,
    ):
        nearest_index = min(
            FRONT_SENSOR_INDICES,
            key=lambda index: float(
                sensor_distances[index]
            ),
        )

        return (
            float(
                sensor_distances[
                    nearest_index
                ]
            ),
            sensor_normals[
                nearest_index
            ],
        )

    # =====================================================
    # 严格镜面反射方向
    #
    # r = d - 2(d·n)n
    # =====================================================
    def _apply_specular_reflection(
        self,
        normal,
    ) -> bool:

        if normal is None:
            return False

        direction = np.asarray(
            [
                math.cos(
                    self.robot_angle
                ),
                -math.sin(
                    self.robot_angle
                ),
            ],
            dtype=np.float64,
        )

        normal = np.asarray(
            normal,
            dtype=np.float64,
        )

        normal_length = float(
            np.linalg.norm(
                normal
            )
        )

        if normal_length <= 1e-12:
            return False

        normal = (
            normal
            / normal_length
        )

        dot_product = float(
            np.dot(
                direction,
                normal,
            )
        )

        reflected = (
            direction
            - 2.0
            * dot_product
            * normal
        )

        reflected_length = float(
            np.linalg.norm(
                reflected
            )
        )

        if reflected_length <= 1e-12:
            return False

        reflected = (
            reflected
            / reflected_length
        )

        # 转换回Pygame角度
        self.robot_angle = (
            math.atan2(
                -reflected[1],
                reflected[0],
            )
            % (
                2.0
                * math.pi
            )
        )

        return True

    def render(
        self,
    ):
        return None

    def close(
        self,
    ):
        pass
