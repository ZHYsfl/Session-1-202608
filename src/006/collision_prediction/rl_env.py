"""完全基于 PyTorch 张量的批量局部避障环境。"""

from __future__ import annotations

import math
from typing import Any

import torch


class BatchedLocalAvoidanceEnv:
    """并行模拟圆形差速机器人、矩形静态障碍和 360° 测距射线。"""

    def __init__(
        self,
        config: dict[str, Any],
        num_envs: int,
        device: torch.device,
        seed: int,
        *,
        training_noise: bool = False,
        auto_reset: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs 必须为正数")
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.training_noise = training_noise
        self.auto_reset = auto_reset

        window = config["window"]
        robot = config["robot"]
        sensors = config["sensors"]
        episode = config["episode"]
        environment = config["environment"]

        self.width = float(window["width"])
        self.height = float(window["height"])
        self.radius = float(robot["radius"])
        self.max_speed = float(robot["max_speed"])
        self.max_angular_speed = float(robot["max_angular_speed"])
        self.dt = float(robot["control_dt"])
        self.sensor_count = int(sensors["count"])
        self.max_range = float(sensors["max_range"])
        self.noise_std_ratio = float(sensors["noise_std_ratio"])
        self.dropout_ratio = float(sensors["dropout_ratio"])
        self.goal_radius = float(episode["goal_radius"])
        self.max_steps = int(episode["max_steps"])
        self.safe_distance = float(episode["safe_distance"])
        self.max_obstacles = int(environment["max_obstacles"])
        self.curriculum_stage = int(environment["initial_curriculum_stage"])

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.ray_offsets = torch.arange(
            self.sensor_count, device=device, dtype=torch.float32
        ) * (2.0 * math.pi / self.sensor_count)

        self.position = torch.zeros((num_envs, 2), device=device)
        self.heading = torch.zeros(num_envs, device=device)
        self.goal = torch.zeros((num_envs, 2), device=device)
        # 障碍物格式为 x、y、宽、高；宽高为 0 表示未启用。
        self.obstacles = torch.zeros((num_envs, self.max_obstacles, 4), device=device)
        self.last_action = torch.zeros((num_envs, 2), device=device)
        self.last_delta = torch.zeros((num_envs, 2), device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.path_length = torch.zeros(num_envs, device=device)
        self.reset()

    @property
    def observation_dim(self) -> int:
        """观测由 R 条射线和 7 维辅助状态组成。"""

        return self.sensor_count + 7

    def set_curriculum_stage(self, stage: int) -> None:
        """设置课程阶段，取值限制为 0、1、2。"""

        self.curriculum_stage = max(0, min(int(stage), 2))

    def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device=self.device)

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """重置全部环境或布尔掩码选中的环境。"""

        if mask is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = torch.nonzero(mask, as_tuple=False).flatten()
        count = int(indices.numel())
        if count == 0:
            return self.observe()

        horizontal = self._rand((count,)) < 0.5
        reverse = self._rand((count,)) < 0.5
        travel = 130.0 + 35.0 * self._rand((count,))
        center_x = self.width * (0.45 + 0.10 * self._rand((count,)))
        center_y = self.height * (0.40 + 0.20 * self._rand((count,)))

        start = torch.stack((center_x - travel / 2.0, center_y), dim=1)
        goal = torch.stack((center_x + travel / 2.0, center_y), dim=1)

        # 先在水平规范坐标中构造障碍，再按需旋转为竖直任务。
        blocker_w = 22.0 + 12.0 * self._rand((count,))
        blocker_h = 62.0 + 28.0 * self._rand((count,))
        side = torch.where(
            self._rand((count,)) < 0.5,
            torch.full((count,), -1.0, device=self.device),
            torch.ones(count, device=self.device),
        )
        if self.curriculum_stage == 0:
            blocker_offset = side * (42.0 + 12.0 * self._rand((count,)))
        else:
            blocker_offset = side * (8.0 * self._rand((count,)) - 4.0)

        blocker = torch.stack(
            (
                center_x - blocker_w / 2.0,
                center_y + blocker_offset - blocker_h / 2.0,
                blocker_w,
                blocker_h,
            ),
            dim=1,
        )
        second_w = 50.0 + 22.0 * self._rand((count,))
        second_h = 18.0 + 10.0 * self._rand((count,))
        second = torch.stack(
            (
                center_x - side * 22.0 - second_w / 2.0,
                center_y + side * (blocker_h / 2.0 + 18.0),
                second_w,
                second_h,
            ),
            dim=1,
        )
        if self.curriculum_stage < 2:
            second[:, 2:] = 0.0

        # 将规范任务绕场地中心旋转 90°，得到竖直任务。
        def rotate_points(points: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                (
                    self.width / 2.0 + points[:, 1] - self.height / 2.0,
                    self.height / 2.0 - points[:, 0] + self.width / 2.0,
                ),
                dim=1,
            )

        rotated_start = rotate_points(start)
        rotated_goal = rotate_points(goal)
        start = torch.where(horizontal[:, None], start, rotated_start)
        goal = torch.where(horizontal[:, None], goal, rotated_goal)

        def rotate_rects(rects: torch.Tensor) -> torch.Tensor:
            rect_center = rects[:, :2] + rects[:, 2:] / 2.0
            rotated_center = rotate_points(rect_center)
            rotated_size = torch.stack((rects[:, 3], rects[:, 2]), dim=1)
            return torch.cat((rotated_center - rotated_size / 2.0, rotated_size), dim=1)

        blocker = torch.where(horizontal[:, None], blocker, rotate_rects(blocker))
        second = torch.where(horizontal[:, None], second, rotate_rects(second))

        swapped_start = torch.where(reverse[:, None], goal, start)
        swapped_goal = torch.where(reverse[:, None], start, goal)
        direction = torch.atan2(
            swapped_goal[:, 1] - swapped_start[:, 1],
            swapped_goal[:, 0] - swapped_start[:, 0],
        )
        direction += (self._rand((count,)) - 0.5) * 0.6

        self.position[indices] = swapped_start
        self.goal[indices] = swapped_goal
        self.heading[indices] = self._wrap_angle(direction)
        self.obstacles[indices, 0] = blocker
        self.obstacles[indices, 1] = second
        self.last_action[indices] = 0.0
        self.last_delta[indices] = 0.0
        self.step_count[indices] = 0
        self.path_length[indices] = 0.0
        return self.observe()

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi

    def _ray_distances(self) -> torch.Tensor:
        """批量计算每条射线到最近矩形或场地边界的距离。"""

        angles = self.heading[:, None] + self.ray_offsets[None, :]
        direction_x = torch.cos(angles)
        direction_y = torch.sin(angles)
        origin_x = self.position[:, 0, None]
        origin_y = self.position[:, 1, None]
        infinity = torch.full_like(direction_x, float("inf"))
        epsilon = 1e-6

        wall_x = torch.where(
            direction_x > epsilon,
            (self.width - origin_x) / direction_x,
            torch.where(
                direction_x < -epsilon,
                -origin_x / direction_x,
                infinity,
            ),
        )
        wall_y = torch.where(
            direction_y > epsilon,
            (self.height - origin_y) / direction_y,
            torch.where(
                direction_y < -epsilon,
                -origin_y / direction_y,
                infinity,
            ),
        )
        nearest = torch.minimum(wall_x, wall_y)

        rect = self.obstacles
        active = (rect[:, :, 2] > 0.0) & (rect[:, :, 3] > 0.0)
        xmin = rect[:, None, :, 0]
        ymin = rect[:, None, :, 1]
        xmax = xmin + rect[:, None, :, 2]
        ymax = ymin + rect[:, None, :, 3]
        ox = origin_x[:, :, None]
        oy = origin_y[:, :, None]
        dx = direction_x[:, :, None]
        dy = direction_y[:, :, None]
        safe_dx = torch.where(dx.abs() < epsilon, torch.full_like(dx, epsilon), dx)
        safe_dy = torch.where(dy.abs() < epsilon, torch.full_like(dy, epsilon), dy)
        tx_min = torch.minimum((xmin - ox) / safe_dx, (xmax - ox) / safe_dx)
        tx_max = torch.maximum((xmin - ox) / safe_dx, (xmax - ox) / safe_dx)
        ty_min = torch.minimum((ymin - oy) / safe_dy, (ymax - oy) / safe_dy)
        ty_max = torch.maximum((ymin - oy) / safe_dy, (ymax - oy) / safe_dy)
        entry = torch.maximum(tx_min, ty_min)
        exit_distance = torch.minimum(tx_max, ty_max)
        hit = (exit_distance >= torch.maximum(entry, torch.zeros_like(entry))) & active[
            :, None, :
        ]
        obstacle_distance = torch.where(
            hit, torch.clamp(entry, min=0.0), torch.full_like(entry, float("inf"))
        )
        nearest = torch.minimum(nearest, obstacle_distance.min(dim=2).values)
        return torch.clamp(nearest, min=0.0, max=self.max_range)

    def _collision(self) -> torch.Tensor:
        outside = (
            (self.position[:, 0] - self.radius <= 0.0)
            | (self.position[:, 0] + self.radius >= self.width)
            | (self.position[:, 1] - self.radius <= 0.0)
            | (self.position[:, 1] + self.radius >= self.height)
        )
        rect = self.obstacles
        active = (rect[:, :, 2] > 0.0) & (rect[:, :, 3] > 0.0)
        closest_x = torch.clamp(
            self.position[:, 0, None],
            min=rect[:, :, 0],
            max=rect[:, :, 0] + rect[:, :, 2],
        )
        closest_y = torch.clamp(
            self.position[:, 1, None],
            min=rect[:, :, 1],
            max=rect[:, :, 1] + rect[:, :, 3],
        )
        distance_sq = (self.position[:, 0, None] - closest_x).square() + (
            self.position[:, 1, None] - closest_y
        ).square()
        obstacle_hit = ((distance_sq <= self.radius**2) & active).any(dim=1)
        return outside | obstacle_hit

    def observe(self) -> torch.Tensor:
        """构造 36 条射线加 7 维目标/控制历史观测。"""

        rays = self._ray_distances() / self.max_range
        if self.training_noise and self.curriculum_stage >= 2:
            noise = torch.randn(
                rays.shape, generator=self.generator, device=self.device
            )
            rays = rays + noise * self.noise_std_ratio
            dropout = self._rand(tuple(rays.shape)) < self.dropout_ratio
            rays = torch.where(dropout, torch.ones_like(rays), rays)
        rays = rays.clamp(0.0, 1.0)

        goal_delta = self.goal - self.position
        goal_angle = self._wrap_angle(
            torch.atan2(goal_delta[:, 1], goal_delta[:, 0]) - self.heading
        )
        goal_distance = torch.linalg.vector_norm(goal_delta, dim=1)
        auxiliary = torch.stack(
            (
                torch.sin(goal_angle),
                torch.cos(goal_angle),
                self.last_action[:, 0],
                self.last_action[:, 1],
                self.last_delta[:, 0],
                self.last_delta[:, 1],
                (goal_distance / self.max_range).clamp(0.0, 1.0),
            ),
            dim=1,
        )
        return torch.cat((rays, auxiliary), dim=1)

    def step(
        self, normalized_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """执行一步连续控制，并在终止后按需自动重置。"""

        if normalized_action.shape != (self.num_envs, 2):
            raise ValueError("动作形状必须为 [num_envs, 2]")
        action = normalized_action.clamp(-1.0, 1.0)
        speed_ratio = (action[:, 0] + 1.0) / 2.0
        angular_ratio = action[:, 1]
        physical_action = torch.stack((speed_ratio, angular_ratio), dim=1)

        previous_distance = torch.linalg.vector_norm(self.goal - self.position, dim=1)
        previous_rays = self._ray_distances()
        previous_position = self.position.clone()

        angular_speed = angular_ratio * self.max_angular_speed
        speed = speed_ratio * self.max_speed
        self.heading = self._wrap_angle(self.heading + angular_speed * self.dt)
        self.position[:, 0] += torch.cos(self.heading) * speed * self.dt
        self.position[:, 1] += torch.sin(self.heading) * speed * self.dt
        movement = torch.linalg.vector_norm(self.position - previous_position, dim=1)
        self.path_length += movement
        self.step_count += 1

        new_distance = torch.linalg.vector_norm(self.goal - self.position, dim=1)
        collision = self._collision()
        success = (new_distance <= self.goal_radius) & ~collision
        timeout = (self.step_count >= self.max_steps) & ~collision & ~success
        done = collision | success | timeout

        reward_config = self.config["reward"]
        progress = previous_distance - new_distance
        risk = (
            (self.safe_distance - previous_rays.min(dim=1).values) / self.safe_distance
        ).clamp(0.0, 1.0)
        delta_action = physical_action - self.last_action
        reward = (
            progress * float(reward_config["progress_scale"])
            + success.float() * float(reward_config["goal"])
            + collision.float() * float(reward_config["collision"])
            + float(reward_config["time"])
            + risk * speed_ratio * float(reward_config["risk"])
            + delta_action.abs().sum(dim=1) * float(reward_config["smooth"])
        )
        self.last_delta = delta_action
        self.last_action = physical_action

        info = {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "movement": movement,
            "episode_path_length": torch.where(
                done, self.path_length, torch.zeros_like(self.path_length)
            ),
        }
        if self.auto_reset and bool(done.any()):
            self.reset(done)
        return self.observe(), reward, done, info

    def snapshot(self, index: int = 0) -> dict[str, object]:
        """把单个环境复制到 CPU，供 Pygame 绘制。"""

        if not 0 <= index < self.num_envs:
            raise IndexError("环境索引越界")
        rays = self._ray_distances()[index].detach().cpu().tolist()
        obstacles = [
            tuple(float(value) for value in rect)
            for rect in self.obstacles[index].detach().cpu().tolist()
            if rect[2] > 0.0 and rect[3] > 0.0
        ]
        return {
            "position": tuple(
                float(value) for value in self.position[index].detach().cpu().tolist()
            ),
            "heading": float(self.heading[index].detach().cpu()),
            "goal": tuple(
                float(value) for value in self.goal[index].detach().cpu().tolist()
            ),
            "obstacles": obstacles,
            "rays": rays,
            "path_length": float(self.path_length[index].detach().cpu()),
            "step_count": int(self.step_count[index].detach().cpu()),
        }
