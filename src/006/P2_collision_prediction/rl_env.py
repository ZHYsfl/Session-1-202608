"""完全基于 PyTorch 张量的批量局部避障环境。"""

from __future__ import annotations

import math
from typing import Any

import torch


class BatchedLocalAvoidanceEnv:
    """并行模拟圆形差速机器人、程序化通道和 360° 测距射线。"""

    def __init__(
        self,
        config: dict[str, Any],
        num_envs: int,
        device: torch.device,
        seed: int,
        *,
        training_noise: bool = False,
        auto_reset: bool = True,
        mix_previous_stages: bool = False,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs 必须为正数")
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.training_noise = training_noise
        self.auto_reset = auto_reset
        self.mix_previous_stages = mix_previous_stages

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
        self.max_gates = int(environment["max_gates"])
        if self.max_obstacles < 2 * self.max_gates:
            raise ValueError("max_obstacles 至少应为 max_gates 的两倍")
        self.stage_configs = list(environment["stages"])
        self.curriculum_stage = int(environment["initial_curriculum_stage"])
        self.previous_stage_ratio = float(environment["previous_stage_ratio"])
        self.wall_thickness_range = tuple(
            float(value) for value in environment["wall_thickness"]
        )
        self.travel_range = tuple(float(value) for value in environment["travel"])

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.ray_offsets = torch.arange(
            self.sensor_count, device=device, dtype=torch.float32
        ) * (2.0 * math.pi / self.sensor_count)

        self.position = torch.zeros((num_envs, 2), device=device)
        self.heading = torch.zeros(num_envs, device=device)
        self.goal = torch.zeros((num_envs, 2), device=device)
        # 障碍格式为 x、y、宽、高；宽高为 0 表示未启用。
        self.obstacles = torch.zeros((num_envs, self.max_obstacles, 4), device=device)
        self.last_action = torch.zeros((num_envs, 2), device=device)
        self.last_delta = torch.zeros((num_envs, 2), device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.path_length = torch.zeros(num_envs, device=device)
        self.episode_stage = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.gate_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.bend_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.horizontal = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.gate_centers = torch.zeros((num_envs, self.max_gates), device=device)
        self.gate_widths = torch.zeros((num_envs, self.max_gates), device=device)
        self.reset()

    @property
    def observation_dim(self) -> int:
        """观测由 R 条射线和 7 维辅助状态组成。"""

        return self.sensor_count + 7

    def set_curriculum_stage(self, stage: int) -> None:
        """设置当前课程阶段。"""

        self.curriculum_stage = max(0, min(int(stage), len(self.stage_configs) - 1))

    def get_generator_state(self) -> torch.Tensor:
        """返回环境随机数生成器状态，供训练断点保存。"""

        return self.generator.get_state()

    def set_generator_state(self, state: torch.Tensor) -> None:
        """恢复环境随机数生成器状态。"""

        self.generator.set_state(state.cpu())

    def _rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device=self.device)

    def _episode_stages(self, count: int) -> torch.Tensor:
        stages = torch.full(
            (count,),
            self.curriculum_stage,
            dtype=torch.long,
            device=self.device,
        )
        if (
            self.mix_previous_stages
            and self.curriculum_stage > 0
            and self.previous_stage_ratio > 0.0
        ):
            previous = self._rand((count,)) < self.previous_stage_ratio
            sampled = torch.floor(self._rand((count,)) * self.curriculum_stage).long()
            stages = torch.where(previous, sampled, stages)
        return stages

    def _stage_values(
        self,
        stages: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        values = torch.tensor(
            [float(stage[key]) for stage in self.stage_configs],
            device=self.device,
        )
        return values[stages]

    def _generate_curve(
        self,
        stages: torch.Tensor,
        gate_count: torch.Tensor,
        gap_widths: torch.Tensor,
        travel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成连续变化的通道中心线，并把折线长度控制在课程范围内。"""

        count = int(stages.numel())
        slots = torch.arange(self.max_gates, device=self.device)[None, :]
        active = slots < gate_count[:, None]
        random_bends = torch.floor(self._rand((count,)) * 3.0).long() + 2
        bends = torch.where(
            stages == 0,
            torch.ones_like(random_bends),
            torch.where(stages == 1, random_bends.clamp(max=2), random_bends),
        )
        bends = torch.minimum(bends, (gate_count - 1).clamp(min=1))

        phase = (self._rand((count,)) - 0.5) * 0.9
        direction = torch.where(
            self._rand((count,)) < 0.5,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        normalized_slot = (slots + 1.0) / (gate_count[:, None] + 1.0)
        wave = torch.cos(
            normalized_slot * (math.pi / 2.0) * bends[:, None] + phase[:, None]
        )
        wave = wave * direction[:, None]
        # 第一扇门必须显著偏离直线，确保直行策略无法穿过。
        first_sign = torch.where(
            wave[:, 0] < 0.0,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        wave[:, 0] = first_sign
        wave += (self._rand((count, self.max_gates)) - 0.5) * 0.18
        wave = wave.clamp(-1.0, 1.0)
        wave = torch.where(active, wave, torch.zeros_like(wave))

        amplitude_min = torch.empty(count, device=self.device)
        amplitude_max = torch.empty(count, device=self.device)
        for stage_index, stage in enumerate(self.stage_configs):
            mask = stages == stage_index
            amplitude_min[mask] = float(stage["curve_amplitude"][0])
            amplitude_max[mask] = float(stage["curve_amplitude"][1])
        configured_amplitude = amplitude_min + (
            amplitude_max - amplitude_min
        ) * self._rand((count,))

        # 第一扇门在膨胀机器人半径后仍需挡住中线。
        first_required = (gap_widths[:, 0] / 2.0 + self.radius + 3.0) / wave[
            :, 0
        ].abs().clamp(min=0.25)
        lower_scale = torch.maximum(configured_amplitude * 0.65, first_required)
        upper_scale = torch.maximum(configured_amplitude * 2.8, lower_scale + 12.0)

        ratio_min = torch.empty(count, device=self.device)
        ratio_max = torch.empty(count, device=self.device)
        for stage_index, stage in enumerate(self.stage_configs):
            mask = stages == stage_index
            ratio_min[mask] = float(stage["path_ratio"][0])
            ratio_max[mask] = float(stage["path_ratio"][1])
        target_ratio = ratio_min + (ratio_max - ratio_min) * self._rand((count,))

        longitudinal = normalized_slot * travel[:, None]
        longitudinal = torch.where(active, longitudinal, torch.zeros_like(longitudinal))

        def curve_ratio(scale: torch.Tensor) -> torch.Tensor:
            offsets = wave * scale[:, None]
            previous_x = torch.zeros(count, device=self.device)
            previous_y = torch.zeros(count, device=self.device)
            length = torch.zeros(count, device=self.device)
            for slot in range(self.max_gates):
                enabled = active[:, slot]
                delta_x = longitudinal[:, slot] - previous_x
                delta_y = offsets[:, slot] - previous_y
                segment = torch.hypot(delta_x, delta_y)
                length += torch.where(enabled, segment, torch.zeros_like(segment))
                previous_x = torch.where(enabled, longitudinal[:, slot], previous_x)
                previous_y = torch.where(enabled, offsets[:, slot], previous_y)
            length += torch.hypot(travel - previous_x, previous_y)
            return length / travel

        low = lower_scale
        high = upper_scale
        for _ in range(10):
            middle = (low + high) / 2.0
            ratio = curve_ratio(middle)
            low = torch.where(ratio < target_ratio, middle, low)
            high = torch.where(ratio < target_ratio, high, middle)
        offsets = wave * ((low + high) / 2.0)[:, None]
        return offsets, bends

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """重置全部环境或布尔掩码选中的环境。"""

        if mask is None:
            indices = torch.arange(self.num_envs, device=self.device)
        else:
            indices = torch.nonzero(mask, as_tuple=False).flatten()
        count = int(indices.numel())
        if count == 0:
            return self.observe()

        stages = self._episode_stages(count)
        stage_min_gates = self._stage_values(stages, "min_gates").long()
        stage_max_gates = self._stage_values(stages, "max_gates").long()
        gate_count = (
            stage_min_gates
            + torch.floor(
                self._rand((count,)) * (stage_max_gates - stage_min_gates + 1)
            ).long()
        )
        slots = torch.arange(self.max_gates, device=self.device)[None, :]
        active = slots < gate_count[:, None]

        gap_min = self._stage_values(stages, "min_gap")
        gap_max = self._stage_values(stages, "max_gap")
        gap_widths = gap_min[:, None] + (gap_max - gap_min)[:, None] * self._rand(
            (count, self.max_gates)
        )
        gap_widths = torch.where(active, gap_widths, torch.zeros_like(gap_widths))

        travel = self.travel_range[0] + (
            self.travel_range[1] - self.travel_range[0]
        ) * self._rand((count,))
        horizontal = self._rand((count,)) < 0.5
        reverse = self._rand((count,)) < 0.5
        long_center = torch.where(
            horizontal,
            torch.full((count,), self.width / 2.0, device=self.device),
            torch.full((count,), self.height / 2.0, device=self.device),
        )
        cross_center = torch.where(
            horizontal,
            torch.full((count,), self.height / 2.0, device=self.device),
            torch.full((count,), self.width / 2.0, device=self.device),
        )
        cross_center += (self._rand((count,)) - 0.5) * 16.0
        start_long = long_center - travel / 2.0
        goal_long = long_center + travel / 2.0

        offsets, bends = self._generate_curve(stages, gate_count, gap_widths, travel)
        gate_centers = cross_center[:, None] + offsets
        cross_size = torch.where(
            horizontal[:, None],
            torch.full_like(gate_centers, self.height),
            torch.full_like(gate_centers, self.width),
        )
        gate_centers = torch.maximum(
            gap_widths / 2.0 + 2.0,
            torch.minimum(gate_centers, cross_size - gap_widths / 2.0 - 2.0),
        )
        gate_long = (
            start_long[:, None]
            + ((slots + 1.0) / (gate_count[:, None] + 1.0)) * travel[:, None]
        )
        gate_long += (self._rand((count, self.max_gates)) - 0.5) * 2.0

        thickness = self.wall_thickness_range[0] + (
            self.wall_thickness_range[1] - self.wall_thickness_range[0]
        ) * self._rand((count, self.max_gates))
        half_gap = gap_widths / 2.0
        first_extent = gate_centers - half_gap
        second_start = gate_centers + half_gap
        cross_extent = cross_size

        horizontal_first = torch.stack(
            (
                gate_long - thickness / 2.0,
                torch.zeros_like(gate_long),
                thickness,
                first_extent,
            ),
            dim=2,
        )
        horizontal_second = torch.stack(
            (
                gate_long - thickness / 2.0,
                second_start,
                thickness,
                cross_extent - second_start,
            ),
            dim=2,
        )
        vertical_first = torch.stack(
            (
                torch.zeros_like(gate_long),
                gate_long - thickness / 2.0,
                first_extent,
                thickness,
            ),
            dim=2,
        )
        vertical_second = torch.stack(
            (
                second_start,
                gate_long - thickness / 2.0,
                cross_extent - second_start,
                thickness,
            ),
            dim=2,
        )
        first_walls = torch.where(
            horizontal[:, None, None],
            horizontal_first,
            vertical_first,
        )
        second_walls = torch.where(
            horizontal[:, None, None],
            horizontal_second,
            vertical_second,
        )
        walls = torch.stack((first_walls, second_walls), dim=2).flatten(1, 2)
        wall_active = active[:, :, None].expand(-1, -1, 2).reshape(count, -1)
        walls = torch.where(wall_active[:, :, None], walls, torch.zeros_like(walls))

        start = torch.stack(
            (
                torch.where(horizontal, start_long, cross_center),
                torch.where(horizontal, cross_center, start_long),
            ),
            dim=1,
        )
        goal = torch.stack(
            (
                torch.where(horizontal, goal_long, cross_center),
                torch.where(horizontal, cross_center, goal_long),
            ),
            dim=1,
        )
        swapped_start = torch.where(reverse[:, None], goal, start)
        swapped_goal = torch.where(reverse[:, None], start, goal)
        direction = torch.atan2(
            swapped_goal[:, 1] - swapped_start[:, 1],
            swapped_goal[:, 0] - swapped_start[:, 0],
        )
        direction += (self._rand((count,)) - 0.5) * 0.5

        self.position[indices] = swapped_start
        self.goal[indices] = swapped_goal
        self.heading[indices] = self._wrap_angle(direction)
        self.obstacles[indices] = 0.0
        self.obstacles[indices, : 2 * self.max_gates] = walls
        self.last_action[indices] = 0.0
        self.last_delta[indices] = 0.0
        self.step_count[indices] = 0
        self.path_length[indices] = 0.0
        self.episode_stage[indices] = stages
        self.gate_count[indices] = gate_count
        self.bend_count[indices] = bends
        self.horizontal[indices] = horizontal
        self.gate_centers[indices] = torch.where(
            active, gate_centers, torch.zeros_like(gate_centers)
        )
        self.gate_widths[indices] = gap_widths
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
            torch.where(direction_x < -epsilon, -origin_x / direction_x, infinity),
        )
        wall_y = torch.where(
            direction_y > epsilon,
            (self.height - origin_y) / direction_y,
            torch.where(direction_y < -epsilon, -origin_y / direction_y, infinity),
        )
        nearest = torch.minimum(wall_x, wall_y).clamp(min=0.0)

        rect = self.obstacles
        active = (rect[:, :, 2] > 0.0) & (rect[:, :, 3] > 0.0)
        lower_x = rect[:, None, :, 0]
        upper_x = lower_x + rect[:, None, :, 2]
        lower_y = rect[:, None, :, 1]
        upper_y = lower_y + rect[:, None, :, 3]
        ray_x = direction_x[:, :, None]
        ray_y = direction_y[:, :, None]
        origin_x_3d = origin_x[:, :, None]
        origin_y_3d = origin_y[:, :, None]
        safe_ray_x = torch.where(ray_x.abs() < epsilon, epsilon, ray_x)
        safe_ray_y = torch.where(ray_y.abs() < epsilon, epsilon, ray_y)
        first_x = (lower_x - origin_x_3d) / safe_ray_x
        second_x = (upper_x - origin_x_3d) / safe_ray_x
        first_y = (lower_y - origin_y_3d) / safe_ray_y
        second_y = (upper_y - origin_y_3d) / safe_ray_y
        enter = torch.maximum(
            torch.minimum(first_x, second_x),
            torch.minimum(first_y, second_y),
        )
        leave = torch.minimum(
            torch.maximum(first_x, second_x),
            torch.maximum(first_y, second_y),
        )
        parallel_x_invalid = (ray_x.abs() < epsilon) & (
            (origin_x_3d < lower_x) | (origin_x_3d > upper_x)
        )
        parallel_y_invalid = (ray_y.abs() < epsilon) & (
            (origin_y_3d < lower_y) | (origin_y_3d > upper_y)
        )
        valid = (
            active[:, None, :]
            & ~parallel_x_invalid
            & ~parallel_y_invalid
            & (leave >= torch.maximum(enter, torch.zeros_like(enter)))
        )
        obstacle_distance = torch.where(
            valid,
            enter.clamp(min=0.0),
            torch.full_like(enter, float("inf")),
        )
        nearest = torch.minimum(nearest, obstacle_distance.min(dim=2).values)
        return nearest.clamp(max=self.max_range)

    def _collision(self) -> torch.Tensor:
        outside = (
            (self.position[:, 0] - self.radius <= 0.0)
            | (self.position[:, 0] + self.radius >= self.width)
            | (self.position[:, 1] - self.radius <= 0.0)
            | (self.position[:, 1] + self.radius >= self.height)
        )
        rect = self.obstacles
        active = (rect[:, :, 2] > 0.0) & (rect[:, :, 3] > 0.0)
        closest_x = torch.maximum(
            rect[:, :, 0],
            torch.minimum(
                self.position[:, None, 0],
                rect[:, :, 0] + rect[:, :, 2],
            ),
        )
        closest_y = torch.maximum(
            rect[:, :, 1],
            torch.minimum(
                self.position[:, None, 1],
                rect[:, :, 1] + rect[:, :, 3],
            ),
        )
        distance_sq = (self.position[:, None, 0] - closest_x).square() + (
            self.position[:, None, 1] - closest_y
        ).square()
        obstacle_hit = ((distance_sq <= self.radius**2) & active).any(dim=1)
        return outside | obstacle_hit

    def observe(self) -> torch.Tensor:
        """构造 36 条射线加 7 维目标/控制历史观测。"""

        rays = self._ray_distances() / self.max_range
        noisy = self.training_noise & (self.episode_stage[:, None] >= 2)
        if bool(noisy.any()):
            noise = torch.randn(
                rays.shape, generator=self.generator, device=self.device
            )
            perturbed = rays + noise * self.noise_std_ratio
            dropout = self._rand(tuple(rays.shape)) < self.dropout_ratio
            perturbed = torch.where(dropout, torch.ones_like(rays), perturbed)
            rays = torch.where(noisy, perturbed, rays)
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
        front_mask = torch.cos(self.ray_offsets).clamp(min=0.0) > 0.5
        front_distance = previous_rays[:, front_mask].min(dim=1).values
        risk = ((self.safe_distance - front_distance) / self.safe_distance).clamp(
            0.0, 1.0
        )
        unsafe_forward = risk * speed_ratio.square() * (1.0 - 0.7 * angular_ratio.abs())
        delta_action = physical_action - self.last_action
        reward = (
            progress * float(reward_config["progress_scale"])
            + success.float() * float(reward_config["goal"])
            + collision.float() * float(reward_config["collision"])
            + timeout.float() * float(reward_config["timeout"])
            + float(reward_config["time"])
            + unsafe_forward * float(reward_config["directional_risk"])
            + delta_action.abs().sum(dim=1) * float(reward_config["smooth"])
        )
        self.last_delta = delta_action
        self.last_action = physical_action

        info = {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "stage": self.episode_stage.clone(),
            "gate_count": self.gate_count.clone(),
            "bend_count": self.bend_count.clone(),
            "movement": movement,
            "episode_path_length": torch.where(
                done, self.path_length, torch.zeros_like(self.path_length)
            ),
        }
        if self.auto_reset and bool(done.any()):
            self.reset(done)
        return self.observe(), reward, done, info

    def snapshot(self, index: int = 0) -> dict[str, object]:
        """把单个环境复制到 CPU，供 Pygame 和离线校准使用。"""

        if not 0 <= index < self.num_envs:
            raise IndexError("环境索引越界")
        rays = self._ray_distances()[index].detach().cpu().tolist()
        obstacles = [
            tuple(float(value) for value in rect)
            for rect in self.obstacles[index].detach().cpu().tolist()
            if rect[2] > 0.0 and rect[3] > 0.0
        ]
        gates = int(self.gate_count[index].detach().cpu())
        return {
            "position": tuple(
                float(value) for value in self.position[index].detach().cpu()
            ),
            "heading": float(self.heading[index].detach().cpu()),
            "goal": tuple(float(value) for value in self.goal[index].detach().cpu()),
            "obstacles": obstacles,
            "rays": rays,
            "path_length": float(self.path_length[index].detach().cpu()),
            "step_count": int(self.step_count[index].detach().cpu()),
            "stage": int(self.episode_stage[index].detach().cpu()),
            "gate_count": gates,
            "bend_count": int(self.bend_count[index].detach().cpu()),
            "horizontal": bool(self.horizontal[index].detach().cpu()),
            "gate_centers": [
                float(value)
                for value in self.gate_centers[index, :gates].detach().cpu()
            ],
            "gate_widths": [
                float(value) for value in self.gate_widths[index, :gates].detach().cpu()
            ],
        }
