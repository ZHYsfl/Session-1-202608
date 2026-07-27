"""DIRECT、REACTIVE 与学习策略控制接口。"""

from __future__ import annotations

import math

import torch
from policy import ActorCritic


def direct_action(observation: torch.Tensor, sensor_count: int) -> torch.Tensor:
    """只朝目标行驶，不使用测距避障。"""

    sin_goal = observation[:, sensor_count]
    cos_goal = observation[:, sensor_count + 1]
    goal_angle = torch.atan2(sin_goal, cos_goal)
    turn = (goal_angle / (math.pi / 2.0)).clamp(-1.0, 1.0)
    speed_ratio = (1.0 - 0.55 * turn.abs()).clamp(0.25, 1.0)
    return torch.stack((2.0 * speed_ratio - 1.0, turn), dim=1)


def reactive_action(observation: torch.Tensor, sensor_count: int) -> torch.Tensor:
    """使用目标吸引力与障碍排斥力构造确定性基线。"""

    rays = observation[:, :sensor_count]
    sin_goal = observation[:, sensor_count]
    cos_goal = observation[:, sensor_count + 1]
    angles = torch.arange(sensor_count, device=observation.device) * (
        2.0 * math.pi / sensor_count
    )
    proximity = ((0.65 - rays) / 0.65).clamp(0.0, 1.0).square()
    repel_x = -(proximity * torch.cos(angles)[None, :]).sum(dim=1) * 0.45
    repel_y = -(proximity * torch.sin(angles)[None, :]).sum(dim=1) * 0.65
    desired_x = cos_goal + repel_x
    desired_y = sin_goal + repel_y
    desired_angle = torch.atan2(desired_y, desired_x)
    turn = (desired_angle / (math.pi / 2.0)).clamp(-1.0, 1.0)

    front_mask = torch.cos(angles) >= 0.5
    front_clearance = rays[:, front_mask].min(dim=1).values
    clearance_speed = ((front_clearance - 0.10) / 0.45).clamp(0.05, 0.85)
    speed_ratio = clearance_speed * (1.0 - 0.65 * turn.abs())
    return torch.stack((2.0 * speed_ratio - 1.0, turn), dim=1)


@torch.no_grad()
def policy_action(model: ActorCritic, observation: torch.Tensor) -> torch.Tensor:
    """使用策略均值输出确定性动作。"""

    action, _, _, _, _ = model.get_action_and_value(observation, deterministic=True)
    return action
