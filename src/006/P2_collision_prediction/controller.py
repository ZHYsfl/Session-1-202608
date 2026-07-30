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
    """使用局部宽通道选择构造无记忆的确定性反应式基线。"""

    rays = observation[:, :sensor_count]
    sin_goal = observation[:, sensor_count]
    cos_goal = observation[:, sensor_count + 1]
    angles = torch.arange(sensor_count, device=observation.device) * (
        2.0 * math.pi / sensor_count
    )
    signed_angles = torch.remainder(angles + math.pi, 2.0 * math.pi) - math.pi
    goal_angle = torch.atan2(sin_goal, cos_goal)

    # 三条相邻射线都较远才视为可通行方向，避免机器人圆周擦到门框。
    corridor_clearance = torch.minimum(
        rays,
        torch.minimum(
            torch.minimum(
                torch.roll(rays, 1, dims=1),
                torch.roll(rays, -1, dims=1),
            ),
            torch.minimum(
                torch.roll(rays, 2, dims=1),
                torch.roll(rays, -2, dims=1),
            ),
        ),
    )
    goal_alignment = torch.cos(signed_angles[None, :] - goal_angle[:, None])
    forward_preference = torch.cos(signed_angles)[None, :]
    score = 4.0 * corridor_clearance + 0.50 * goal_alignment + 0.10 * forward_preference
    # 机器人不能倒车，禁止选择身后的“宽阔区域”作为当前行驶方向。
    score = torch.where(
        signed_angles.abs()[None, :] <= math.pi / 2.0,
        score,
        torch.full_like(score, -torch.inf),
    )
    best_index = score.argmax(dim=1)
    desired_angle = signed_angles[best_index]
    desired_turn = (desired_angle / (math.pi / 3.0)).clamp(-1.0, 1.0)
    previous_turn = observation[:, sensor_count + 3]
    # 使用上一控制量形成轻微迟滞，避免门口射线排序变化导致左右抖动。
    turn = (0.55 * previous_turn + 0.45 * desired_turn).clamp(-1.0, 1.0)

    front_clearance = corridor_clearance[:, 0]
    selected_clearance = corridor_clearance.gather(1, best_index[:, None]).squeeze(1)
    clearance_speed = ((front_clearance - 0.07) / 0.24).clamp(0.0, 0.65)
    selected_speed = ((selected_clearance - 0.07) / 0.22).clamp(0.0, 0.75)
    speed_ratio = torch.minimum(clearance_speed, selected_speed)
    speed_ratio *= 1.0 - 0.78 * turn.abs()
    return torch.stack((2.0 * speed_ratio - 1.0, turn), dim=1)


@torch.no_grad()
def policy_action(model: ActorCritic, observation: torch.Tensor) -> torch.Tensor:
    """使用策略均值输出确定性动作。"""

    action, _, _, _, _ = model.get_action_and_value(observation, deterministic=True)
    return action
