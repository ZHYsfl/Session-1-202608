# -*- coding: utf-8 -*-
"""
reward.py — 奖励函数（api.md §6.4）与 终止掩码（api.md §6.5）

奖励默认系数集中在 config.py，改动必须记录进实验日志（README「实验记录」）。
"""

import numpy as np

from config import (D_SAFE, R_COLLISION, R_GOAL_REACHED, R_TIME_STEP,
                    W_APPROACH, W_DANGER, W_SMOOTH)


def compute_reward(obs_prev: dict, a01: np.ndarray, a01_prev: np.ndarray,
                   obs_next: dict) -> float:
    """
    api.md §6.4 默认奖励（每一步，作用于 (obs_prev → action → obs_next)）：

        r = 5.0 × (dist_{t-1} − dist_t)        # 朝目标靠近
            + 200   (若 goal_reached)
            − 200   (若 collision)
            − 0.1                              # 时间惩罚，每步
            − 3.0 × max(0, 0.5 − min(lidar_t)) # 障碍接近惩罚（run2 新增）
            − 0.5 × (a0 − a0_prev)² − 0.5 × (a1 − a1_prev)²   # 平滑惩罚

    参数：
        obs_prev : 上一步 obs 消息（step t-1 的观测）
        a01      : 本步网络动作（(-1,1)²，未缩放）
        a01_prev : 上一步网络动作；episode 第一步传与 a01 相同值 → 平滑惩罚为 0
        obs_next : 本步执行后返回的 obs 消息（step t）
    """
    dist_prev = obs_prev["goal"]["dist"]
    dist_next = obs_next["goal"]["dist"]

    r = W_APPROACH * (dist_prev - dist_next)
    if obs_next["flags"]["goal_reached"]:
        r += R_GOAL_REACHED
    if obs_next["flags"]["collision"]:
        r += R_COLLISION
    r += R_TIME_STEP
    r -= W_DANGER * max(0.0, D_SAFE - float(np.min(obs_next["lidar"])))
    r -= W_SMOOTH * (float(a01[0]) - float(a01_prev[0])) ** 2
    r -= W_SMOOTH * (float(a01[1]) - float(a01_prev[1])) ** 2
    return float(r)


def done_mask(obs_next: dict) -> float:
    """
    api.md §6.5 终止掩码（SAC bootstrap 关键）：
        collision 或 goal_reached → 1（真终止，target 不 bootstrap）
        timeout                → 0（人为截断，target 照常 bootstrap）
    返回 float（0.0 / 1.0），直接参与 γ·(1−mask) 计算。
    """
    if obs_next["flags"]["collision"] or obs_next["flags"]["goal_reached"]:
        return 1.0
    return 0.0


def outcome_of(obs_next: dict) -> str:
    """episode 结局文案（日志用）：collision / goal_reached / timeout。"""
    f = obs_next["flags"]
    if f["collision"]:
        return "collision"
    if f["goal_reached"]:
        return "goal_reached"
    return "timeout"
