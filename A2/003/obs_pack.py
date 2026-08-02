# -*- coding: utf-8 -*-
"""
obs_pack.py — 观测打包/归一化 与 动作缩放（api.md §4）

分工红线（api.md §4）：001 发原始物理量；003 负责归一化打包；
009 的网络只见归一化向量，永远工作在 (-1,1)² 动作空间；
001 收到的是物理单位的速度指令。任何一侧多做一次缩放，动作就错一倍。
"""

import math

import numpy as np

from config import DIST_NORM  # api.md §4: min(d, 10.0) / 10.0，= 场地对角线约 2 倍

ONE_OVER_PI = 1.0 / math.pi


def pack_obs(obs: dict, cfg: dict) -> np.ndarray:
    """
    把一条 obs 消息打包成 68 维归一化观测向量（api.md §4 表，顺序固定）。

    | 下标       | 来源          | 归一化公式                        | 结果范围 |
    | [0:64]     | lidar[i]      | clip(x,0,max_range)/max_range     | [0,1]   |
    | [64]       | goal.dist     | min(d,10)/10                      | [0,1]   |
    | [65]       | goal.bearing  | b/π                               | (-1,1]  |
    | [66]       | vel.v         | clip(v,±v_max)/v_max              | [-1,1]  |
    | [67]       | vel.w         | clip(w,±w_max)/w_max              | [-1,1]  |

    cfg 为 hello.config（含 lidar_count / lidar_max_range / v_max / w_max）。
    返回 np.float32 数组 shape (68,)。
    """
    lidar_max = float(cfg["lidar_max_range"])
    v_max = float(cfg["v_max"])
    w_max = float(cfg["w_max"])

    lidar = np.asarray(obs["lidar"], dtype=np.float32)
    if lidar.shape[0] != cfg["lidar_count"]:
        raise ValueError(
            f"lidar 维数不符: 期望 {cfg['lidar_count']}, 收到 {lidar.shape[0]}")

    vec = np.empty(68, dtype=np.float32)
    vec[0:64] = np.clip(lidar, 0.0, lidar_max) / lidar_max
    vec[64] = min(obs["goal"]["dist"], DIST_NORM) / DIST_NORM
    vec[65] = obs["goal"]["bearing"] * ONE_OVER_PI      # 方位角在 (-π, π]
    vec[66] = np.clip(obs["vel"]["v"], -v_max, v_max) / v_max
    vec[67] = np.clip(obs["vel"]["w"], -w_max, w_max) / w_max
    return vec


def scale_action(a01: np.ndarray, cfg: dict) -> tuple:
    """
    网络动作 → 底盘物理指令（api.md §4 动作映射）：
        v = a0 * v_max    （a0 ∈ (-1,1) → v ∈ (-0.5, 0.5) m/s）
        w = a1 * w_max    （a1 ∈ (-1,1) → w ∈ (-1.5, 1.5) rad/s）
    返回 (v, w) 浮点数，直接写入 action 消息。
    """
    a0, a1 = float(a01[0]), float(a01[1])
    return a0 * float(cfg["v_max"]), a1 * float(cfg["w_max"])
