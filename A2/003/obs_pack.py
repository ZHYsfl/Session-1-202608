# -*- coding: utf-8 -*-
"""
obs_pack.py — 观测打包/归一化 与 动作缩放（api.md §4）

分工红线（api.md §4）：001 发原始物理量；003 负责归一化打包；
009 的网络只见归一化向量，永远工作在 (-1,1)² 动作空间；
001 收到的是物理单位的速度指令。任何一侧多做一次缩放，动作就错一倍。

2026-08-02 升级：帧堆叠。obs 由 68 维升级为 132 维：
    [0:64]   当前帧雷达（归一化）
    [64:128] 上一帧雷达（归一化；episode 首帧复制当前帧）
    [128]    goal.dist   min(d,10)/10
    [129]    goal.bearing b/π
    [130]    vel.v       clip(v,±v_max)/v_max
    [131]    vel.w       clip(w,±w_max)/w_max
"""

import math

import numpy as np

from config import DIST_NORM  # api.md §4: min(d, 10.0) / 10.0，= 场地对角线约 2 倍

ONE_OVER_PI = 1.0 / math.pi

LIDAR_COUNT = 64
LIDAR_STACK = 2
OBS_DIM = LIDAR_COUNT * LIDAR_STACK + 4   # 132


class ObsPacker:
    """2 帧雷达堆叠打包器。

    server 照发单帧 obs，打包侧维护上一帧雷达形成堆叠；
    episode 开始（reset）时必须调用 reset() 清空历史。
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prev_lidar: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_lidar = None

    def pack(self, obs: dict) -> np.ndarray:
        lidar_max = float(self.cfg["lidar_max_range"])
        v_max = float(self.cfg["v_max"])
        w_max = float(self.cfg["w_max"])

        lidar = np.asarray(obs["lidar"], dtype=np.float32)
        if lidar.shape[0] != LIDAR_COUNT:
            raise ValueError(
                f"lidar 维数不符: 期望 {LIDAR_COUNT}, 收到 {lidar.shape[0]}")
        cur = np.clip(lidar, 0.0, lidar_max) / lidar_max
        if self.prev_lidar is None:
            self.prev_lidar = cur.copy()   # 首帧：历史=当前

        vec = np.empty(OBS_DIM, dtype=np.float32)
        vec[0:LIDAR_COUNT] = cur
        vec[LIDAR_COUNT:2 * LIDAR_COUNT] = self.prev_lidar
        vec[2 * LIDAR_COUNT] = min(obs["goal"]["dist"], DIST_NORM) / DIST_NORM
        vec[2 * LIDAR_COUNT + 1] = obs["goal"]["bearing"] * ONE_OVER_PI
        vec[2 * LIDAR_COUNT + 2] = np.clip(obs["vel"]["v"], -v_max, v_max) / v_max
        vec[2 * LIDAR_COUNT + 3] = np.clip(obs["vel"]["w"], -w_max, w_max) / w_max

        self.prev_lidar = cur.copy()
        return vec


def pack_obs(obs: dict, cfg: dict) -> np.ndarray:
    """兼容入口：无历史时的单帧打包（等效于 ObsPacker 首帧）。"""
    return ObsPacker(cfg).pack(obs)


def scale_action(a01: np.ndarray, cfg: dict) -> tuple:
    """
    网络动作 → 底盘物理指令（api.md §4 动作映射）：
        v = a0 * v_max    （a0 ∈ (-1,1) → v ∈ (-0.5, 0.5) m/s）
        w = a1 * w_max    （a1 ∈ (-1,1) → w ∈ (-1.5, 1.5) rad/s）
    返回 (v, w) 浮点数，直接写入 action 消息。
    """
    a0, a1 = float(a01[0]), float(a01[1])
    return a0 * float(cfg["v_max"]), a1 * float(cfg["w_max"])
