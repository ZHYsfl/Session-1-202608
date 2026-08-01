# -*- coding: utf-8 -*-
"""
buffer.py — 经验回放缓冲（api.md §6.3）

存 (vec, a01, r, next_vec, done_mask)，容量 200 000。
numpy 环形数组实现，float32 存储，避免 Python 对象逐条驻留内存。
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pos = 0
        self.size = 0

        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.action = np.empty((capacity, act_dim), dtype=np.float32)
        self.reward = np.empty((capacity,), dtype=np.float32)
        self.next_obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.done = np.empty((capacity,), dtype=np.float32)

    def push(self, vec, a01, r, next_vec, mask):
        """mask 为 done_mask（0.0/1.0，见 reward.py）。"""
        self.obs[self.pos] = vec
        self.action[self.pos] = a01
        self.reward[self.pos] = r
        self.next_obs[self.pos] = next_vec
        self.done[self.pos] = mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        """均匀采样 batch 条，返回五元组 numpy 数组。"""
        idx = np.random.randint(0, self.size, size=batch)
        return (self.obs[idx], self.action[idx], self.reward[idx],
                self.next_obs[idx], self.done[idx])

    def __len__(self):
        return self.size
