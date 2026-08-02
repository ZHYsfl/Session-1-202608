# -*- coding: utf-8 -*-
"""
sac.py — SAC 智能体（api.md §6.3）

要点（与 api.md 完全对齐）：
  - 双 Q 网络 + target 网络，soft update τ=0.005
  - α 自动调节：target entropy = −ACT_DIM（=−2），log_alpha 走 Adam，lr 同 3e-4
  - lr=3e-4、batch=256、γ=0.99
  - 训练循环每 episode 结束后做 K = 本 episode 步数 次 update（约 1:1 更新比）
  - 网络来自 models.get_model_module()（009 model.py 或开发桩）
"""

import numpy as np
import torch
import torch.nn.functional as F

from config import (ALPHA_LOG_INIT, BATCH_SIZE, GAMMA, LR, TAU,
                    TARGET_ENTROPY)
from models import get_model_module


def _as_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


class SACAgent:
    def __init__(self, device, models: dict, lr: float = LR,
                 batch_size: int = BATCH_SIZE):
        self.device = device
        self.batch_size = batch_size

        self.actor = models["actor"]
        self.critic1 = models["critic1"]
        self.critic2 = models["critic2"]
        self.critic1_target = models["critic1_target"]
        self.critic2_target = models["critic2_target"]

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.opt_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        # α 自动调节（api.md §6.3：target entropy = −ACT_DIM）
        self.log_alpha = torch.tensor(ALPHA_LOG_INIT, device=device,
                                      requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = TARGET_ENTROPY

        self.update_step = 0  # 累计梯度更新次数（写入 checkpoint meta）

    @property
    def alpha(self):
        return torch.exp(self.log_alpha.detach())

    # ---------- 动作 ----------
    @torch.no_grad()
    def select_action(self, vec: np.ndarray, deterministic: bool = False):
        """单条归一化观测 → (-1,1)² 动作（numpy）。
        deterministic=True 为评估用确定性策略：直接取 mean（api.md §6.7）。"""
        obs = _as_tensor(vec, self.device).unsqueeze(0)
        if deterministic:
            mean, _ = self.actor.forward(obs)
            return mean[0].cpu().numpy()
        action, _, _ = self.actor.sample(obs)
        return action[0].cpu().numpy()

    # ---------- 训练 ----------
    def update(self, buffer, batch_size: int | None = None):
        """从 buffer 采样一个 batch 做一次完整 SAC 更新（critic×2 + actor + α + soft update）。
        返回本轮各 loss 的均值（日志用）。"""
        bs = batch_size or self.batch_size
        obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(bs)

        obs = _as_tensor(obs_b, self.device)
        act = _as_tensor(act_b, self.device)
        rew = _as_tensor(rew_b, self.device).unsqueeze(1)
        next_obs = _as_tensor(next_obs_b, self.device)
        done = _as_tensor(done_b, self.device).unsqueeze(1)

        # ---- critic loss（clipped double-Q + 熵正则 target）----
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q1_next = self.critic1_target(next_obs, next_action)
            q2_next = self.critic2_target(next_obs, next_action)
            q_next_min = torch.min(q1_next, q2_next)
            # done_mask=1（collision/goal_reached）→ 不 bootstrap；timeout → 照常
            target = rew + GAMMA * (1.0 - done) * (
                q_next_min - self.alpha * next_log_prob.unsqueeze(1))

        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        loss_c1 = F.mse_loss(q1, target)
        loss_c2 = F.mse_loss(q2, target)

        self.opt_critic1.zero_grad()
        loss_c1.backward()
        self.opt_critic1.step()
        self.opt_critic2.zero_grad()
        loss_c2.backward()
        self.opt_critic2.step()

        # ---- actor loss（最大化 Q − α·log π）----
        act_new, log_prob, _ = self.actor.sample(obs)
        q1a = self.critic1(obs, act_new)
        q2a = self.critic2(obs, act_new)
        q_min = torch.min(q1a, q2a)
        loss_actor = (self.alpha * log_prob - q_min).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        # ---- α loss（自动调节，target entropy = −ACT_DIM）----
        loss_alpha = -(self.log_alpha
                       * (log_prob + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        # ---- target soft update（τ=0.005）----
        with torch.no_grad():
            for tp, p in zip(self.critic1_target.parameters(),
                             self.critic1.parameters()):
                tp.copy_(TAU * p + (1.0 - TAU) * tp)
            for tp, p in zip(self.critic2_target.parameters(),
                             self.critic2.parameters()):
                tp.copy_(TAU * p + (1.0 - TAU) * tp)

        self.update_step += 1
        return {"loss_critic1": float(loss_c1.item()),
                "loss_critic2": float(loss_c2.item()),
                "loss_actor": float(loss_actor.item()),
                "loss_alpha": float(loss_alpha.item()),
                "alpha": float(self.alpha.item())}

    # ---------- 优化器存取（挂进 checkpoint 的 optimizers 字典） ----------
    def optimizers_dict(self) -> dict:
        return {"actor": self.opt_actor, "critic1": self.opt_critic1,
                "critic2": self.opt_critic2, "alpha": self.opt_alpha}

    def load_optimizers(self, optimizers: dict) -> None:
        """把 checkpoint 里的优化器状态加载回来（恢复训练用）。"""
        for k, opt in self.optimizers_dict().items():
            if k in optimizers:
                opt.load_state_dict(optimizers[k])
