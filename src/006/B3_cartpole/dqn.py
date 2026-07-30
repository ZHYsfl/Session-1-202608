"""Cart-Pole 的 PyTorch DQN、经验回放与 checkpoint。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """输入 4 维状态，输出左右动作的 Q 值。"""

    def __init__(self, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = 4
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 2))
        self.network = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        final = self.network[-1]
        if isinstance(final, nn.Linear):
            nn.init.orthogonal_(final.weight, 0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """固定容量的 NumPy 环形经验回放。"""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须为正数")
        self.capacity = capacity
        self.states = np.zeros((capacity, 4), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 4), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("经验数量小于 batch_size")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )


class DQNAgent:
    """使用 epsilon-greedy 决策和目标网络更新 Q 函数。"""

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        seed: int,
    ) -> None:
        dqn_config = config["dqn"]
        hidden_sizes = [int(value) for value in dqn_config["hidden_sizes"]]
        self.device = device
        self.online = QNetwork(hidden_sizes).to(device)
        self.target = QNetwork(hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=float(dqn_config["learning_rate"])
        )
        self.gamma = float(dqn_config["gamma"])
        self.batch_size = int(dqn_config["batch_size"])
        self.gradient_clip = float(dqn_config["gradient_clip"])
        self.state_scale = torch.tensor(
            dqn_config["state_scale"],
            dtype=torch.float32,
            device=device,
        )
        if self.state_scale.shape != (4,) or torch.any(self.state_scale <= 0):
            raise ValueError("dqn.state_scale 必须包含 4 个正数")
        self.rng = np.random.default_rng(seed)

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        """按各状态量的典型范围缩放，避免角度等小量被网络忽略。"""

        return states / self.state_scale

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """按 epsilon-greedy 决策函数选择动作。"""

        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(0, 2))
        tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return int(self.online(self.normalize(tensor)).argmax(dim=1).item())

    def optimize(self, replay: ReplayBuffer) -> float | None:
        """从经验回放采样一次并最小化 TD 误差。"""

        if replay.size < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = replay.sample(
            self.batch_size, self.device
        )
        normalized_states = self.normalize(states)
        normalized_next_states = self.normalize(next_states)
        predicted = (
            self.online(normalized_states).gather(1, actions[:, None]).squeeze(1)
        )
        with torch.no_grad():
            # Double DQN：在线网络选动作，目标网络估值，减轻 Q 值高估。
            next_actions = self.online(normalized_next_states).argmax(
                dim=1, keepdim=True
            )
            next_values = (
                self.target(normalized_next_states).gather(1, next_actions).squeeze(1)
            )
            target = rewards + self.gamma * (1.0 - dones) * next_values
        loss = nn.functional.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()
        return float(loss.detach())

    def update_target(self) -> None:
        """把在线 Q 网络参数复制到目标网络。"""

        self.target.load_state_dict(self.online.state_dict())


def epsilon_by_step(config: dict[str, Any], global_step: int) -> float:
    """按交互步数线性降低 DQN 探索率。"""

    dqn_config = config["dqn"]
    start = float(dqn_config["epsilon_start"])
    end = float(dqn_config["epsilon_end"])
    progress = min(global_step / float(dqn_config["epsilon_decay_steps"]), 1.0)
    return start + progress * (end - start)


def save_dqn(
    path: Path,
    agent: DQNAgent,
    metadata: dict[str, Any],
) -> None:
    """保存推理所需网络和训练元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "online_state": agent.online.state_dict(),
            "target_state": agent.target.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_dqn(
    path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[DQNAgent, dict[str, Any]]:
    """加载并校验 DQN checkpoint。"""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(checkpoint.get("metadata", {}))
    expected = str(config["experiment"]["version"])
    if metadata.get("experiment_version") != expected:
        raise RuntimeError(
            f"checkpoint 版本不匹配：需要 {expected}，"
            f"实际为 {metadata.get('experiment_version')!r}"
        )
    agent = DQNAgent(config, device, int(metadata.get("seed", 0)))
    agent.online.load_state_dict(checkpoint["online_state"])
    agent.target.load_state_dict(checkpoint["target_state"])
    if "optimizer_state" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
    agent.online.eval()
    return agent, metadata
