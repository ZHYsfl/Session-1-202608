from __future__ import annotations

import numpy as np
import torch
from common import load_config
from dqn import DQNAgent, QNetwork, ReplayBuffer


def test_q_network_output_shape() -> None:
    network = QNetwork([32, 32])
    output = network(torch.zeros((7, 4)))
    assert output.shape == (7, 2)


def test_replay_and_single_optimization_step() -> None:
    config = load_config()
    config["dqn"]["batch_size"] = 4
    agent = DQNAgent(config, torch.device("cpu"), seed=7)
    replay = ReplayBuffer(capacity=8, seed=7)
    for index in range(4):
        state = np.full(4, index, dtype=np.float32)
        replay.add(state, index % 2, 1.0, state + 0.1, False)
    loss = agent.optimize(replay)
    assert loss is not None
    assert np.isfinite(loss)


def test_dqn_action_is_valid() -> None:
    agent = DQNAgent(load_config(), torch.device("cpu"), seed=3)
    assert agent.act(np.zeros(4, dtype=np.float32), epsilon=0.0) in (0, 1)


def test_dqn_normalizes_each_state_dimension() -> None:
    config = load_config()
    agent = DQNAgent(config, torch.device("cpu"), seed=3)
    scale = torch.tensor(config["dqn"]["state_scale"])
    normalized = agent.normalize(scale)
    torch.testing.assert_close(normalized, torch.ones(4))
