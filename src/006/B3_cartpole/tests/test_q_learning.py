from __future__ import annotations

import numpy as np
from common import load_config
from q_learning import QLearningAgent, StateDiscretizer


def test_discretizer_clips_extreme_states_to_table_bounds() -> None:
    discretizer = StateDiscretizer(load_config())
    low = discretizer.encode(np.full(4, -1e9, dtype=np.float32))
    high = discretizer.encode(np.full(4, 1e9, dtype=np.float32))
    assert low == (0, 0, 0, 0)
    assert high == tuple(value - 1 for value in discretizer.bins)


def test_q_learning_update_changes_selected_value() -> None:
    agent = QLearningAgent(load_config(), seed=9)
    state = np.zeros(4, dtype=np.float32)
    index = agent.discretizer.encode(state)
    before = float(agent.q_table[index][1])
    td_error = agent.update(state, 1, 1.0, state, done=True)
    after = float(agent.q_table[index][1])
    assert td_error == 1.0
    assert after > before


def test_q_learning_action_is_valid() -> None:
    agent = QLearningAgent(load_config(), seed=9)
    assert agent.act(np.zeros(4, dtype=np.float32), epsilon=0.0) in (0, 1)
