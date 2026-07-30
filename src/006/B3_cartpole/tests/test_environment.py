from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from common import load_config
from environment import CartPoleEnv


def test_reset_is_reproducible_and_has_expected_shape() -> None:
    config = load_config()
    environment = CartPoleEnv(config)
    first = environment.reset(seed=123)
    second = environment.reset(seed=123)
    assert first.shape == (4,)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_step_returns_finite_transition() -> None:
    environment = CartPoleEnv(load_config(), seed=1)
    state = environment.reset()
    next_state, reward, terminated, truncated, info = environment.step(1)
    assert next_state.shape == state.shape
    assert np.isfinite(next_state).all()
    assert np.isfinite(reward)
    assert not (terminated and truncated)
    assert info["step"] == 1


def test_invalid_action_is_rejected() -> None:
    environment = CartPoleEnv(load_config())
    environment.reset()
    with pytest.raises(ValueError, match="action"):
        environment.step(2)


def test_reaching_time_limit_counts_as_success() -> None:
    config = deepcopy(load_config())
    config["environment"]["max_steps"] = 1
    environment = CartPoleEnv(config)
    environment.reset(seed=5)
    _, reward, terminated, truncated, _ = environment.step(0)
    assert not terminated
    assert truncated
    assert reward == pytest.approx(
        config["environment"]["alive_reward"] + config["environment"]["success_bonus"]
    )
