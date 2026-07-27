"""批量局部避障环境测试。"""

from __future__ import annotations

import torch
from common import load_config
from planner import astar_path_length
from rl_env import BatchedLocalAvoidanceEnv


def test_observation_and_step_shapes() -> None:
    config = load_config()
    env = BatchedLocalAvoidanceEnv(
        config, num_envs=8, device=torch.device("cpu"), seed=7
    )
    env.set_curriculum_stage(2)
    observation = env.reset()
    assert observation.shape == (8, 43)
    assert torch.all((0.0 <= observation[:, :36]) & (observation[:, :36] <= 1.0))

    next_observation, reward, done, info = env.step(torch.zeros((8, 2)))
    assert next_observation.shape == observation.shape
    assert reward.shape == (8,)
    assert done.shape == (8,)
    assert set(info) == {
        "success",
        "collision",
        "timeout",
        "movement",
        "episode_path_length",
    }


def test_same_seed_reproduces_initial_layout() -> None:
    config = load_config()
    first = BatchedLocalAvoidanceEnv(config, 2, torch.device("cpu"), seed=99)
    second = BatchedLocalAvoidanceEnv(config, 2, torch.device("cpu"), seed=99)
    assert torch.allclose(first.position, second.position)
    assert torch.allclose(first.goal, second.goal)
    assert torch.allclose(first.obstacles, second.obstacles)


def test_stage_two_layouts_are_solvable_by_offline_oracle() -> None:
    config = load_config()
    env = BatchedLocalAvoidanceEnv(config, 16, torch.device("cpu"), seed=123)
    env.set_curriculum_stage(2)
    env.reset()
    for index in range(env.num_envs):
        snapshot = env.snapshot(index)
        length = astar_path_length(
            env.width,
            env.height,
            snapshot["obstacles"],
            snapshot["position"],
            snapshot["goal"],
            env.radius + 2.0,
        )
        assert length is not None


def test_boundary_contact_terminates_as_collision() -> None:
    config = load_config()
    env = BatchedLocalAvoidanceEnv(
        config,
        1,
        torch.device("cpu"),
        seed=3,
        auto_reset=False,
    )
    env.position[0] = torch.tensor([env.radius, env.height / 2.0])
    _, _, done, info = env.step(torch.tensor([[-1.0, 0.0]]))
    assert bool(done[0])
    assert bool(info["collision"][0])
