"""批量程序化局部避障环境测试。"""

from __future__ import annotations

import copy
import math

import torch
from common import load_config
from geometry import expand_rect, segment_intersects_rect
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
        "stage",
        "gate_count",
        "bend_count",
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
    assert torch.equal(first.gate_count, second.gate_count)


def test_stage_two_layouts_are_solvable_and_require_detours() -> None:
    config = load_config()
    env = BatchedLocalAvoidanceEnv(config, 32, torch.device("cpu"), seed=123)
    env.set_curriculum_stage(2)
    env.reset()
    assert set(env.gate_count.tolist()) == {4, 5, 6}
    assert set(env.bend_count.tolist()) == {2, 3, 4}
    assert not bool(env._collision().any())

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
        direct = math.dist(snapshot["position"], snapshot["goal"])
        assert 1.15 <= length / direct <= 1.80
        assert any(
            segment_intersects_rect(
                snapshot["position"],
                snapshot["goal"],
                expand_rect(rect, env.radius + 2.0),
            )
            for rect in snapshot["obstacles"]
        )


def test_training_mix_keeps_previous_stages() -> None:
    config = load_config()
    env = BatchedLocalAvoidanceEnv(
        config,
        256,
        torch.device("cpu"),
        seed=2026,
        mix_previous_stages=True,
    )
    env.set_curriculum_stage(2)
    env.reset()
    current_ratio = float((env.episode_stage == 2).float().mean())
    assert 0.60 <= current_ratio <= 0.80
    assert set(env.episode_stage.tolist()) == {0, 1, 2}


def test_timeout_has_explicit_penalty() -> None:
    config = copy.deepcopy(load_config())
    config["episode"]["max_steps"] = 1
    env = BatchedLocalAvoidanceEnv(
        config,
        1,
        torch.device("cpu"),
        seed=3,
        auto_reset=False,
    )
    _, reward, done, info = env.step(torch.tensor([[-1.0, 0.0]]))
    assert bool(done[0])
    assert bool(info["timeout"][0])
    assert float(reward[0]) < -4.5


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
