"""在固定种子局部任务上评估 PPO 与非学习基线。"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from common import (
    DEFAULT_CONFIG,
    load_config,
    resolve_project_path,
    select_device,
)
from controller import direct_action, policy_action, reactive_action
from planner import astar_path_length
from policy import ActorCritic, load_policy
from rl_env import BatchedLocalAvoidanceEnv

Controller = Callable[[torch.Tensor], torch.Tensor]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--controller",
        choices=("direct", "reactive", "mlp", "cnn", "all"),
        default="all",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--noise", action="store_true", help="运行独立噪声鲁棒性测试")
    return parser


def _load_controller(
    name: str,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[Controller, ActorCritic | None]:
    sensor_count = int(config["sensors"]["count"])
    if name == "direct":
        return lambda obs: direct_action(obs, sensor_count), None
    if name == "reactive":
        return lambda obs: reactive_action(obs, sensor_count), None
    path = resolve_project_path(str(config["paths"][f"{name}_checkpoint"]))
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {name.upper()} checkpoint：{path}，请先训练该模型"
        )
    model, _ = load_policy(path, config["model"], device)
    return lambda obs: policy_action(model, obs), model


def _reference_lengths(
    env: BatchedLocalAvoidanceEnv,
) -> torch.Tensor:
    values: list[float] = []
    for index in range(env.num_envs):
        snapshot = env.snapshot(index)
        result = astar_path_length(
            env.width,
            env.height,
            snapshot["obstacles"],  # type: ignore[arg-type]
            snapshot["position"],  # type: ignore[arg-type]
            snapshot["goal"],  # type: ignore[arg-type]
            env.radius + 2.0,
        )
        values.append(float("nan") if result is None else result)
    return torch.tensor(values, device=env.device)


@torch.no_grad()
def evaluate_controller(
    name: str,
    config: dict[str, Any],
    device: torch.device,
    episodes: int,
    num_envs: int,
    noisy: bool,
) -> dict[str, float | int | str | None]:
    """评估一个控制器并汇总互斥回合结果。"""

    controller, model = _load_controller(name, config, device)
    _ = model  # 保持模型生命周期覆盖整个评估函数。
    env = BatchedLocalAvoidanceEnv(
        config,
        min(num_envs, episodes),
        device,
        int(config["evaluation"]["seed"]),
        training_noise=noisy,
    )
    env.set_curriculum_stage(2)
    observation = env.reset()
    references = _reference_lengths(env)
    success_count = collision_count = timeout_count = 0
    completed = 0
    path_ratios: list[float] = []
    path_lengths: list[float] = []

    while completed < episodes:
        action = controller(observation)
        observation, _, done, info = env.step(action)
        finished = torch.nonzero(done, as_tuple=False).flatten()
        remaining = episodes - completed
        finished = finished[:remaining]
        if finished.numel() == 0:
            continue
        success_count += int(info["success"][finished].sum())
        collision_count += int(info["collision"][finished].sum())
        timeout_count += int(info["timeout"][finished].sum())
        final_paths = info["episode_path_length"][finished]
        success_flags = info["success"][finished].detach().cpu().tolist()
        for path, reference, succeeded in zip(
            final_paths.detach().cpu().tolist(),
            references[finished].detach().cpu().tolist(),
            success_flags,
            strict=True,
        ):
            path_lengths.append(float(path))
            if succeeded and not math.isnan(reference) and reference > 0.0:
                path_ratios.append(float(path) / float(reference))
        completed += int(finished.numel())
        # step 已自动重置结束环境，更新它们的新离线参考长度。
        new_references = _reference_lengths(env)
        references[finished] = new_references[finished]

    return {
        "controller": name,
        "episodes": completed,
        "noise": "on" if noisy else "off",
        "success_rate": success_count / completed,
        "collision_rate": collision_count / completed,
        "timeout_rate": timeout_count / completed,
        "mean_path_length": sum(path_lengths) / len(path_lengths),
        "mean_path_ratio": (
            sum(path_ratios) / len(path_ratios) if path_ratios else None
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    episodes = args.episodes or int(config["evaluation"]["episodes"])
    names = (
        ["direct", "reactive", "mlp", "cnn"]
        if args.controller == "all"
        else [args.controller]
    )
    results: list[dict[str, float | int | str | None]] = []
    for name in names:
        try:
            metrics = evaluate_controller(
                name,
                config,
                device,
                episodes,
                args.num_envs,
                args.noise,
            )
        except FileNotFoundError as error:
            if args.controller != "all":
                raise
            print(f"跳过 {name.upper()}：{error}")
            continue
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_path = resolve_project_path(str(config["paths"]["evaluation_metrics"]))
    if args.noise:
        output_path = output_path.with_name(
            f"{output_path.stem}_noise{output_path.suffix}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"评估结果：{output_path}")


if __name__ == "__main__":
    main()
