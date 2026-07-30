"""在固定留出任务上评估 PPO 与非学习基线。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from common import DEFAULT_CONFIG, load_config, resolve_project_path, select_device
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
    parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=2)
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
            f"找不到 {name.upper()} checkpoint：{path}，请先完成正式训练"
        )
    model, _ = load_policy(
        path,
        config["model"],
        device,
        expected_version=str(config["experiment"]["version"]),
    )
    return lambda obs: policy_action(model, obs), model


def _reference_length(env: BatchedLocalAvoidanceEnv, index: int) -> float:
    snapshot = env.snapshot(index)
    result = astar_path_length(
        env.width,
        env.height,
        snapshot["obstacles"],  # type: ignore[arg-type]
        snapshot["position"],  # type: ignore[arg-type]
        snapshot["goal"],  # type: ignore[arg-type]
        env.radius + 2.0,
    )
    return float("nan") if result is None else result


@torch.no_grad()
def evaluate_controller(
    name: str,
    config: dict[str, Any],
    device: torch.device,
    episodes: int,
    num_envs: int,
    noisy: bool,
    *,
    stage: int = 2,
    seed: int | None = None,
    compute_oracle: bool = True,
) -> dict[str, object]:
    """在相同固定任务上评估一个控制器并汇总互斥回合结果。"""

    controller, model = _load_controller(name, config, device)
    _ = model
    evaluation_seed = int(config["evaluation"]["seed"]) if seed is None else seed
    success_count = collision_count = timeout_count = 0
    oracle_solvable_count = completed = 0
    path_ratios: list[float] = []
    path_lengths: list[float] = []
    gate_stats: defaultdict[int, dict[str, int]] = defaultdict(
        lambda: {
            "episodes": 0,
            "success": 0,
            "collision": 0,
            "timeout": 0,
            "oracle_solvable": 0,
        }
    )

    while completed < episodes:
        batch_size = min(num_envs, episodes - completed)
        env = BatchedLocalAvoidanceEnv(
            config,
            batch_size,
            device,
            evaluation_seed + completed,
            training_noise=noisy,
            auto_reset=False,
        )
        env.set_curriculum_stage(stage)
        observation = env.reset()
        references = torch.tensor(
            (
                [_reference_length(env, index) for index in range(batch_size)]
                if compute_oracle
                else [float("nan")] * batch_size
            ),
            device=device,
        )
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        while bool(active.any()):
            action = controller(observation)
            action[~active, 0] = -1.0
            action[~active, 1] = 0.0
            observation, _, done, info = env.step(action)
            finished = done & active
            if not bool(finished.any()):
                continue
            indices = torch.nonzero(finished, as_tuple=False).flatten()
            success_count += int(info["success"][indices].sum())
            collision_count += int(info["collision"][indices].sum())
            timeout_count += int(info["timeout"][indices].sum())
            for index in indices.detach().cpu().tolist():
                succeeded = bool(info["success"][index])
                collided = bool(info["collision"][index])
                timed_out = bool(info["timeout"][index])
                gates = int(info["gate_count"][index].detach().cpu())
                reference = float(references[index].detach().cpu())
                path = float(info["episode_path_length"][index].detach().cpu())
                path_lengths.append(path)
                stats = gate_stats[gates]
                stats["episodes"] += 1
                stats["success"] += int(succeeded)
                stats["collision"] += int(collided)
                stats["timeout"] += int(timed_out)
                if not math.isnan(reference) and reference > 0.0:
                    oracle_solvable_count += 1
                    stats["oracle_solvable"] += 1
                    if succeeded:
                        path_ratios.append(path / reference)
            active[indices] = False
        completed += batch_size

    return {
        "controller": name,
        "stage": stage,
        "episodes": completed,
        "noise": "on" if noisy else "off",
        "success_rate": success_count / completed,
        "collision_rate": collision_count / completed,
        "timeout_rate": timeout_count / completed,
        "oracle_solvable_rate": (
            oracle_solvable_count / completed if compute_oracle else None
        ),
        "mean_path_length": _safe_mean(path_lengths),
        "mean_path_ratio": _safe_mean(path_ratios) if path_ratios else None,
        "gate_count_breakdown": dict(sorted(gate_stats.items())),
    }


def _safe_mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


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
    results: list[dict[str, object]] = []
    for name in names:
        try:
            metrics = evaluate_controller(
                name,
                config,
                device,
                episodes,
                args.num_envs,
                args.noise,
                stage=args.stage,
            )
        except FileNotFoundError as error:
            if args.controller != "all":
                raise
            print(f"跳过 {name.upper()}：{error}")
            continue
        results.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_path = resolve_project_path(str(config["paths"]["evaluation_metrics"]))
    suffix = "_noise" if args.noise else ""
    output_path = output_path.with_name(
        f"{output_path.stem}_stage{args.stage}{suffix}{output_path.suffix}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"评估结果：{output_path}")


if __name__ == "__main__":
    main()
