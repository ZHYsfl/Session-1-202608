"""校准程序化 Stage 2 场景的可解性、多样性与基线难度。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pygame
import torch
from common import DEFAULT_CONFIG, load_config, resolve_project_path, select_device
from evaluate import evaluate_controller
from geometry import (
    circle_intersects_rect,
    circle_outside_bounds,
    expand_rect,
    segment_intersects_rect,
)
from planner import astar_path, astar_path_length
from rl_env import BatchedLocalAvoidanceEnv

BACKGROUND = (18, 22, 30)
OBSTACLE = (210, 92, 74)
START = (75, 180, 245)
GOAL = (75, 210, 130)
ORACLE = (245, 196, 88)
TEXT = (235, 238, 243)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--preview-count", type=int, default=36)
    return parser


def _signature(snapshot: dict[str, object]) -> tuple[object, ...]:
    """用 2 px 量化后的几何构造近重复签名。"""

    def quantize(value: float) -> int:
        return round(value / 2.0)

    obstacles = tuple(
        tuple(quantize(float(value)) for value in rect)
        for rect in snapshot["obstacles"]  # type: ignore[union-attr]
    )
    return (
        bool(snapshot["horizontal"]),
        int(snapshot["gate_count"]),
        int(snapshot["bend_count"]),
        obstacles,
    )


def analyze_terrain(
    config: dict[str, Any],
    episodes: int,
    seed: int,
) -> dict[str, object]:
    """逐张检查固定场景，不在训练重置路径中运行 A*。"""

    env = BatchedLocalAvoidanceEnv(
        config,
        episodes,
        torch.device("cpu"),
        seed,
        auto_reset=False,
    )
    env.set_curriculum_stage(2)
    env.reset()
    ratios: list[float] = []
    signatures: set[tuple[object, ...]] = set()
    solvable = safe_initial = direct_blocked = 0
    gate_counts: dict[int, int] = {}
    bend_counts: dict[int, int] = {}
    gap_widths: list[float] = []

    for index in range(episodes):
        snapshot = env.snapshot(index)
        obstacles = snapshot["obstacles"]  # type: ignore[assignment]
        start = snapshot["position"]  # type: ignore[assignment]
        goal = snapshot["goal"]  # type: ignore[assignment]
        signatures.add(_signature(snapshot))
        gates = int(snapshot["gate_count"])
        bends = int(snapshot["bend_count"])
        gate_counts[gates] = gate_counts.get(gates, 0) + 1
        bend_counts[bends] = bend_counts.get(bends, 0) + 1
        gap_widths.extend(float(value) for value in snapshot["gate_widths"])  # type: ignore[union-attr]

        start_safe = not circle_outside_bounds(
            start[0], start[1], env.radius, env.width, env.height
        ) and not any(
            circle_intersects_rect(start[0], start[1], env.radius, rect)
            for rect in obstacles
        )
        goal_safe = not circle_outside_bounds(
            goal[0], goal[1], env.radius, env.width, env.height
        ) and not any(
            circle_intersects_rect(goal[0], goal[1], env.radius, rect)
            for rect in obstacles
        )
        safe_initial += int(start_safe and goal_safe)
        direct_blocked += int(
            any(
                segment_intersects_rect(
                    start,
                    goal,
                    expand_rect(rect, env.radius + 2.0),
                )
                for rect in obstacles
            )
        )
        reference = astar_path_length(
            env.width,
            env.height,
            obstacles,
            start,
            goal,
            env.radius + 2.0,
        )
        if reference is not None:
            solvable += 1
            ratios.append(reference / math.dist(start, goal))

    duplicate_rate = 1.0 - len(signatures) / episodes
    return {
        "episodes": episodes,
        "oracle_solvable_rate": solvable / episodes,
        "safe_initial_rate": safe_initial / episodes,
        "direct_path_blocked_rate": direct_blocked / episodes,
        "duplicate_rate": duplicate_rate,
        "minimum_path_ratio": min(ratios) if ratios else None,
        "maximum_path_ratio": max(ratios) if ratios else None,
        "mean_path_ratio": sum(ratios) / len(ratios) if ratios else None,
        "minimum_gate_width": min(gap_widths),
        "maximum_gate_width": max(gap_widths),
        "gate_count_distribution": gate_counts,
        "bend_count_distribution": bend_counts,
    }


def render_preview(
    config: dict[str, Any],
    seed: int,
    count: int,
    output_path: Path,
) -> None:
    """保存包含 A* 参考路线的固定地图预览拼图。"""

    if count <= 0:
        return
    columns = min(6, count)
    rows = math.ceil(count / columns)
    tile_width = int(config["window"]["width"]) // 2
    tile_height = int(config["window"]["height"]) // 2
    scale = 0.5
    pygame.init()
    surface = pygame.Surface((columns * tile_width, rows * tile_height))
    surface.fill(BACKGROUND)
    font = pygame.font.Font(None, 18)
    env = BatchedLocalAvoidanceEnv(
        config,
        count,
        torch.device("cpu"),
        seed,
        auto_reset=False,
    )
    env.set_curriculum_stage(2)
    env.reset()

    for index in range(count):
        snapshot = env.snapshot(index)
        tile_x = (index % columns) * tile_width
        tile_y = (index // columns) * tile_height
        pygame.draw.rect(
            surface,
            (29, 35, 45),
            pygame.Rect(tile_x, tile_y, tile_width - 1, tile_height - 1),
        )
        for rect in snapshot["obstacles"]:  # type: ignore[union-attr]
            pygame.draw.rect(
                surface,
                OBSTACLE,
                pygame.Rect(
                    tile_x + round(rect[0] * scale),
                    tile_y + round(rect[1] * scale),
                    max(1, round(rect[2] * scale)),
                    max(1, round(rect[3] * scale)),
                ),
            )
        start = snapshot["position"]  # type: ignore[assignment]
        goal = snapshot["goal"]  # type: ignore[assignment]
        path = astar_path(
            env.width,
            env.height,
            snapshot["obstacles"],  # type: ignore[arg-type]
            start,
            goal,
            env.radius + 2.0,
        )
        if path and len(path) >= 2:
            pygame.draw.lines(
                surface,
                ORACLE,
                False,
                [(tile_x + x * scale, tile_y + y * scale) for x, y in path],
                2,
            )
        pygame.draw.circle(
            surface,
            START,
            (tile_x + round(start[0] * scale), tile_y + round(start[1] * scale)),
            max(3, round(env.radius * scale)),
        )
        pygame.draw.circle(
            surface,
            GOAL,
            (tile_x + round(goal[0] * scale), tile_y + round(goal[1] * scale)),
            max(4, round(env.goal_radius * scale)),
            2,
        )
        label = font.render(
            f"#{index:02d} G{snapshot['gate_count']} B{snapshot['bend_count']}",
            True,
            TEXT,
        )
        surface.blit(label, (tile_x + 5, tile_y + 5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, output_path)
    pygame.quit()


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    calibration = config["terrain_calibration"]
    episodes = args.episodes or int(calibration["episodes"])
    seed = int(calibration["seed"])
    terrain = analyze_terrain(config, episodes, seed)
    baselines = [
        evaluate_controller(
            controller,
            config,
            device,
            episodes,
            args.num_envs,
            noisy=False,
            stage=2,
            seed=seed,
            compute_oracle=False,
        )
        for controller in ("direct", "reactive")
    ]
    direct, reactive = baselines
    checks = {
        "oracle_solvable_rate_is_100_percent": (terrain["oracle_solvable_rate"] == 1.0),
        "initial_states_are_safe": terrain["safe_initial_rate"] == 1.0,
        "direct_paths_are_blocked": terrain["direct_path_blocked_rate"] == 1.0,
        "path_ratio_is_in_range": (
            terrain["minimum_path_ratio"] is not None
            and terrain["maximum_path_ratio"] is not None
            and float(terrain["minimum_path_ratio"])
            >= float(calibration["min_path_ratio"])
            and float(terrain["maximum_path_ratio"])
            <= float(calibration["max_path_ratio"])
        ),
        "duplicate_rate_is_low": (
            float(terrain["duplicate_rate"]) <= float(calibration["max_duplicate_rate"])
        ),
        "stage_two_covers_gate_counts": (
            set(terrain["gate_count_distribution"]) == {4, 5, 6}  # type: ignore[arg-type]
        ),
        "stage_two_covers_bend_counts": (
            set(terrain["bend_count_distribution"]) == {2, 3, 4}  # type: ignore[arg-type]
        ),
        "minimum_gate_width_is_safe": (float(terrain["minimum_gate_width"]) >= 56.0),
        "direct_success_rate_is_low": (
            float(direct["success_rate"])
            < float(calibration["direct_max_success_rate"])
        ),
        "reactive_success_rate_is_in_range": (
            float(calibration["reactive_min_success_rate"])
            <= float(reactive["success_rate"])
            <= float(calibration["reactive_max_success_rate"])
        ),
    }
    report = {
        "experiment_version": config["experiment"]["version"],
        "terrain": terrain,
        "baselines": baselines,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path = resolve_project_path(str(config["paths"]["terrain_calibration"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    preview_path = resolve_project_path(str(config["paths"]["terrain_preview"]))
    render_preview(config, seed, args.preview_count, preview_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"校准报告：{output_path}")
    print(f"地图预览：{preview_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
