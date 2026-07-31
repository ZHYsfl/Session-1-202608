"""B3 正式实验矩阵的训练、隔离评估与报告入口。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from common import (
    DEFAULT_CONFIG,
    PROJECT_DIR,
    load_config,
    seed_everything,
    select_device,
)
from dqn import load_dqn
from evaluate import evaluate_policy
from formal_report import generate_report
from formal_training import (
    train_dqn_fixed_steps,
    train_q_learning_fixed_steps,
    write_json,
)
from q_learning import load_q_table

DEFAULT_MATRIX = PROJECT_DIR / "formal_experiments.yaml"
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "formal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 B3 正式实验流水线")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="列出冻结实验矩阵")
    list_parser.set_defaults(action=list_experiments)

    train_parser = subparsers.add_parser("train", help="训练实验矩阵")
    train_parser.add_argument("--variants", nargs="*", default=None)
    train_parser.add_argument("--seeds", nargs="*", type=int, default=None)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="仅用于烟雾测试，覆盖正式70,000步预算",
    )
    train_parser.add_argument(
        "--validation-interval",
        type=int,
        default=None,
        help="仅用于烟雾测试",
    )
    train_parser.add_argument(
        "--validation-episodes",
        type=int,
        default=None,
        help="仅用于烟雾测试",
    )
    train_parser.set_defaults(action=train_matrix)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="评估 best/final checkpoint"
    )
    evaluate_parser.add_argument(
        "--split",
        choices=("development", "final"),
        required=True,
    )
    evaluate_parser.add_argument("--variants", nargs="*", default=None)
    evaluate_parser.add_argument("--seeds", nargs="*", type=int, default=None)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="正式测试集的显式解封开关",
    )
    evaluate_parser.set_defaults(action=evaluate_matrix)

    report_parser = subparsers.add_parser("report", help="生成 CSV 和结果图")
    report_parser.add_argument(
        "--split",
        choices=("development", "final"),
        required=True,
    )
    report_parser.set_defaults(action=report_results)
    return parser.parse_args()


def read_matrix(path: Path) -> dict[str, Any]:
    """读取并校验正式实验矩阵。"""

    with path.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    if not isinstance(matrix, dict) or "protocol" not in matrix:
        raise ValueError("formal_experiments.yaml 缺少 protocol")
    return matrix


def formal_hash(matrix_path: Path) -> str:
    """同时哈希基础配置和正式矩阵，防止混用不同协议的结果。"""

    digest = hashlib.sha256()
    digest.update(DEFAULT_CONFIG.read_bytes())
    digest.update(matrix_path.read_bytes())
    return digest.hexdigest()


def variant_configs(
    matrix: dict[str, Any],
    config_sha256: str,
) -> dict[str, tuple[str, str, dict[str, Any]]]:
    """展开 DQN 与 Q-Learning 变体，并生成各自配置副本。"""

    variants: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for name, settings in matrix["dqn_variants"].items():
        config = copy.deepcopy(load_config())
        config["dqn"].update(
            {
                "normalize_state": bool(settings["normalize_state"]),
                "double_dqn": bool(settings["double_dqn"]),
                "epsilon_decay_steps": int(settings["epsilon_decay_steps"]),
            }
        )
        config["formal"] = {
            "config_sha256": config_sha256,
            "variant": name,
        }
        variants[name] = ("dqn", str(settings["label"]), config)

    for name, settings in matrix["q_learning_variants"].items():
        config = copy.deepcopy(load_config())
        config["q_learning"]["bins"] = [int(value) for value in settings["bins"]]
        config["formal"] = {
            "config_sha256": config_sha256,
            "variant": name,
        }
        variants[name] = ("q_learning", str(settings["label"]), config)
    return variants


def select_names(
    available: dict[str, Any],
    requested: list[str] | None,
) -> list[str]:
    """保持 YAML 顺序选择实验，并拒绝拼写错误。"""

    if not requested:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"未知实验变体：{', '.join(unknown)}")
    return [name for name in available if name in requested]


def selected_seeds(
    matrix: dict[str, Any],
    requested: list[int] | None,
) -> list[int]:
    """选择训练种子；命令行覆盖仅用于调试子集。"""

    seeds = (
        [int(value) for value in matrix["protocol"]["training_seeds"]]
        if not requested
        else requested
    )
    if not seeds:
        raise ValueError("至少需要一个训练种子")
    return seeds


def write_manifest(
    matrix_path: Path,
    matrix: dict[str, Any],
    config_sha256: str,
) -> None:
    """记录正式协议、软件和硬件环境。"""

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "formal_config_sha256": config_sha256,
        "matrix_path": str(matrix_path.resolve()),
        "protocol": matrix["protocol"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)


def run_directory(variant: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / variant / f"seed_{seed}"


def completed_training(
    directory: Path,
    config_sha256: str,
    environment_steps: int,
) -> bool:
    """只有哈希和步数均匹配时才跳过已完成任务。"""

    path = directory / "training_metrics.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    return bool(
        metadata.get("completed")
        and metadata.get("formal_config_sha256") == config_sha256
        and int(metadata.get("environment_steps", -1)) == environment_steps
    )


def checkpoint_paths(directory: Path, algorithm: str) -> tuple[Path, Path]:
    """返回某次训练必须同时存在的 best/final checkpoint。"""

    suffix = ".pt" if algorithm == "dqn" else ".npz"
    return directory / f"best{suffix}", directory / f"final{suffix}"


def evaluation_is_current(path: Path, config_sha256: str, split: str) -> bool:
    """仅复用协议哈希和数据划分均匹配的已有评估结果。"""

    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    return bool(
        metadata.get("formal_config_sha256") == config_sha256
        and metadata.get("split") == split
    )


def list_experiments(args: argparse.Namespace) -> None:
    matrix = read_matrix(args.matrix)
    print("DQN variants:")
    for name, values in matrix["dqn_variants"].items():
        print(f"  {name}: {values['label']}")
    print("Q-Learning variants:")
    for name, values in matrix["q_learning_variants"].items():
        print(f"  {name}: {values['label']} bins={values['bins']}")
    print(f"Training seeds: {matrix['protocol']['training_seeds']}")
    print(f"Protocol: {matrix['protocol']['environment_steps']} steps/run")


def train_matrix(args: argparse.Namespace) -> None:
    """依次训练矩阵；重启命令会自动跳过完整运行。"""

    matrix = read_matrix(args.matrix)
    config_sha256 = formal_hash(args.matrix)
    variants = variant_configs(matrix, config_sha256)
    names = select_names(variants, args.variants)
    seeds = selected_seeds(matrix, args.seeds)
    protocol = matrix["protocol"]
    environment_steps = int(args.max_steps or protocol["environment_steps"])
    validation_interval = int(
        args.validation_interval or protocol["validation_interval_steps"]
    )
    validation_episodes = int(
        args.validation_episodes or protocol["validation"]["episodes"]
    )
    validation_seed = int(protocol["validation"]["seed"])
    if environment_steps <= 0 or validation_interval <= 0:
        raise ValueError("训练步数和验证间隔必须为正数")
    write_manifest(args.matrix, matrix, config_sha256)
    device = select_device(args.device)

    for name in names:
        algorithm, label, config = variants[name]
        for seed in seeds:
            directory = run_directory(name, seed)
            if completed_training(directory, config_sha256, environment_steps):
                print(f"SKIP completed: {name} seed={seed}")
                continue
            seed_everything(seed)
            print(
                f"START {name} seed={seed} algorithm={algorithm} "
                f"steps={environment_steps}"
            )
            if algorithm == "dqn":
                train_dqn_fixed_steps(
                    config,
                    variant=name,
                    label=label,
                    seed=seed,
                    environment_steps=environment_steps,
                    validation_interval=validation_interval,
                    validation_episodes=validation_episodes,
                    validation_seed=validation_seed,
                    output_directory=directory,
                    device=device,
                )
            else:
                train_q_learning_fixed_steps(
                    config,
                    variant=name,
                    label=label,
                    seed=seed,
                    environment_steps=environment_steps,
                    validation_interval=validation_interval,
                    validation_episodes=validation_episodes,
                    validation_seed=validation_seed,
                    output_directory=directory,
                )
            print(f"DONE {name} seed={seed}")


def evaluation_split(
    args: argparse.Namespace,
    matrix: dict[str, Any],
) -> tuple[int, int]:
    """返回开发或正式测试的回合数与首个种子，并保护正式测试。"""

    if args.split == "final" and not args.confirm_final_test:
        raise PermissionError(
            "正式测试集仍封存；确认配置冻结后添加 --confirm-final-test"
        )
    settings = matrix["protocol"][args.split]
    return int(settings["episodes"]), int(settings["seed"])


def evaluate_matrix(args: argparse.Namespace) -> None:
    """分别评估学习方法的 best/final，并加入 Random baseline。"""

    matrix = read_matrix(args.matrix)
    config_sha256 = formal_hash(args.matrix)
    variants = variant_configs(matrix, config_sha256)
    if args.split == "final" and (args.variants or args.seeds):
        raise ValueError("正式测试必须一次评估完整矩阵，不允许筛选变体或种子")
    names = select_names(variants, args.variants)
    seeds = selected_seeds(matrix, args.seeds)
    episodes, evaluation_seed = evaluation_split(args, matrix)
    device = select_device(args.device)
    missing: list[str] = []

    # 正式测试解封前一次性检查全部训练任务，避免只测部分方法后泄漏结果。
    if args.split == "final":
        required_steps = int(matrix["protocol"]["environment_steps"])
        for name in names:
            algorithm = variants[name][0]
            for seed in seeds:
                directory = run_directory(name, seed)
                if not completed_training(
                    directory,
                    config_sha256,
                    required_steps,
                ):
                    missing.append(f"{name}/seed_{seed}")
                    continue
                if not all(
                    path.exists() for path in checkpoint_paths(directory, algorithm)
                ):
                    missing.append(f"{name}/seed_{seed} checkpoints")
        if missing:
            raise RuntimeError(
                "正式测试仍封存；以下任务未完成或 checkpoint 不完整："
                f"{', '.join(missing)}"
            )
        missing.clear()

    for name in names:
        algorithm, label, config = variants[name]
        for seed in seeds:
            directory = run_directory(name, seed)
            metrics_path = directory / "training_metrics.json"
            if not metrics_path.exists():
                missing.append(f"{name}/seed_{seed}")
                continue
            training = json.loads(metrics_path.read_text(encoding="utf-8"))
            training_metadata = training["metadata"]
            if training_metadata["formal_config_sha256"] != config_sha256:
                raise RuntimeError(f"{name}/seed_{seed} 的配置哈希不匹配")
            result_path = directory / f"evaluation_{args.split}.json"
            if evaluation_is_current(result_path, config_sha256, args.split):
                print(f"SKIP evaluated: {name} seed={seed} split={args.split}")
                continue

            checkpoint_results: dict[str, dict[str, float | int]] = {}
            checkpoints = zip(
                ("best", "final"),
                checkpoint_paths(directory, algorithm),
                strict=True,
            )
            for kind, checkpoint_path in checkpoints:
                if algorithm == "dqn":
                    agent, checkpoint_metadata_value = load_dqn(
                        checkpoint_path, config, device
                    )
                    if (
                        checkpoint_metadata_value["formal_config_sha256"]
                        != config_sha256
                    ):
                        raise RuntimeError(f"{checkpoint_path} 配置哈希不匹配")
                    policy = lambda state, current=agent: current.act(
                        state, epsilon=0.0
                    )
                else:
                    agent, checkpoint_metadata_value = load_q_table(
                        checkpoint_path, config
                    )
                    if (
                        checkpoint_metadata_value["formal_config_sha256"]
                        != config_sha256
                    ):
                        raise RuntimeError(f"{checkpoint_path} 配置哈希不匹配")
                    policy = lambda state, current=agent: current.act(
                        state,
                        epsilon=0.0,
                        deterministic=True,
                    )
                checkpoint_results[kind] = evaluate_policy(
                    policy,
                    config,
                    episodes,
                    evaluation_seed,
                )

            write_json(
                result_path,
                {
                    "metadata": {
                        "formal_config_sha256": config_sha256,
                        "split": args.split,
                        "algorithm": algorithm,
                        "variant": name,
                        "label": label,
                        "seed": seed,
                        "environment_steps": training_metadata["environment_steps"],
                        "wall_clock_seconds": training_metadata["wall_clock_seconds"],
                    },
                    "checkpoints": checkpoint_results,
                },
            )
            print(f"EVALUATED {name} seed={seed} split={args.split}")

    random_label = str(matrix["baselines"]["random"]["label"])
    for seed in seeds:
        directory = run_directory("random", seed)
        result_path = directory / f"evaluation_{args.split}.json"
        if evaluation_is_current(result_path, config_sha256, args.split):
            continue
        rng = np.random.default_rng(seed)
        random_result = evaluate_policy(
            lambda state, generator=rng: int(generator.integers(0, 2)),
            load_config(),
            episodes,
            evaluation_seed,
        )
        write_json(
            result_path,
            {
                "metadata": {
                    "formal_config_sha256": config_sha256,
                    "split": args.split,
                    "algorithm": "random",
                    "variant": "random",
                    "label": random_label,
                    "seed": seed,
                    "environment_steps": 0,
                    "wall_clock_seconds": 0.0,
                },
                "checkpoints": {"policy": random_result},
            },
        )

    if missing:
        message = f"缺少训练结果：{', '.join(missing)}"
        print(f"WARNING {message}")


def report_results(args: argparse.Namespace) -> None:
    paths = generate_report(OUTPUT_ROOT, args.split)
    for name, path in paths.items():
        print(f"{name}: {path}")


def main() -> None:
    args = parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
