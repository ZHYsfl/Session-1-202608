"""Offline BC benchmark on official RoboMimic v1.5 low-dimensional data.

The benchmark compares feed-forward BC, BC-RNN, and BC-Transformer on the
proficient-human Lift, Can, Square, and Transport datasets. It intentionally
reports offline action-prediction metrics. Simulator rollout success requires
MuJoCo and robosuite and is not inferred from validation loss.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

try:
    import h5py
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "Missing official benchmark dependencies. Run setup_official_benchmark.ps1 "
        "or install torch and h5py."
    ) from exc

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = ROOT / "data" / "official"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "official_checkpoints"
IMAGES_DIR = ROOT.parent / "images"

TASK_FILES = {
    "lift": "lift_ph_low_dim_v15.hdf5",
    "can": "can_ph_low_dim_v15.hdf5",
    "square": "square_ph_low_dim_v15.hdf5",
    "transport": "transport_ph_low_dim_v15.hdf5",
}

EXPECTED_SHA256 = {
    "lift": "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540",
    "can": "3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962",
    "square": "45d8cabb6d57a4c03e839aa5e4b3e58fb60fe8bd20e1951fec563a2abfd14951",
    "transport": "260515618f4c8b660e54171497ecceb08c7c6463691a9ce862b158bcacd950b4",
}

SINGLE_ARM_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]

TRANSPORT_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
    "object",
]

MODEL_NAMES = ["BC", "BC-RNN", "BC-Transformer"]
MODEL_COLORS = {
    "BC": "#D1495B",
    "BC-RNN": "#176B87",
    "BC-Transformer": "#2A9D8F",
}


@dataclass(frozen=True)
class Config:
    seed: int = 20260730
    context_length: int = 10
    batch_size: int = 512
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_train_windows: int = 60_000
    max_valid_windows: int = 20_000
    early_stopping_patience: int = 4
    bootstrap_repetitions: int = 2_000
    torch_threads: int = 12
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_keys(values: Iterable[bytes | str]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def observation_keys(task: str) -> list[str]:
    return TRANSPORT_KEYS if task == "transport" else SINGLE_ARM_KEYS


def read_task_metadata(task: str, path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        train_keys = decode_keys(handle["mask"]["train"][:])
        valid_keys = decode_keys(handle["mask"]["valid"][:])
        first_demo = handle["data"][train_keys[0]]
        obs_dim = sum(
            int(np.prod(first_demo["obs"][key].shape[1:]))
            for key in observation_keys(task)
        )
        lengths = [
            int(handle["data"][demo].attrs["num_samples"])
            for demo in handle["data"].keys()
        ]
        env_args = json.loads(handle["data"].attrs["env_args"])
        action_dim = int(first_demo["actions"].shape[1])
        return {
            "task": task,
            "environment": env_args["env_name"],
            "file": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "trajectories": len(handle["data"]),
            "train_trajectories": len(train_keys),
            "valid_trajectories": len(valid_keys),
            "total_samples": int(handle["data"].attrs["total"]),
            "mean_trajectory_length": float(np.mean(lengths)),
            "min_trajectory_length": int(np.min(lengths)),
            "max_trajectory_length": int(np.max(lengths)),
            "observation_keys": observation_keys(task),
            "observation_dim": obs_dim,
            "action_dim": action_dim,
            "train_keys": train_keys,
            "valid_keys": valid_keys,
        }


def read_observation(group: h5py.Group, keys: list[str]) -> np.ndarray:
    components = [
        np.asarray(group[key], dtype=np.float32).reshape(len(group[key]), -1)
        for key in keys
    ]
    return np.concatenate(components, axis=1)


def compute_normalization(
    path: Path, demo_keys: list[str], obs_keys: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_values: np.ndarray | None = None
    sum_squares: np.ndarray | None = None
    with h5py.File(path, "r") as handle:
        for demo_key in demo_keys:
            obs = read_observation(handle["data"][demo_key]["obs"], obs_keys)
            if sum_values is None:
                sum_values = np.zeros(obs.shape[1], dtype=np.float64)
                sum_squares = np.zeros(obs.shape[1], dtype=np.float64)
            sum_values += obs.sum(axis=0, dtype=np.float64)
            sum_squares += np.square(obs, dtype=np.float64).sum(axis=0)
            total += len(obs)
    assert sum_values is not None and sum_squares is not None
    mean = sum_values / total
    variance = np.maximum(sum_squares / total - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def select_window_indices(
    trajectory_lengths: list[int], max_windows: int, seed: int
) -> np.ndarray:
    total = int(sum(trajectory_lengths))
    if total <= max_windows:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_windows, replace=False))


def build_windows(
    path: Path,
    demo_keys: list[str],
    obs_keys: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    context_length: int,
    max_windows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        lengths = [
            int(handle["data"][demo_key].attrs["num_samples"])
            for demo_key in demo_keys
        ]
        selected = select_window_indices(lengths, max_windows, seed)
        first_demo = handle["data"][demo_keys[0]]
        obs_dim = int(mean.shape[0])
        action_dim = int(first_demo["actions"].shape[1])
        sequences = np.empty(
            (len(selected), context_length, obs_dim), dtype=np.float32
        )
        actions = np.empty((len(selected), action_dim), dtype=np.float32)
        demo_indices = np.empty(len(selected), dtype=np.int64)
        progress = np.empty(len(selected), dtype=np.float32)

        offset = 0
        output_cursor = 0
        for demo_index, (demo_key, length) in enumerate(zip(demo_keys, lengths)):
            left = int(np.searchsorted(selected, offset, side="left"))
            right = int(np.searchsorted(selected, offset + length, side="left"))
            local_steps = selected[left:right] - offset
            if len(local_steps) == 0:
                offset += length
                continue

            demo = handle["data"][demo_key]
            obs = read_observation(demo["obs"], obs_keys)
            obs = (obs - mean) / std
            demo_actions = np.asarray(demo["actions"], dtype=np.float32)
            for step in local_steps:
                start = max(0, int(step) - context_length + 1)
                window = obs[start : int(step) + 1]
                pad_count = context_length - len(window)
                if pad_count:
                    sequences[output_cursor, :pad_count] = window[0]
                    sequences[output_cursor, pad_count:] = window
                else:
                    sequences[output_cursor] = window
                actions[output_cursor] = demo_actions[int(step)]
                demo_indices[output_cursor] = demo_index
                progress[output_cursor] = float(step) / max(1, length - 1)
                output_cursor += 1
            offset += length

    assert output_cursor == len(selected)
    return sequences, actions, demo_indices, progress


class FeedForwardBC(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.policy(sequence[:, -1])


class RecurrentBC(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(sequence)
        output, _ = self.rnn(encoded)
        return self.head(output[:, -1])


class TransformerBC(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, context_length: int
    ):
        super().__init__()
        embed_dim = 128
        self.input_projection = nn.Linear(obs_dim, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, context_length, embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(sequence) + self.position_embedding
        encoded = self.encoder(encoded)
        return self.head(self.norm(encoded[:, -1]))


def create_model(
    name: str, obs_dim: int, action_dim: int, context_length: int
) -> nn.Module:
    if name == "BC":
        return FeedForwardBC(obs_dim, action_dim)
    if name == "BC-RNN":
        return RecurrentBC(obs_dim, action_dim)
    if name == "BC-Transformer":
        return TransformerBC(obs_dim, action_dim, context_length)
    raise ValueError(f"Unknown model: {name}")


def resolve_device(config: Config) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def component_indices(action_dim: int) -> dict[str, list[int]]:
    if action_dim == 7:
        return {
            "translation": [0, 1, 2],
            "rotation": [3, 4, 5],
            "gripper": [6],
        }
    if action_dim == 14:
        return {
            "translation": [0, 1, 2, 7, 8, 9],
            "rotation": [3, 4, 5, 10, 11, 12],
            "gripper": [6, 13],
        }
    raise ValueError(f"Unsupported action dimension: {action_dim}")


def bootstrap_ci(
    values: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values, size=(repetitions, len(values)), replace=True
    )
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    action_dim: int,
    device: torch.device,
    bootstrap_repetitions: int,
    seed: int,
) -> dict:
    model.eval()
    squared_error_chunks: list[np.ndarray] = []
    absolute_error_chunks: list[np.ndarray] = []
    demo_chunks: list[np.ndarray] = []
    progress_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, target, demo_index, progress in loader:
            sequence = sequence.to(device)
            target_device = target.to(device)
            prediction = model(sequence)
            difference = prediction - target_device
            squared_error_chunks.append(
                difference.square().cpu().numpy()
            )
            absolute_error_chunks.append(
                difference.abs().cpu().numpy()
            )
            demo_chunks.append(demo_index.numpy())
            progress_chunks.append(progress.numpy())

    squared = np.concatenate(squared_error_chunks)
    absolute = np.concatenate(absolute_error_chunks)
    demo_indices = np.concatenate(demo_chunks)
    progress = np.concatenate(progress_chunks)
    per_sample_mse = squared.mean(axis=1)

    trajectory_mse = np.asarray(
        [
            per_sample_mse[demo_indices == index].mean()
            for index in np.unique(demo_indices)
        ],
        dtype=np.float64,
    )
    ci_low, ci_high = bootstrap_ci(
        trajectory_mse, bootstrap_repetitions, seed
    )
    phase_metrics = {}
    for label, low, high in [
        ("early", 0.0, 0.25),
        ("middle", 0.25, 0.75),
        ("late", 0.75, 1.01),
    ]:
        mask = (progress >= low) & (progress < high)
        phase_metrics[label] = float(squared[mask].mean())

    component_metrics = {
        name: float(squared[:, indices].mean())
        for name, indices in component_indices(action_dim).items()
    }
    return {
        "mse": float(squared.mean()),
        "rmse": float(math.sqrt(squared.mean())),
        "mae": float(absolute.mean()),
        "trajectory_mse_mean": float(trajectory_mse.mean()),
        "trajectory_mse_std": float(trajectory_mse.std(ddof=1)),
        "trajectory_bootstrap_95_ci": [ci_low, ci_high],
        "phase_mse": phase_metrics,
        "component_mse": component_metrics,
        "validation_samples": int(len(squared)),
        "validation_trajectories": int(len(trajectory_mse)),
    }


def inference_latency_ms(
    model: nn.Module,
    example_batch: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    batch = example_batch[: min(256, len(example_batch))].to(device)
    with torch.no_grad():
        for _ in range(5):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations = []
        for _ in range(30):
            start = time.perf_counter()
            model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
    return float(np.median(durations) * 1000 / len(batch))


def train_one_model(
    model_name: str,
    task: str,
    train_tensors: tuple[torch.Tensor, ...],
    valid_tensors: tuple[torch.Tensor, ...],
    obs_dim: int,
    action_dim: int,
    config: Config,
    device: torch.device,
) -> tuple[dict, dict]:
    seed = config.seed + MODEL_NAMES.index(model_name) * 101
    set_seed(seed)
    model = create_model(
        model_name, obs_dim, action_dim, config.context_length
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    loss_function = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        TensorDataset(*valid_tensors),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    history = {"train_mse": [], "valid_mse": [], "learning_rate": []}
    best_valid = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_sum = 0.0
        train_elements = 0
        for sequence, target, _, _ in train_loader:
            sequence = sequence.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sequence)
            loss = loss_function(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_sum += float(loss.item()) * target.numel()
            train_elements += target.numel()

        model.eval()
        valid_sum = 0.0
        valid_elements = 0
        with torch.no_grad():
            for sequence, target, _, _ in valid_loader:
                sequence = sequence.to(device)
                target = target.to(device)
                loss = loss_function(model(sequence), target)
                valid_sum += float(loss.item()) * target.numel()
                valid_elements += target.numel()
        train_mse = train_sum / train_elements
        valid_mse = valid_sum / valid_elements
        scheduler.step(valid_mse)
        history["train_mse"].append(train_mse)
        history["valid_mse"].append(valid_mse)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        print(
            f"task={task:<9} model={model_name:<14} epoch={epoch:02d} "
            f"train={train_mse:.6f} valid={valid_mse:.6f}"
        )

        if valid_mse < best_valid - 1e-7:
            best_valid = valid_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    metrics = evaluate_model(
        model,
        valid_loader,
        action_dim,
        device,
        config.bootstrap_repetitions,
        seed,
    )
    metrics.update(
        {
            "model": model_name,
            "best_epoch": best_epoch,
            "epochs_ran": len(history["train_mse"]),
            "training_seconds": float(training_seconds),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "inference_ms_per_sample": inference_latency_ms(
                model, valid_tensors[0], device
            ),
        }
    )
    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{task}_{model_name.lower().replace('-', '_')}.pt"
    )
    torch.save(
        {
            "model_name": model_name,
            "task": task,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "context_length": config.context_length,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return metrics, history


def tensors_from_numpy(
    sequences: np.ndarray,
    actions: np.ndarray,
    demo_indices: np.ndarray,
    progress: np.ndarray,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(sequences),
        torch.from_numpy(actions),
        torch.from_numpy(demo_indices),
        torch.from_numpy(progress),
    )


def plot_dataset_summary(dataset_metrics: dict[str, dict]) -> None:
    tasks = list(dataset_metrics)
    samples = [dataset_metrics[task]["total_samples"] for task in tasks]
    mean_lengths = [
        dataset_metrics[task]["mean_trajectory_length"] for task in tasks
    ]
    obs_dims = [dataset_metrics[task]["observation_dim"] for task in tasks]
    action_dims = [dataset_metrics[task]["action_dim"] for task in tasks]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    bars = axes[0].bar(
        [task.title() for task in tasks],
        samples,
        color=["#176B87", "#2A9D8F", "#E9C46A", "#6D597A"][: len(tasks)],
    )
    axes[0].set(
        ylabel="State-action samples",
        title="Official RoboMimic PH low-dim datasets",
    )
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].bar_label(bars, fmt="%d", padding=3, fontsize=9)

    x = np.arange(len(tasks))
    width = 0.25
    axes[1].bar(
        x - width,
        mean_lengths,
        width,
        label="Mean trajectory length",
        color="#176B87",
    )
    axes[1].bar(
        x,
        obs_dims,
        width,
        label="Observation dim",
        color="#D1495B",
    )
    axes[1].bar(
        x + width,
        action_dims,
        width,
        label="Action dim",
        color="#2A9D8F",
    )
    axes[1].set(
        xticks=x,
        xticklabels=[task.title() for task in tasks],
        ylabel="Count",
        title="Trajectory and control dimensions",
    )
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp4_official_dataset.png", dpi=180)
    plt.close(fig)


def plot_model_comparison(results: dict[str, dict[str, dict]]) -> None:
    tasks = list(results)
    x = np.arange(len(tasks))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    for index, model_name in enumerate(MODEL_NAMES):
        values = [
            results[task][model_name]["trajectory_mse_mean"] for task in tasks
        ]
        lower = [
            value
            - results[task][model_name]["trajectory_bootstrap_95_ci"][0]
            for task, value in zip(tasks, values)
        ]
        upper = [
            results[task][model_name]["trajectory_bootstrap_95_ci"][1]
            - value
            for task, value in zip(tasks, values)
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            yerr=np.asarray([lower, upper]),
            capsize=3,
            color=MODEL_COLORS[model_name],
            label=model_name,
        )
    ax.set(
        xticks=x,
        xticklabels=[task.title() for task in tasks],
        ylabel="Validation action MSE",
        title="Official RoboMimic offline policy comparison",
        yscale="log",
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp4_official_model_comparison.png", dpi=180)
    plt.close(fig)


def plot_training_curves(histories: dict[str, dict[str, dict]]) -> None:
    tasks = list(histories)
    columns = 2 if len(tasks) > 1 else 1
    rows = math.ceil(len(tasks) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.2, 4.1 * rows),
        sharex=False,
        squeeze=False,
    )
    for ax, task in zip(axes.flat, tasks):
        for model_name in MODEL_NAMES:
            values = histories[task][model_name]["valid_mse"]
            ax.plot(
                np.arange(1, len(values) + 1),
                values,
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=MODEL_COLORS[model_name],
                label=model_name,
            )
        ax.set(
            xlabel="Epoch",
            ylabel="Validation MSE",
            title=task.title(),
            yscale="log",
        )
        ax.grid(alpha=0.22)
    for ax in axes.flat[len(tasks) :]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Training curves on official PH low-dimensional data", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(IMAGES_DIR / "exp4_official_training_curves.png", dpi=180)
    plt.close(fig)


def plot_error_breakdown(results: dict[str, dict[str, dict]]) -> None:
    tasks = list(results)
    labels = [
        f"{task.title()}\n{model_name.replace('BC-', '')}"
        for task in tasks
        for model_name in MODEL_NAMES
    ]
    components = ["translation", "rotation", "gripper", "late"]
    matrix = []
    for task in tasks:
        for model_name in MODEL_NAMES:
            metric = results[task][model_name]
            matrix.append(
                [
                    metric["component_mse"]["translation"],
                    metric["component_mse"]["rotation"],
                    metric["component_mse"]["gripper"],
                    metric["phase_mse"]["late"],
                ]
            )
    values = np.asarray(matrix).T
    display = np.log10(np.maximum(values, 1e-8))
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    image = ax.imshow(display, aspect="auto", cmap="viridis_r")
    ax.set(
        xticks=np.arange(len(labels)),
        xticklabels=labels,
        yticks=np.arange(len(components)),
        yticklabels=["Translation", "Rotation", "Gripper", "Late phase"],
        title="Action-error breakdown (color = log10 MSE)",
    )
    ax.tick_params(axis="x", rotation=50, labelsize=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=6.7,
                color="white" if display[row, column] > np.median(display) else "black",
            )
    fig.colorbar(image, ax=ax, label="log10 MSE", fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp4_official_error_breakdown.png", dpi=180)
    plt.close(fig)


def plot_efficiency(results: dict[str, dict[str, dict]]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    markers = {"lift": "o", "can": "s", "square": "^", "transport": "D"}
    for task in results:
        for model_name in MODEL_NAMES:
            metric = results[task][model_name]
            ax.scatter(
                metric["parameter_count"],
                metric["inference_ms_per_sample"],
                s=75,
                marker=markers[task],
                color=MODEL_COLORS[model_name],
                alpha=0.85,
            )
    model_handles = [
        ax.scatter([], [], color=MODEL_COLORS[model_name], label=model_name)
        for model_name in MODEL_NAMES
    ]
    task_handles = [
        ax.scatter(
            [],
            [],
            color="#555555",
            marker=marker,
            label=task.title(),
        )
        for task, marker in markers.items()
    ]
    ax.set(
        xlabel="Trainable parameters",
        ylabel="CPU inference time (ms/sample)",
        title="Model capacity and computational cost",
        xscale="log",
        yscale="log",
    )
    ax.grid(alpha=0.22)
    model_legend = ax.legend(
        handles=model_handles,
        title="Policy",
        frameon=False,
        loc="upper left",
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=task_handles,
        title="Task",
        frameon=False,
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp4_official_efficiency.png", dpi=180)
    plt.close(fig)


def run(config: Config, tasks: list[str]) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(config.torch_threads, os.cpu_count() or 1))
    device = resolve_device(config)
    print(f"device={device} torch={torch.__version__}")

    dataset_metrics: dict[str, dict] = {}
    results: dict[str, dict[str, dict]] = {}
    histories: dict[str, dict[str, dict]] = {}
    for task in tasks:
        path = DATA_DIR / TASK_FILES[task]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run download_official_datasets.py first."
            )
        metadata = read_task_metadata(task, path)
        if metadata["sha256"] != EXPECTED_SHA256[task]:
            raise ValueError(f"SHA-256 mismatch for {path}")
        dataset_metrics[task] = {
            key: value
            for key, value in metadata.items()
            if key not in {"train_keys", "valid_keys"}
        }

        mean, std = compute_normalization(
            path, metadata["train_keys"], metadata["observation_keys"]
        )
        train_arrays = build_windows(
            path,
            metadata["train_keys"],
            metadata["observation_keys"],
            mean,
            std,
            config.context_length,
            config.max_train_windows,
            config.seed,
        )
        valid_arrays = build_windows(
            path,
            metadata["valid_keys"],
            metadata["observation_keys"],
            mean,
            std,
            config.context_length,
            config.max_valid_windows,
            config.seed + 1,
        )
        train_tensors = tensors_from_numpy(*train_arrays)
        valid_tensors = tensors_from_numpy(*valid_arrays)
        dataset_metrics[task]["benchmark_train_windows"] = len(train_arrays[0])
        dataset_metrics[task]["benchmark_valid_windows"] = len(valid_arrays[0])
        results[task] = {}
        histories[task] = {}

        for model_name in MODEL_NAMES:
            metrics, history = train_one_model(
                model_name,
                task,
                train_tensors,
                valid_tensors,
                metadata["observation_dim"],
                metadata["action_dim"],
                config,
                device,
            )
            results[task][model_name] = metrics
            histories[task][model_name] = history

        del train_tensors, valid_tensors, train_arrays, valid_arrays

    benchmark = {
        "experiment": "Official RoboMimic v1.5 PH low-dimensional offline benchmark",
        "scope": (
            "Offline action prediction on official HDF5 train/valid masks. "
            "No MuJoCo rollout success is claimed."
        ),
        "config": asdict(config),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "h5py": h5py.__version__,
            "device": str(device),
            "platform": platform.platform(),
        },
        "datasets": dataset_metrics,
        "results": results,
    }
    (RESULTS_DIR / "official_benchmark_metrics.json").write_text(
        json.dumps(benchmark, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "official_training_histories.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )
    plot_dataset_summary(dataset_metrics)
    plot_model_comparison(results)
    plot_training_curves(histories)
    plot_error_breakdown(results)
    plot_efficiency(results)
    return benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_FILES),
        default=list(TASK_FILES),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-windows", type=int)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    benchmark_config = Config()
    if arguments.quick:
        benchmark_config = replace(
            benchmark_config,
            epochs=2,
            max_train_windows=8_000,
            max_valid_windows=3_000,
            bootstrap_repetitions=200,
            early_stopping_patience=2,
        )
    if arguments.epochs is not None:
        benchmark_config = replace(
            benchmark_config, epochs=arguments.epochs
        )
    if arguments.max_train_windows is not None:
        benchmark_config = replace(
            benchmark_config,
            max_train_windows=arguments.max_train_windows,
        )
    output = run(benchmark_config, arguments.tasks)
    print(
        json.dumps(
            {
                task: {
                    model: output["results"][task][model]["mse"]
                    for model in MODEL_NAMES
                }
                for task in arguments.tasks
            },
            indent=2,
        )
    )
