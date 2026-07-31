"""Analyze measured outputs from Experiments 1, 2, and 4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"

MODEL_NAMES = ("BC", "BC-RNN", "BC-Transformer")
MODEL_COLORS = {
    "BC": "#D1495B",
    "BC-RNN": "#E09F3E",
    "BC-Transformer": "#176B87",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_width(states: np.ndarray) -> np.ndarray:
    low, high = np.percentile(states, [1, 99], axis=0)
    return high - low


def average_model_mse(exp4: dict) -> dict[str, float]:
    return {
        model_name: float(
            np.mean(
                [
                    task_results[model_name]["mse"]
                    for task_results in exp4["results"].values()
                ]
            )
        )
        for model_name in MODEL_NAMES
    }


def plot_summary(exp1: dict, exp2: dict, exp4: dict) -> None:
    model_mse = average_model_mse(exp4)
    exp1_results = exp1["results"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))

    axes[0].bar(
        ["Train", "Validation"],
        [
            exp1_results["final_train_mse"],
            exp1_results["final_validation_mse"],
        ],
        color=["#176B87", "#D1495B"],
        width=0.62,
    )
    axes[0].set(
        ylabel="Final MSE",
        title="Experiment 1: driving BC",
        ylim=(0, 0.08),
    )
    axes[0].grid(axis="y", alpha=0.22)

    bars = axes[1].bar(
        ["BC", "DAgger", "Expert"],
        [
            exp2["bc"]["mean_return"],
            exp2["dagger"]["mean_return"],
            exp2["expert"]["mean_return"],
        ],
        yerr=[
            exp2["bc"]["std_return"],
            exp2["dagger"]["std_return"],
            exp2["expert"]["std_return"],
        ],
        capsize=4,
        color=["#D1495B", "#176B87", "#2A9D8F"],
        width=0.62,
    )
    axes[1].set(
        ylabel="Mean return",
        title="Experiment 2: CartPole",
        ylim=(0, 550),
    )
    axes[1].grid(axis="y", alpha=0.22)
    for bar, value in zip(
        bars,
        [
            exp2["bc"]["success_rate"],
            exp2["dagger"]["success_rate"],
            exp2["expert"]["success_rate"],
        ],
        strict=True,
    ):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 12,
            f"{100 * value:.0f}%",
            ha="center",
        )

    names = list(MODEL_NAMES)
    values = [model_mse[name] for name in names]
    bars = axes[2].bar(
        ["BC", "BC-RNN", "Transformer"],
        values,
        color=[MODEL_COLORS[name] for name in names],
        width=0.62,
    )
    axes[2].set(
        ylabel="Mean validation action MSE",
        title="Experiment 4: four-task average",
        ylim=(0, max(values) * 1.22),
    )
    axes[2].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, values, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.025,
            f"{value:.4f}",
            ha="center",
            fontsize=9,
        )

    fig.suptitle("Imitation-learning experiment summary", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(IMAGES_DIR / "exp3_summary.png", dpi=180)
    plt.close(fig)


def plot_coverage_gain(
    initial_states: np.ndarray, aggregated_states: np.ndarray, exp2: dict
) -> dict:
    initial_width = coverage_width(initial_states)
    aggregated_width = coverage_width(aggregated_states)
    gain = aggregated_width / np.maximum(initial_width, 1e-12)
    labels = ["x", "x velocity", "pole angle", "angle velocity"]

    history = exp2["history"]
    samples = np.asarray([entry["dataset_size"] for entry in history])
    returns = np.asarray([entry["mean_return"] for entry in history])
    success = 100 * np.asarray([entry["success_rate"] for entry in history])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    axes[0].bar(
        labels,
        gain,
        color=["#176B87", "#E9C46A", "#D1495B", "#2A9D8F"],
    )
    axes[0].set(
        ylabel="99%-range gain (DAgger / initial)",
        title="State coverage expansion",
    )
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].plot(
        samples,
        returns,
        marker="o",
        linewidth=2.2,
        color="#176B87",
        label="Return",
    )
    axis_right = axes[1].twinx()
    axis_right.plot(
        samples,
        success,
        marker="s",
        linewidth=2.0,
        color="#D1495B",
        label="Success",
    )
    axes[1].set(
        xlabel="Aggregated samples",
        ylabel="Mean return",
        title="Aggregated data and performance",
        xscale="log",
        ylim=(0, 530),
    )
    axis_right.set(ylabel="Success rate (%)", ylim=(0, 105))
    axes[1].grid(alpha=0.22)
    lines = axes[1].lines + axis_right.lines
    axes[1].legend(
        lines,
        [line.get_label() for line in lines],
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp3_coverage_gain.png", dpi=180)
    plt.close(fig)
    return {
        label: {
            "initial_1_to_99_percent_width": float(initial_width[index]),
            "aggregated_1_to_99_percent_width": float(
                aggregated_width[index]
            ),
            "coverage_gain": float(gain[index]),
        }
        for index, label in enumerate(labels)
    }


def downsample(
    states: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    if len(states) <= count:
        return states
    return states[rng.choice(len(states), count, replace=False)]


def plot_policy_state_distribution(state_data: np.lib.npyio.NpzFile) -> dict:
    rng = np.random.default_rng(20260730)
    groups = {
        "Expert": state_data["expert_states"],
        "BC": state_data["bc_states"],
        "DAgger": state_data["dagger_states"],
    }
    colors = {"Expert": "#2F6BFF", "BC": "#D1495B", "DAgger": "#2A9D8F"}
    sampled = {
        name: downsample(states, 5_000, rng)
        for name, states in groups.items()
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    for name in ("Expert", "BC", "DAgger"):
        states = sampled[name]
        ax.scatter(
            states[:, 0],
            states[:, 2],
            s=8,
            alpha=0.20,
            color=colors[name],
            edgecolors="none",
            label=f"{name} (n={len(groups[name]):,})",
        )
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="#777777", linewidth=0.8, alpha=0.5)
    ax.set(
        xlabel="Cart position x (m)",
        ylabel=r"Pole angle $\theta$ (rad)",
        title=r"Policy-visited state distributions in $(x,\theta)$",
        xlim=(-2.5, 2.5),
        ylim=(-0.22, 0.22),
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, markerscale=2.2)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp3_policy_state_distribution.png", dpi=200)
    plt.close(fig)

    return {
        name.lower(): {
            "state_count": int(len(states)),
            "x_1_to_99_percent": [
                float(value) for value in np.percentile(states[:, 0], [1, 99])
            ],
            "theta_1_to_99_percent": [
                float(value) for value in np.percentile(states[:, 2], [1, 99])
            ],
        }
        for name, states in groups.items()
    }


def plot_error_analysis(exp2: dict) -> dict:
    errors = exp2["action_error"]
    keys = ["expert_states", "bc_states", "dagger_states", "pooled"]
    labels = ["Expert states", "BC states", "DAgger states", "Pooled"]
    x = np.arange(len(keys))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis, metric, ylabel, scale in (
        (axes[0], "probability_mse", "Action probability MSE", 1),
        (axes[1], "action_error_rate", "Action error rate (%)", 100),
    ):
        bc_values = [scale * errors["bc"][key][metric] for key in keys]
        dagger_values = [
            scale * errors["dagger"][key][metric] for key in keys
        ]
        axis.bar(
            x - width / 2,
            bc_values,
            width,
            color="#D1495B",
            label="BC",
        )
        axis.bar(
            x + width / 2,
            dagger_values,
            width,
            color="#176B87",
            label="DAgger",
        )
        axis.set(
            ylabel=ylabel,
            xticks=x,
            xticklabels=labels,
        )
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False)
    axes[0].set_title("Expert-action probability error")
    axes[1].set_title("Deterministic action disagreement")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp3_error_analysis.png", dpi=180)
    plt.close(fig)

    bc_pooled = errors["bc"]["pooled"]
    dagger_pooled = errors["dagger"]["pooled"]
    return {
        "definition": errors["definition"],
        "bc_pooled": bc_pooled,
        "dagger_pooled": dagger_pooled,
        "mse_reduction_percent": float(
            100
            * (
                1
                - dagger_pooled["probability_mse"]
                / bc_pooled["probability_mse"]
            )
        ),
        "action_error_reduction_percentage_points": float(
            100
            * (
                bc_pooled["action_error_rate"]
                - dagger_pooled["action_error_rate"]
            )
        ),
    }


def summarize_exp4(exp4: dict) -> dict:
    means = average_model_mse(exp4)
    tasks = {
        task: {
            model_name: {
                "mse": values[model_name]["mse"],
                "trajectory_bootstrap_95_ci": values[model_name][
                    "trajectory_bootstrap_95_ci"
                ],
            }
            for model_name in MODEL_NAMES
        }
        for task, values in exp4["results"].items()
    }
    return {
        "tasks": tasks,
        "four_task_mean_mse": means,
        "transformer_improvement_over_bc_percent": float(
            100 * (1 - means["BC-Transformer"] / means["BC"])
        ),
        "best_model_by_task": {
            task: min(
                MODEL_NAMES,
                key=lambda name: values[name]["mse"],
            )
            for task, values in exp4["results"].items()
        },
    }


def run() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    exp2 = load_json(
        PROJECT_ROOT / "Experiment2_DAgger" / "results" / "metrics.json"
    )
    exp1 = load_json(
        PROJECT_ROOT
        / "Experiment1_Behavior_Cloning"
        / "results"
        / "metrics.json"
    )
    exp4 = load_json(
        PROJECT_ROOT
        / "Experiment4_RoboMimic"
        / "results"
        / "official_benchmark_metrics.json"
    )
    coverage_data = np.load(
        PROJECT_ROOT
        / "Experiment2_DAgger"
        / "results"
        / "state_coverage.npz"
    )
    policy_state_data = np.load(
        PROJECT_ROOT
        / "Experiment2_DAgger"
        / "results"
        / "policy_state_distributions.npz"
    )

    plot_summary(exp1, exp2, exp4)
    coverage = plot_coverage_gain(
        coverage_data["initial_states"],
        coverage_data["aggregated_states"],
        exp2,
    )
    distributions = plot_policy_state_distribution(policy_state_data)
    error_analysis = plot_error_analysis(exp2)
    exp4_summary = summarize_exp4(exp4)
    summary = {
        "experiment_1": {
            "status": exp1["status"],
            "dataset_samples": exp1["dataset"]["recorded_samples"],
            "model_parameters": exp1["model"]["parameter_count"],
            "train_mse": exp1["results"]["final_train_mse"],
            "validation_mse": exp1["results"]["final_validation_mse"],
            "model_h5_sha256": exp1["verification"]["model_h5_sha256"],
        },
        "experiment_2": {
            "bc_mean_return": exp2["bc"]["mean_return"],
            "bc_success_rate": exp2["bc"]["success_rate"],
            "dagger_mean_return": exp2["dagger"]["mean_return"],
            "dagger_success_rate": exp2["dagger"]["success_rate"],
            "absolute_return_gain": (
                exp2["dagger"]["mean_return"] - exp2["bc"]["mean_return"]
            ),
            "success_rate_gain_percentage_points": 100
            * (
                exp2["dagger"]["success_rate"]
                - exp2["bc"]["success_rate"]
            ),
            "dataset_growth_factor": (
                exp2["dagger"]["dataset_size"] / exp2["bc"]["dataset_size"]
            ),
            "beta_ablation": exp2["beta_ablation"],
            "state_coverage": coverage,
        },
        "experiment_3": {
            "action_error_analysis": error_analysis,
            "policy_state_distributions": distributions,
        },
        "experiment_4": exp4_summary,
    }
    (RESULTS_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
