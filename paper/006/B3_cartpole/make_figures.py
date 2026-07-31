"""Generate publication figures from the frozen B3 formal experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = (
    REPOSITORY_ROOT / "src" / "006" / "B3_cartpole" / "outputs" / "formal"
)
FIGURE_ROOT = Path(__file__).resolve().parent / "figures"
SEEDS = (42, 123, 2026)
VARIANTS = (
    "dqn_standard",
    "dqn_normalized",
    "double_dqn",
    "double_dqn_fast_exploration",
    "q_coarse",
    "q_medium",
    "q_fine",
)
LABELS = {
    "dqn_standard": "Standard DQN",
    "dqn_normalized": "Normalized DQN",
    "double_dqn": "Double DQN",
    "double_dqn_fast_exploration": "Full DQN",
    "q_coarse": "Q-Coarse",
    "q_medium": "Q-Medium",
    "q_fine": "Q-Fine",
    "random": "Random",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_ROOT / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_ROOT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def evaluation_values(
    variant: str,
    checkpoint: str,
    metric: str,
) -> np.ndarray:
    values = []
    for seed in SEEDS:
        path = (
            RESULT_ROOT
            / "runs"
            / variant
            / f"seed_{seed}"
            / "evaluation_final.json"
        )
        values.append(read_json(path)["checkpoints"][checkpoint][metric])
    return np.asarray(values, dtype=float)


def random_values(metric: str) -> np.ndarray:
    values = []
    for seed in SEEDS:
        path = (
            RESULT_ROOT
            / "runs"
            / "random"
            / f"seed_{seed}"
            / "evaluation_final.json"
        )
        values.append(read_json(path)["checkpoints"]["policy"][metric])
    return np.asarray(values, dtype=float)


def performance_figure() -> None:
    variants = ("random", *VARIANTS)
    episode_means = []
    episode_stds = []
    success_means = []
    success_stds = []
    for variant in variants:
        if variant == "random":
            steps = random_values("mean_steps")
            success = 100.0 * random_values("success_rate")
        else:
            steps = evaluation_values(variant, "best", "mean_steps")
            success = 100.0 * evaluation_values(
                variant,
                "best",
                "success_rate",
            )
        episode_means.append(steps.mean())
        episode_stds.append(steps.std(ddof=1))
        success_means.append(success.mean())
        success_stds.append(success.std(ddof=1))

    positions = np.arange(len(variants))
    colors = ["#7f7f7f"] + ["#4c78a8"] * 4 + ["#59a14f"] * 3
    colors[6] = "#e15759"
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.25))
    axes[0].barh(
        positions,
        episode_means,
        xerr=episode_stds,
        color=colors,
        capsize=2.5,
    )
    axes[0].axvline(500, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Mean episode length")
    axes[0].set_xlim(0, 620)
    axes[1].barh(
        positions,
        success_means,
        xerr=success_stds,
        color=colors,
        capsize=2.5,
    )
    axes[1].axvline(100, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Success rate (%)")
    axes[1].set_xlim(0, 145)
    for axis in axes:
        axis.set_yticks(positions, [LABELS[value] for value in variants])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        axis.tick_params(labelsize=7.5)
    axes[1].tick_params(labelleft=False)
    figure.tight_layout()
    save_figure(figure, "final_performance")


def checkpoint_figure() -> None:
    positions = np.arange(len(VARIANTS))
    best = [
        evaluation_values(variant, "best", "mean_steps").mean()
        for variant in VARIANTS
    ]
    final = [
        evaluation_values(variant, "final", "mean_steps").mean()
        for variant in VARIANTS
    ]
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.16, 3.0))
    axis.bar(positions - width / 2, best, width, label="Best checkpoint")
    axis.bar(positions + width / 2, final, width, label="Final checkpoint")
    axis.set_xticks(
        positions,
        [LABELS[value] for value in VARIANTS],
        rotation=24,
        ha="right",
    )
    axis.set_ylabel("Mean episode length")
    axis.set_ylim(0, 525)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    axis.tick_params(labelsize=7.5)
    figure.tight_layout()
    save_figure(figure, "checkpoint_degradation")


def validation_curves() -> None:
    groups = {
        "DQN variants": VARIANTS[:4],
        "Q-Learning discretizations": VARIANTS[4:],
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.95), sharey=True)
    for axis, (title, variants) in zip(axes, groups.items(), strict=True):
        for variant in variants:
            seed_curves: list[dict[int, float]] = []
            for seed in SEEDS:
                path = (
                    RESULT_ROOT
                    / "runs"
                    / variant
                    / f"seed_{seed}"
                    / "training_metrics.json"
                )
                payload = read_json(path)
                seed_curves.append(
                    {
                        int(item["global_step"]): float(item["mean_steps"])
                        for item in payload["validation"]
                    }
                )
            common_steps = sorted(
                set.intersection(*(set(curve) for curve in seed_curves))
            )
            values = np.asarray(
                [[curve[step] for step in common_steps] for curve in seed_curves]
            )
            mean = values.mean(axis=0)
            standard_deviation = values.std(axis=0, ddof=1)
            axis.plot(common_steps, mean, label=LABELS[variant], linewidth=1.25)
            axis.fill_between(
                common_steps,
                np.clip(mean - standard_deviation, 0, 500),
                np.clip(mean + standard_deviation, 0, 500),
                alpha=0.13,
            )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Environment steps")
        axis.set_xlim(5_000, 70_000)
        axis.set_ylim(0, 510)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=6.8, loc="upper right")
        axis.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Validation episode length")
    figure.tight_layout()
    save_figure(figure, "validation_curves")


def main() -> None:
    if not RESULT_ROOT.exists():
        raise FileNotFoundError(
            "Formal results are required at "
            "src/006/B3_cartpole/outputs/formal."
        )
    performance_figure()
    checkpoint_figure()
    validation_curves()
    print(f"Figures written to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
