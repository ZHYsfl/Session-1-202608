"""Reproducible DAgger experiment on a self-contained CartPole environment.

The CartPole dynamics follow the Gym / Gymnasium classic-control equations.
Evaluation uses a wider reset distribution than the initial demonstrations so
that covariate shift, DAgger recovery, and beta-schedule effects are measurable.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
IMAGES_DIR = ROOT.parent / "images"


@dataclass(frozen=True)
class Config:
    seed: int = 20260730
    max_steps: int = 500
    initial_expert_episodes: int = 2
    dagger_iterations: int = 6
    dagger_rollouts: int = 10
    beta_decay: float = 0.5
    ablation_rollouts: int = 3
    evaluation_episodes: int = 80
    distribution_episodes: int = 40
    solved_steps: int = 475


class CartPole:
    """Minimal CartPole-v1 dynamics without a Gymnasium dependency."""

    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4

    def __init__(self, rng: np.random.Generator, max_steps: int = 500):
        self.rng = rng
        self.max_steps = max_steps
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0

    def reset(self, wide: bool = False) -> np.ndarray:
        if wide:
            low = np.array([-0.35, -0.45, -0.105, -0.45])
            high = np.array([0.35, 0.45, 0.105, 0.45])
            self.state = self.rng.uniform(low, high)
        else:
            self.state = self.rng.uniform(-0.02, 0.02, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if int(action) == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.steps += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= self.max_steps
        )
        return self.state.copy(), terminated


def expert_policy(state: np.ndarray) -> int:
    """A stabilizing expert with extra correction near track boundaries."""

    x, x_dot, theta, theta_dot = state
    balance_signal = (
        1.05 * theta
        + 0.19 * theta_dot
        + 0.012 * x
        + 0.018 * x_dot
        + 0.35 * theta**3
    )
    if abs(x) > 1.65:
        balance_signal += 0.055 * np.sign(x)
    return int(balance_signal > 0.0)


def build_policy(seed: int, sample_count: int) -> Pipeline:
    max_iter = 180 if sample_count < 5_000 else 240
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 32),
                    activation="tanh",
                    alpha=2e-4,
                    batch_size=min(256, sample_count),
                    learning_rate_init=2e-3,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_policy(states: np.ndarray, actions: np.ndarray, seed: int) -> Pipeline:
    model = build_policy(seed, len(states))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(states, actions)
    return model


def collect_expert_data(
    config: Config, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[int] = []
    env = CartPole(rng, config.max_steps)
    for _ in range(config.initial_expert_episodes):
        state = env.reset(wide=False)
        done = False
        while not done:
            action = expert_policy(state)
            states.append(state)
            actions.append(action)
            state, done = env.step(action)
    return np.asarray(states), np.asarray(actions)


def evaluate(
    policy: Pipeline,
    config: Config,
    seeds: np.ndarray,
    capture_first: bool = False,
) -> tuple[dict[str, float], np.ndarray]:
    lengths: list[int] = []
    first_angles: list[float] = []
    for episode_seed in seeds:
        env = CartPole(np.random.default_rng(int(episode_seed)), config.max_steps)
        state = env.reset(wide=True)
        done = False
        episode_angles: list[float] = []
        while not done:
            episode_angles.append(float(state[2]))
            action = int(policy.predict(state.reshape(1, -1))[0])
            state, done = env.step(action)
        lengths.append(env.steps)
        if capture_first and not len(first_angles):
            first_angles = episode_angles

    values = np.asarray(lengths, dtype=np.float64)
    metrics = {
        "mean_return": float(values.mean()),
        "std_return": float(values.std()),
        "median_return": float(np.median(values)),
        "success_rate": float(np.mean(values >= config.solved_steps)),
    }
    return metrics, np.asarray(first_angles)


def evaluate_expert(config: Config, seeds: np.ndarray) -> dict[str, float]:
    lengths: list[int] = []
    for episode_seed in seeds:
        env = CartPole(np.random.default_rng(int(episode_seed)), config.max_steps)
        state = env.reset(wide=True)
        done = False
        while not done:
            state, done = env.step(expert_policy(state))
        lengths.append(env.steps)
    values = np.asarray(lengths, dtype=np.float64)
    return {
        "mean_return": float(values.mean()),
        "std_return": float(values.std()),
        "median_return": float(np.median(values)),
        "success_rate": float(np.mean(values >= config.solved_steps)),
    }


def aggregate_dagger_data(
    policy: Pipeline,
    config: Config,
    rng: np.random.Generator,
    beta: float,
    rollouts: int,
) -> tuple[np.ndarray, np.ndarray]:
    queried_states: list[np.ndarray] = []
    expert_actions: list[int] = []
    env = CartPole(rng, config.max_steps)
    for _ in range(rollouts):
        state = env.reset(wide=True)
        done = False
        while not done:
            label = expert_policy(state)
            queried_states.append(state)
            expert_actions.append(label)
            learner_action = int(policy.predict(state.reshape(1, -1))[0])
            behavior_action = label if rng.random() < beta else learner_action
            state, done = env.step(behavior_action)
    return np.asarray(queried_states), np.asarray(expert_actions)


def run_dagger_schedule(
    initial_states: np.ndarray,
    initial_actions: np.ndarray,
    config: Config,
    evaluation_seeds: np.ndarray,
    beta_decay: float,
    rollouts: int,
    seed_offset: int,
) -> tuple[list[dict[str, float]], Pipeline, np.ndarray, np.ndarray]:
    states = initial_states.copy()
    actions = initial_actions.copy()
    rng = np.random.default_rng(config.seed + seed_offset)
    policy = fit_policy(states, actions, config.seed)
    baseline, _ = evaluate(policy, config, evaluation_seeds)
    history: list[dict[str, float]] = [
        {
            "iteration": 0,
            "beta": 1.0,
            "dataset_size": int(len(states)),
            **baseline,
        }
    ]

    for iteration in range(1, config.dagger_iterations + 1):
        beta = beta_decay**iteration
        new_states, new_actions = aggregate_dagger_data(
            policy, config, rng, beta, rollouts
        )
        states = np.concatenate([states, new_states], axis=0)
        actions = np.concatenate([actions, new_actions], axis=0)
        policy = fit_policy(
            states, actions, config.seed + seed_offset + iteration
        )
        iteration_metrics, _ = evaluate(policy, config, evaluation_seeds)
        history.append(
            {
                "iteration": iteration,
                "beta": float(beta),
                "dataset_size": int(len(states)),
                **iteration_metrics,
            }
        )
    return history, policy, states, actions


def collect_policy_states(
    policy: Pipeline | None,
    config: Config,
    seeds: np.ndarray,
) -> np.ndarray:
    states: list[np.ndarray] = []
    for episode_seed in seeds:
        env = CartPole(np.random.default_rng(int(episode_seed)), config.max_steps)
        state = env.reset(wide=True)
        done = False
        while not done:
            states.append(state)
            action = (
                expert_policy(state)
                if policy is None
                else int(policy.predict(state.reshape(1, -1))[0])
            )
            state, done = env.step(action)
    return np.asarray(states)


def prediction_error(policy: Pipeline, states: np.ndarray) -> dict[str, float]:
    labels = np.asarray([expert_policy(state) for state in states])
    class_index = int(np.where(policy.classes_ == 1)[0][0])
    probabilities = policy.predict_proba(states)[:, class_index]
    clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "probability_mse": float(np.mean((probabilities - labels) ** 2)),
        "action_error_rate": float(np.mean(predictions != labels)),
        "binary_cross_entropy": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        ),
        "sample_count": int(len(states)),
    }


def plot_learning(metrics: dict) -> None:
    iterations = np.arange(len(metrics["history"]))
    returns = np.asarray(
        [entry["mean_return"] for entry in metrics["history"]]
    )
    success = 100 * np.asarray(
        [entry["success_rate"] for entry in metrics["history"]]
    )
    bc_return = metrics["bc"]["mean_return"]
    bc_success = 100 * metrics["bc"]["success_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(
        iterations,
        np.full_like(returns, bc_return),
        linestyle="--",
        linewidth=2,
        color="#D1495B",
        label="Fixed BC",
    )
    axes[0].plot(
        iterations,
        returns,
        marker="o",
        linewidth=2.2,
        color="#176B87",
        label="DAgger",
    )
    axes[0].axhline(
        metrics["expert"]["mean_return"],
        color="#2A9D8F",
        linestyle=":",
        linewidth=2,
        label="Expert",
    )
    axes[0].set(
        xlabel="DAgger iteration",
        ylabel="Mean episode return",
        title="BC baseline vs DAgger",
        xticks=iterations,
        ylim=(0, 520),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        iterations,
        np.full_like(success, bc_success),
        linestyle="--",
        linewidth=2,
        color="#D1495B",
        label="Fixed BC",
    )
    axes[1].plot(
        iterations,
        success,
        marker="s",
        linewidth=2.2,
        color="#176B87",
        label="DAgger",
    )
    axes[1].set(
        xlabel="DAgger iteration",
        ylabel="Success rate (%)",
        title="Closed-loop success",
        xticks=iterations,
        ylim=(-3, 103),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp2_learning_curve.png", dpi=180)
    plt.close(fig)


def plot_beta_ablation(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    colors = ["#A23B72", "#176B87", "#E09F3E"]
    for (name, history), color in zip(
        metrics["beta_ablation"].items(), colors, strict=True
    ):
        iterations = [entry["iteration"] for entry in history]
        label = f"$\\lambda={name}$"
        axes[0].plot(
            iterations,
            [entry["mean_return"] for entry in history],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        axes[1].plot(
            iterations,
            [100 * entry["success_rate"] for entry in history],
            marker="s",
            linewidth=2,
            color=color,
            label=label,
        )
        axes[2].plot(
            iterations,
            [entry["dataset_size"] for entry in history],
            marker="^",
            linewidth=2,
            color=color,
            label=label,
        )

    axes[0].set(
        xlabel="DAgger iteration",
        ylabel="Mean episode return",
        title=r"Return under $\beta_k=\lambda^k$",
        ylim=(0, 520),
    )
    axes[1].set(
        xlabel="DAgger iteration",
        ylabel="Success rate (%)",
        title=r"Success under $\beta_k=\lambda^k$",
        ylim=(-3, 103),
    )
    axes[2].set(
        xlabel="DAgger iteration",
        ylabel="Aggregated state-action samples",
        title="Annotation volume",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp2_beta_ablation.png", dpi=180)
    plt.close(fig)


def plot_action_error(error_analysis: dict) -> None:
    sources = ["Expert states", "BC states", "DAgger states", "Pooled"]
    keys = ["expert_states", "bc_states", "dagger_states", "pooled"]
    bc_mse = [
        error_analysis["bc"][key]["probability_mse"] for key in keys
    ]
    dagger_mse = [
        error_analysis["dagger"][key]["probability_mse"] for key in keys
    ]
    x = np.arange(len(sources))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width / 2, bc_mse, width, color="#D1495B", label="BC")
    ax.bar(
        x + width / 2,
        dagger_mse,
        width,
        color="#176B87",
        label="DAgger",
    )
    ax.set(
        ylabel="Action probability MSE",
        title="Expert-action prediction error on held-out policy states",
        xticks=x,
        xticklabels=sources,
    )
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp2_action_error.png", dpi=180)
    plt.close(fig)


def plot_coverage(initial_states: np.ndarray, all_states: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    aggregate_only = all_states[len(initial_states) :]
    step = max(1, len(aggregate_only) // 6_000)
    ax.scatter(
        aggregate_only[::step, 2],
        aggregate_only[::step, 3],
        s=8,
        alpha=0.18,
        color="#2A9D8F",
        label="DAgger queried states",
    )
    ax.scatter(
        initial_states[:, 2],
        initial_states[:, 3],
        s=13,
        alpha=0.50,
        color="#D1495B",
        label="Initial expert demonstrations",
    )
    ax.set(
        xlabel="Pole angle (rad)",
        ylabel="Pole angular velocity (rad/s)",
        title="State-distribution coverage",
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp2_state_coverage.png", dpi=180)
    plt.close(fig)


def plot_rollouts(
    bc_angles: np.ndarray, dagger_angles: np.ndarray, threshold: float
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot(bc_angles, color="#D1495B", linewidth=2, label="BC rollout")
    ax.plot(dagger_angles, color="#176B87", linewidth=2, label="DAgger rollout")
    ax.axhline(threshold, color="#555555", linestyle="--", linewidth=1)
    ax.axhline(-threshold, color="#555555", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Time step",
        ylabel="Pole angle (rad)",
        title="Matched-seed rollout comparison",
        xlim=(0, max(len(bc_angles), len(dagger_angles))),
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp2_rollout.png", dpi=180)
    plt.close(fig)


def run(config: Config) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    initial_rng = np.random.default_rng(config.seed)
    evaluation_seeds = np.random.default_rng(config.seed + 1).integers(
        0, 2**31 - 1, size=config.evaluation_episodes
    )
    distribution_seeds = np.random.default_rng(config.seed + 2).integers(
        0, 2**31 - 1, size=config.distribution_episodes
    )
    initial_states, initial_actions = collect_expert_data(config, initial_rng)

    bc_policy = fit_policy(initial_states, initial_actions, config.seed)
    bc_metrics, bc_angles = evaluate(
        bc_policy, config, evaluation_seeds, capture_first=True
    )
    history, dagger_policy, states, actions = run_dagger_schedule(
        initial_states,
        initial_actions,
        config,
        evaluation_seeds,
        config.beta_decay,
        config.dagger_rollouts,
        seed_offset=100,
    )
    final_metrics, dagger_angles = evaluate(
        dagger_policy, config, evaluation_seeds, capture_first=True
    )

    beta_ablation: dict[str, list[dict[str, float]]] = {}
    for index, decay in enumerate((0.9, 0.5, 0.1)):
        schedule, _, _, _ = run_dagger_schedule(
            initial_states,
            initial_actions,
            config,
            evaluation_seeds,
            decay,
            config.ablation_rollouts,
            seed_offset=1_000 + 100 * index,
        )
        beta_ablation[f"{decay:.1f}"] = schedule
        print(
            f"beta_decay={decay:.1f} final_return="
            f"{schedule[-1]['mean_return']:.1f} "
            f"success={schedule[-1]['success_rate']:.1%}"
        )

    expert_states = collect_policy_states(None, config, distribution_seeds)
    bc_states = collect_policy_states(bc_policy, config, distribution_seeds)
    dagger_states = collect_policy_states(
        dagger_policy, config, distribution_seeds
    )
    pooled_states = np.concatenate(
        [expert_states, bc_states, dagger_states], axis=0
    )
    sample_rng = np.random.default_rng(config.seed + 3)
    if len(pooled_states) > 30_000:
        pooled_states = pooled_states[
            sample_rng.choice(len(pooled_states), size=30_000, replace=False)
        ]
    state_groups = {
        "expert_states": expert_states,
        "bc_states": bc_states,
        "dagger_states": dagger_states,
        "pooled": pooled_states,
    }
    error_analysis = {
        "definition": (
            "MSE between predicted P(expert action=1|state) and the deterministic "
            "expert label on independent policy-visited states."
        ),
        "bc": {
            name: prediction_error(bc_policy, values)
            for name, values in state_groups.items()
        },
        "dagger": {
            name: prediction_error(dagger_policy, values)
            for name, values in state_groups.items()
        },
    }

    expert_metrics = evaluate_expert(config, evaluation_seeds)
    metrics = {
        "experiment": "CartPole-v1 DAgger",
        "config": asdict(config),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "evaluation": {
            "reset_distribution": {
                "x": [-0.35, 0.35],
                "x_dot": [-0.45, 0.45],
                "theta": [-0.105, 0.105],
                "theta_dot": [-0.45, 0.45],
            },
            "episodes": config.evaluation_episodes,
            "success_definition": f"episode length >= {config.solved_steps}",
        },
        "expert": expert_metrics,
        "bc": {**history[0], **bc_metrics},
        "dagger": {**history[-1], **final_metrics},
        "history": history,
        "learning_curve": [
            {
                "iteration": entry["iteration"],
                "bc_mean_return": bc_metrics["mean_return"],
                "bc_success_rate": bc_metrics["success_rate"],
                "dagger_mean_return": entry["mean_return"],
                "dagger_success_rate": entry["success_rate"],
            }
            for entry in history
        ],
        "beta_ablation": beta_ablation,
        "action_error": error_analysis,
    }

    joblib.dump(bc_policy, RESULTS_DIR / "bc_policy.joblib")
    joblib.dump(dagger_policy, RESULTS_DIR / "dagger_policy.joblib")
    np.savez_compressed(
        RESULTS_DIR / "state_coverage.npz",
        initial_states=initial_states,
        aggregated_states=states,
        labels=actions,
    )
    np.savez_compressed(
        RESULTS_DIR / "policy_state_distributions.npz",
        expert_states=expert_states,
        bc_states=bc_states,
        dagger_states=dagger_states,
    )
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    plot_learning(metrics)
    plot_beta_ablation(metrics)
    plot_action_error(error_analysis)
    plot_coverage(initial_states, states)
    plot_rollouts(
        bc_angles, dagger_angles, CartPole.theta_threshold_radians
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use fewer evaluations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = (
        Config(evaluation_episodes=30, distribution_episodes=15)
        if args.quick
        else Config()
    )
    output = run(config)
    print(
        json.dumps(
            {
                "bc": output["bc"],
                "dagger": output["dagger"],
                "action_error": output["action_error"],
            },
            indent=2,
        )
    )
