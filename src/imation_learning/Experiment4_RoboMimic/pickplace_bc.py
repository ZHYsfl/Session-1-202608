"""Lightweight RoboMimic-style low-dimensional BC on a PickPlace task.

This is a local, simulator-free reproduction of RoboMimic's core offline
learning workflow: collect demonstration trajectories, split by trajectory,
train a state-to-action policy, and evaluate the policy with closed-loop
rollouts on unseen task instances.
"""

from __future__ import annotations

import json
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT.parent / "images"


@dataclass(frozen=True)
class Config:
    seed: int = 20260730
    demonstrations: int = 240
    validation_fraction: float = 0.2
    epochs: int = 90
    evaluation_episodes: int = 100
    max_steps: int = 120
    max_delta: float = 0.055
    success_radius: float = 0.075


class PickPlaceEnv:
    """Small continuous-control environment with grasp and release dynamics."""

    def __init__(self, rng: np.random.Generator, config: Config):
        self.rng = rng
        self.config = config
        self.eef = np.zeros(3)
        self.obj = np.zeros(3)
        self.target = np.zeros(3)
        self.gripper_open = True
        self.held = False
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.eef = np.array(
            [
                self.rng.uniform(-0.05, 0.05),
                self.rng.uniform(-0.08, 0.08),
                self.rng.uniform(0.48, 0.58),
            ]
        )
        self.obj = np.array(
            [
                self.rng.uniform(-0.43, -0.13),
                self.rng.uniform(-0.30, 0.30),
                0.035,
            ]
        )
        self.target = np.array(
            [
                self.rng.uniform(0.16, 0.46),
                self.rng.uniform(-0.30, 0.30),
                0.035,
            ]
        )
        self.gripper_open = True
        self.held = False
        self.steps = 0
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate(
            [
                self.eef,
                self.obj,
                self.target,
                np.array([float(self.gripper_open)]),
                self.obj - self.eef,
                self.target - self.obj,
            ]
        ).astype(np.float64)

    def is_success(self) -> bool:
        return bool(
            not self.held
            and np.linalg.norm(self.obj[:2] - self.target[:2])
            <= self.config.success_radius
            and abs(self.obj[2] - self.target[2]) <= 0.04
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.eef += action[:3] * self.config.max_delta
        self.eef = np.clip(
            self.eef,
            np.array([-0.60, -0.48, 0.035]),
            np.array([0.60, 0.48, 0.65]),
        )

        if action[3] < -0.25:
            self.gripper_open = False
            if np.linalg.norm(self.eef - self.obj) <= 0.072:
                self.held = True
        elif action[3] > 0.25:
            self.gripper_open = True
            if self.held:
                self.held = False
                self.obj[2] = 0.035

        if self.held:
            self.obj = self.eef + np.array([0.0, 0.0, -0.028])

        self.steps += 1
        success = self.is_success()
        done = success or self.steps >= self.config.max_steps
        return self.observation(), done, success


def move_action(env: PickPlaceEnv, waypoint: np.ndarray, grip: float) -> np.ndarray:
    delta = (waypoint - env.eef) / env.config.max_delta
    return np.concatenate([np.clip(delta, -1.0, 1.0), [grip]])


def expert_policy(env: PickPlaceEnv) -> np.ndarray:
    """Waypoint expert: approach, grasp, lift, transport, descend, release."""

    if not env.held and env.gripper_open:
        xy_error = np.linalg.norm(env.eef[:2] - env.obj[:2])
        if xy_error > 0.025:
            return move_action(
                env, np.array([env.obj[0], env.obj[1], 0.22]), 1.0
            )
        if env.eef[2] > 0.078:
            return move_action(
                env, np.array([env.obj[0], env.obj[1], 0.058]), 1.0
            )
        return np.array([0.0, 0.0, 0.0, -1.0])

    if env.held:
        target_xy_error = np.linalg.norm(env.eef[:2] - env.target[:2])
        if target_xy_error > 0.028 and env.eef[2] < 0.33:
            return move_action(
                env, np.array([env.eef[0], env.eef[1], 0.39]), -1.0
            )
        if target_xy_error > 0.028:
            return move_action(
                env, np.array([env.target[0], env.target[1], 0.39]), -1.0
            )
        if env.eef[2] > 0.083:
            return move_action(
                env, np.array([env.target[0], env.target[1], 0.060]), -1.0
            )
        return np.array([0.0, 0.0, 0.0, 1.0])

    # Recovery label for an unsuccessful early close.
    return np.array([0.0, 0.0, 0.5, 1.0])


def collect_demonstrations(
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict]:
    rng = np.random.default_rng(config.seed)
    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_dones: list[bool] = []
    trajectory_ids: list[int] = []
    lengths: list[int] = []
    successful = 0

    for demo_id in range(config.demonstrations):
        env = PickPlaceEnv(rng, config)
        obs = env.reset()
        done = False
        trajectory_length = 0
        while not done:
            action = expert_policy(env)
            # Small control variation produces a less degenerate training set.
            action[:3] = np.clip(
                action[:3] + rng.normal(0.0, 0.012, size=3), -1.0, 1.0
            )
            all_obs.append(obs)
            all_actions.append(action.copy())
            trajectory_ids.append(demo_id)
            obs, done, success = env.step(action)
            all_dones.append(done)
            trajectory_length += 1
        successful += int(success)
        lengths.append(trajectory_length)

    manifest = {
        "format": "robomimic-style flattened trajectory archive",
        "official_equivalent": {
            "group": "data",
            "trajectory_keys": "demo_0 ... demo_N",
            "datasets": ["obs/low_dim", "actions", "dones"],
        },
        "observation_keys": [
            "robot0_eef_pos",
            "object_pos",
            "target_pos",
            "robot0_gripper_open",
            "object_to_eef",
            "target_to_object",
        ],
        "observation_dim": 16,
        "action_keys": ["delta_x", "delta_y", "delta_z", "gripper_command"],
        "action_dim": 4,
        "num_trajectories": config.demonstrations,
        "total_samples": len(all_obs),
        "expert_success_rate": successful / config.demonstrations,
        "trajectory_lengths": lengths,
    }
    return (
        np.asarray(all_obs),
        np.asarray(all_actions),
        np.asarray(trajectory_ids),
        lengths,
        manifest,
    )


def train_policy(
    obs: np.ndarray,
    actions: np.ndarray,
    trajectory_ids: np.ndarray,
    config: Config,
) -> tuple[dict, dict[str, list[float]], dict]:
    split_rng = np.random.default_rng(config.seed + 11)
    demo_ids = np.arange(config.demonstrations)
    split_rng.shuffle(demo_ids)
    valid_count = int(round(config.validation_fraction * len(demo_ids)))
    valid_ids = set(demo_ids[:valid_count].tolist())
    valid_mask = np.array([int(i) in valid_ids for i in trajectory_ids])
    train_mask = ~valid_mask

    scaler = StandardScaler().fit(obs[train_mask])
    x_train = scaler.transform(obs[train_mask])
    x_valid = scaler.transform(obs[valid_mask])
    movement_train = actions[train_mask, :3]
    movement_valid = actions[valid_mask, :3]
    grip_train = (actions[train_mask, 3] > 0).astype(int)
    grip_valid = (actions[valid_mask, 3] > 0).astype(int)

    movement_model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=1,
        random_state=config.seed,
    )
    grip_model = MLPClassifier(
        hidden_layer_sizes=(32, 32),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=1,
        random_state=config.seed + 1,
    )

    history = {
        "epoch": [],
        "train_movement_mse": [],
        "valid_movement_mse": [],
        "train_gripper_accuracy": [],
        "valid_gripper_accuracy": [],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for epoch in range(1, config.epochs + 1):
            movement_model.partial_fit(x_train, movement_train)
            grip_model.partial_fit(x_train, grip_train, classes=np.array([0, 1]))

            train_move_pred = movement_model.predict(x_train)
            valid_move_pred = movement_model.predict(x_valid)
            train_grip_pred = grip_model.predict(x_train)
            valid_grip_pred = grip_model.predict(x_valid)
            history["epoch"].append(epoch)
            history["train_movement_mse"].append(
                float(mean_squared_error(movement_train, train_move_pred))
            )
            history["valid_movement_mse"].append(
                float(mean_squared_error(movement_valid, valid_move_pred))
            )
            history["train_gripper_accuracy"].append(
                float(accuracy_score(grip_train, train_grip_pred))
            )
            history["valid_gripper_accuracy"].append(
                float(accuracy_score(grip_valid, valid_grip_pred))
            )

    policy = {
        "scaler": scaler,
        "movement_model": movement_model,
        "gripper_model": grip_model,
    }
    split_info = {
        "train_trajectories": int(config.demonstrations - valid_count),
        "valid_trajectories": valid_count,
        "train_samples": int(train_mask.sum()),
        "valid_samples": int(valid_mask.sum()),
    }
    final = {key: values[-1] for key, values in history.items() if key != "epoch"}
    return policy, history, {"split": split_info, "final": final}


def policy_action(policy: dict, observation: np.ndarray) -> np.ndarray:
    scaled = policy["scaler"].transform(observation.reshape(1, -1))
    movement = np.clip(policy["movement_model"].predict(scaled)[0], -1.0, 1.0)
    grip_open = int(policy["gripper_model"].predict(scaled)[0])
    grip_command = 1.0 if grip_open else -1.0
    return np.concatenate([movement, [grip_command]])


def evaluate_policy(
    policy: dict, config: Config
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    rng = np.random.default_rng(config.seed + 101)
    success_values: list[bool] = []
    episode_lengths: list[int] = []
    placement_errors: list[float] = []
    captured: dict[str, np.ndarray] | None = None

    for _ in range(config.evaluation_episodes):
        env = PickPlaceEnv(rng, config)
        obs = env.reset()
        initial_obj = env.obj.copy()
        target = env.target.copy()
        done = False
        eef_path = [env.eef.copy()]
        obj_path = [env.obj.copy()]
        while not done:
            action = policy_action(policy, obs)
            obs, done, success = env.step(action)
            eef_path.append(env.eef.copy())
            obj_path.append(env.obj.copy())

        success_values.append(success)
        episode_lengths.append(env.steps)
        placement_errors.append(float(np.linalg.norm(env.obj[:2] - env.target[:2])))
        if success and captured is None:
            captured = {
                "eef_path": np.asarray(eef_path),
                "obj_path": np.asarray(obj_path),
                "initial_obj": initial_obj,
                "target": target,
            }

    if captured is None:
        captured = {
            "eef_path": np.asarray(eef_path),
            "obj_path": np.asarray(obj_path),
            "initial_obj": initial_obj,
            "target": target,
        }
    success_array = np.asarray(success_values, dtype=np.float64)
    length_array = np.asarray(episode_lengths, dtype=np.float64)
    error_array = np.asarray(placement_errors, dtype=np.float64)
    metrics = {
        "success_rate": float(success_array.mean()),
        "successful_episodes": int(success_array.sum()),
        "mean_episode_length": float(length_array.mean()),
        "median_episode_length": float(np.median(length_array)),
        "mean_final_xy_error": float(error_array.mean()),
        "median_final_xy_error": float(np.median(error_array)),
    }
    return metrics, captured


def plot_training(history: dict[str, list[float]]) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    axes[0].plot(
        epochs, history["train_movement_mse"], label="Train", color="#176B87"
    )
    axes[0].plot(
        epochs, history["valid_movement_mse"], label="Validation", color="#D1495B"
    )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Movement MSE",
        title="Continuous action learning",
        yscale="log",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    axes[1].plot(
        epochs,
        np.asarray(history["train_gripper_accuracy"]) * 100,
        label="Train",
        color="#176B87",
    )
    axes[1].plot(
        epochs,
        np.asarray(history["valid_gripper_accuracy"]) * 100,
        label="Validation",
        color="#D1495B",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Gripper accuracy (%)",
        title="Discrete gripper learning",
        ylim=(75, 101),
    )
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp4_training.png", dpi=180)
    plt.close(fig)


def plot_dataset(lengths: list[int], actions: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.2))
    axes[0].hist(lengths, bins=14, color="#176B87", edgecolor="white")
    axes[0].set(
        xlabel="Steps per demonstration",
        ylabel="Trajectory count",
        title="Expert trajectory lengths",
    )
    labels = ["dx", "dy", "dz", "gripper"]
    means = np.mean(np.abs(actions), axis=0)
    axes[1].bar(labels, means, color=["#2A9D8F", "#E9C46A", "#D1495B", "#6D597A"])
    axes[1].set(
        ylabel="Mean absolute command",
        title="Action-channel activity",
        ylim=(0, 1.05),
    )
    axes[1].grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp4_dataset.png", dpi=180)
    plt.close(fig)


def plot_trajectory(rollout: dict[str, np.ndarray]) -> None:
    eef = rollout["eef_path"]
    obj = rollout["obj_path"]
    initial_obj = rollout["initial_obj"]
    target = rollout["target"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    axes[0].plot(eef[:, 0], eef[:, 1], color="#176B87", linewidth=2, label="EEF")
    axes[0].plot(obj[:, 0], obj[:, 1], color="#D1495B", linewidth=2, label="Object")
    axes[0].scatter(
        initial_obj[0], initial_obj[1], s=75, color="#D1495B", marker="s"
    )
    axes[0].scatter(target[0], target[1], s=95, color="#2A9D8F", marker="*")
    axes[0].add_patch(
        Circle(
            target[:2],
            radius=0.075,
            facecolor="#2A9D8F",
            alpha=0.12,
            edgecolor="#2A9D8F",
        )
    )
    axes[0].set(
        xlabel="x position",
        ylabel="y position",
        title="Top-down PickPlace rollout",
        xlim=(-0.55, 0.55),
        ylim=(-0.45, 0.45),
        aspect="equal",
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(eef[:, 2], color="#176B87", linewidth=2, label="EEF height")
    axes[1].plot(obj[:, 2], color="#D1495B", linewidth=2, label="Object height")
    axes[1].set(
        xlabel="Time step",
        ylabel="z position",
        title="Grasp, lift, and placement",
        ylim=(0, 0.65),
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "exp4_trajectory.png", dpi=180)
    plt.close(fig)


def run(config: Config) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    obs, actions, trajectory_ids, lengths, manifest = collect_demonstrations(config)
    np.savez_compressed(
        DATA_DIR / "pickplace_low_dim.npz",
        observations=obs,
        actions=actions,
        trajectory_ids=trajectory_ids,
    )
    (DATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    policy, history, training = train_policy(
        obs, actions, trajectory_ids, config
    )
    rollout_metrics, rollout = evaluate_policy(policy, config)
    metrics = {
        "experiment": "RoboMimic-style low-dimensional PickPlace BC",
        "scope": (
            "Local simulator-free reproduction of the offline BC workflow; "
            "not the official MuJoCo benchmark."
        ),
        "config": asdict(config),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "trajectories": config.demonstrations,
            "samples": int(len(obs)),
            "mean_trajectory_length": float(np.mean(lengths)),
            "expert_success_rate": manifest["expert_success_rate"],
            "observation_dim": manifest["observation_dim"],
            "action_dim": manifest["action_dim"],
        },
        "training": training,
        "rollout": rollout_metrics,
    }

    joblib.dump(policy, RESULTS_DIR / "pickplace_bc_policy.joblib")
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    plot_training(history)
    plot_dataset(lengths, actions)
    plot_trajectory(rollout)
    return metrics


if __name__ == "__main__":
    output = run(Config())
    print(json.dumps(output, indent=2))
