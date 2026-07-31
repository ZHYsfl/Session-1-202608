import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from robot_env_v3 import RobotReflectionEnv


# =========================================================
# 训练配置
# =========================================================
SEED = 2026

TOTAL_TIMESTEPS = 100_000

# 每5000步评价一次
EVAL_FREQUENCY = 5_000

# 每次评价20个固定初始状态
EVAL_EPISODES = 20
EVAL_BASE_SEED = 9000

# 每20000步保存一次检查点
CHECKPOINT_FREQUENCY = 20_000

POLICY_KWARGS = {
    "net_arch": [64, 64],
}


# =========================================================
# 文件路径
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "dqn_reflection_v3_safe"
)

CHECKPOINT_DIRECTORY = (
    MODEL_DIRECTORY
    / "checkpoints"
)

BEST_POLICY_PATH = (
    MODEL_DIRECTORY
    / "best_policy.pt"
)

FINAL_POLICY_PATH = (
    MODEL_DIRECTORY
    / "final_policy.pt"
)

CONFIG_PATH = (
    MODEL_DIRECTORY
    / "training_config.json"
)

LOG_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "dqn_v3_logs"
)

EVALUATION_LOG_PATH = (
    LOG_DIRECTORY
    / "evaluations.csv"
)


def create_dqn(
    environment,
    *,
    verbose: int,
) -> DQN:
    """建立第三版DQN模型。"""

    return DQN(
        policy="MlpPolicy",
        env=environment,

        policy_kwargs=POLICY_KWARGS,

        learning_rate=1e-4,

        buffer_size=100_000,
        learning_starts=5_000,

        batch_size=64,
        gamma=0.99,

        train_freq=4,
        gradient_steps=1,

        target_update_interval=2_000,

        exploration_fraction=0.40,
        exploration_initial_eps=1.00,
        exploration_final_eps=0.05,

        verbose=verbose,
        seed=SEED,
        device="cpu",
    )


def safe_save_policy(
    model: DQN,
    destination: Path,
    *,
    label: str,
    metrics: Optional[dict] = None,
) -> None:
    """
    使用普通PyTorch文件安全保存DQN策略。

    保存流程：
    1. 写入临时文件；
    2. 立即重新读取；
    3. 加载到全新DQN；
    4. 执行一次预测；
    5. 验证成功后替换正式文件。
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_name(
        destination.name + ".tmp"
    )

    policy_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor
        in model.policy.state_dict().items()
    }

    payload = {
        "algorithm": "DQN",
        "format": "safe_policy_state_dict_v3",
        "environment_version": "robot_env_v3",

        "policy_state_dict": policy_state,

        "policy_kwargs": {
            "net_arch": [64, 64],
        },

        "observation_shape": [14],
        "action_count": 2,

        "num_timesteps": int(
            model.num_timesteps
        ),

        "label": str(label),

        "selection_metrics": (
            metrics
            if metrics is not None
            else {}
        ),

        "torch_version": str(
            torch.__version__
        ),
    }

    verification_environment = None

    try:
        torch.save(
            payload,
            temporary_path,
        )

        loaded = torch.load(
            temporary_path,
            map_location="cpu",
            weights_only=True,
        )

        if (
            loaded.get("format")
            != "safe_policy_state_dict_v3"
        ):
            raise RuntimeError(
                "模型格式标记不正确。"
            )

        if (
            loaded.get("environment_version")
            != "robot_env_v3"
        ):
            raise RuntimeError(
                "模型环境版本不正确。"
            )

        if (
            loaded.get("observation_shape")
            != [14]
        ):
            raise RuntimeError(
                "模型状态维数不正确。"
            )

        if loaded.get("action_count") != 2:
            raise RuntimeError(
                "模型动作数量不正确。"
            )

        verification_environment = (
            RobotReflectionEnv()
        )

        verification_model = create_dqn(
            verification_environment,
            verbose=0,
        )

        verification_model.policy.load_state_dict(
            loaded["policy_state_dict"],
            strict=True,
        )

        verification_model.policy.set_training_mode(
            False
        )

        observation, _ = (
            verification_environment.reset(
                seed=SEED
            )
        )

        action, _ = (
            verification_model.predict(
                observation,
                deterministic=True,
            )
        )

        action_value = int(
            np.asarray(action).item()
        )

        if action_value not in (0, 1):
            raise RuntimeError(
                "验证模型输出了无效动作。"
            )

        os.replace(
            temporary_path,
            destination,
        )

    finally:
        if verification_environment is not None:
            verification_environment.close()

        if temporary_path.exists():
            temporary_path.unlink()

    print(
        f"安全保存验证成功：{destination}"
    )


def evaluate_current_policy(
    model: DQN,
) -> dict:
    """使用固定的20个Episode评价当前策略。"""

    environment = RobotReflectionEnv()

    rewards = []
    lengths = []
    collisions = []
    survivals = []
    reflections = []
    emergencies = []
    average_speeds = []
    action_one_ratios = []

    try:
        for episode_index in range(
            EVAL_EPISODES
        ):
            episode_seed = (
                EVAL_BASE_SEED
                + episode_index
            )

            observation, _ = (
                environment.reset(
                    seed=episode_seed
                )
            )

            total_reward = 0.0
            speed_sum = 0.0
            action_one_count = 0

            terminated = False
            truncated = False

            info = {}

            while not (
                terminated or truncated
            ):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                action_value = int(
                    np.asarray(action).item()
                )

                if action_value == 1:
                    action_one_count += 1

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = environment.step(
                    action_value
                )

                total_reward += float(
                    reward
                )

                speed_sum += float(
                    info["speed"]
                )

            episode_length = int(
                info["step_count"]
            )

            collision = int(
                info["episode_collisions"] > 0
            )

            survived = int(
                truncated
                and not terminated
            )

            rewards.append(
                total_reward
            )

            lengths.append(
                episode_length
            )

            collisions.append(
                collision
            )

            survivals.append(
                survived
            )

            reflections.append(
                int(
                    info[
                        "episode_reflections"
                    ]
                )
            )

            emergencies.append(
                int(
                    info[
                        "episode_emergency_triggers"
                    ]
                )
            )

            average_speeds.append(
                speed_sum
                / max(
                    episode_length,
                    1,
                )
            )

            action_one_ratios.append(
                action_one_count
                / max(
                    episode_length,
                    1,
                )
            )

    finally:
        environment.close()

    return {
        "mean_reward": float(
            np.mean(rewards)
        ),

        "reward_std": float(
            np.std(rewards)
        ),

        "mean_episode_length": float(
            np.mean(lengths)
        ),

        "episode_length_std": float(
            np.std(lengths)
        ),

        "collision_rate": float(
            np.mean(collisions)
        ),

        "survival_rate": float(
            np.mean(survivals)
        ),

        "mean_reflections": float(
            np.mean(reflections)
        ),

        "mean_emergencies": float(
            np.mean(emergencies)
        ),

        "mean_speed": float(
            np.mean(average_speeds)
        ),

        "mean_action_1_ratio": float(
            np.mean(action_one_ratios)
        ),
    }


def is_better_policy(
    current: dict,
    best: Optional[dict],
) -> bool:
    """
    判断当前策略是否优于已保存策略。

    优先顺序：
    1. 碰撞率更低；
    2. 生存率更高；
    3. 平均Episode长度更长；
    4. 平均奖励更高；
    5. 紧急保护次数更少。
    """

    if best is None:
        return True

    tolerance = 1e-9

    if (
        current["collision_rate"]
        < best["collision_rate"]
        - tolerance
    ):
        return True

    if (
        current["collision_rate"]
        > best["collision_rate"]
        + tolerance
    ):
        return False

    if (
        current["survival_rate"]
        > best["survival_rate"]
        + tolerance
    ):
        return True

    if (
        current["survival_rate"]
        < best["survival_rate"]
        - tolerance
    ):
        return False

    if (
        current["mean_episode_length"]
        > best["mean_episode_length"]
        + tolerance
    ):
        return True

    if (
        current["mean_episode_length"]
        < best["mean_episode_length"]
        - tolerance
    ):
        return False

    if (
        current["mean_reward"]
        > best["mean_reward"]
        + tolerance
    ):
        return True

    if (
        current["mean_reward"]
        < best["mean_reward"]
        - tolerance
    ):
        return False

    return (
        current["mean_emergencies"]
        < best["mean_emergencies"]
        - tolerance
    )


def append_evaluation_record(
    record: dict,
) -> None:
    """把评价结果追加到CSV文件。"""

    file_exists = (
        EVALUATION_LOG_PATH.exists()
    )

    with EVALUATION_LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                record.keys()
            ),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            record
        )


class DQNV2Callback(BaseCallback):
    """第三版评价与安全保存回调。"""

    def __init__(self) -> None:
        super().__init__(verbose=1)

        self.best_metrics = None

    def _on_step(self) -> bool:
        current_step = int(
            self.num_timesteps
        )

        # =================================================
        # 定期评价
        # =================================================
        if (
            current_step
            % EVAL_FREQUENCY
            == 0
        ):
            metrics = (
                evaluate_current_policy(
                    self.model
                )
            )

            record = {
                "timesteps": current_step,
                **metrics,
            }

            append_evaluation_record(
                record
            )

            print()
            print("=" * 68)
            print(
                f"V3安全评价 step={current_step}"
            )

            print(
                "碰撞率："
                f"{metrics['collision_rate']:.2%}"
            )

            print(
                "运行到600步比例："
                f"{metrics['survival_rate']:.2%}"
            )

            print(
                "平均Episode长度："
                f"{metrics['mean_episode_length']:.2f} "
                "± "
                f"{metrics['episode_length_std']:.2f}"
            )

            print(
                "平均奖励："
                f"{metrics['mean_reward']:.2f} "
                "± "
                f"{metrics['reward_std']:.2f}"
            )

            print(
                "平均反射次数："
                f"{metrics['mean_reflections']:.2f}"
            )

            print(
                "平均紧急保护次数："
                f"{metrics['mean_emergencies']:.2f}"
            )

            print(
                "平均运动速度："
                f"{metrics['mean_speed']:.2f}"
            )

            print(
                "动作1选择比例："
                f"{metrics['mean_action_1_ratio']:.2%}"
            )

            print("=" * 68)

            if is_better_policy(
                metrics,
                self.best_metrics,
            ):
                self.best_metrics = (
                    metrics.copy()
                )

                safe_save_policy(
                    self.model,
                    BEST_POLICY_PATH,
                    label=(
                        f"best_v3_at_"
                        f"{current_step}_steps"
                    ),
                    metrics=metrics,
                )

                print(
                    "发现更安全的策略，"
                    "已保存为最佳模型。"
                )

        # =================================================
        # 定期检查点
        # =================================================
        if (
            current_step
            % CHECKPOINT_FREQUENCY
            == 0
        ):
            checkpoint_path = (
                CHECKPOINT_DIRECTORY
                / (
                    f"policy_v3_"
                    f"{current_step}_steps.pt"
                )
            )

            safe_save_policy(
                self.model,
                checkpoint_path,
                label=(
                    f"checkpoint_v3_"
                    f"{current_step}"
                ),
            )

        return True


def main() -> int:
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 清除上一次V2训练的评价日志
    if EVALUATION_LOG_PATH.exists():
        EVALUATION_LOG_PATH.unlink()

    training_config = {
        "environment": "robot_env_v3",

        "seed": SEED,

        "total_timesteps": (
            TOTAL_TIMESTEPS
        ),

        "eval_frequency": (
            EVAL_FREQUENCY
        ),

        "eval_episodes": (
            EVAL_EPISODES
        ),

        "checkpoint_frequency": (
            CHECKPOINT_FREQUENCY
        ),

        "network": [
            14,
            64,
            64,
            2,
        ],

        "best_model_priority": [
            "lowest_collision_rate",
            "highest_survival_rate",
            "longest_episode",
            "highest_reward",
            "fewest_emergencies",
        ],

        "actions": {
            "0": "保持当前方向",

            "1": (
                "减速-严格镜面反射-"
                "重新加速"
            ),
        },
    }

    CONFIG_PATH.write_text(
        json.dumps(
            training_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_environment = (
        RobotReflectionEnv()
    )

    training_environment = Monitor(
        raw_environment,

        filename=str(
            LOG_DIRECTORY
            / "train"
        ),

        info_keywords=(
            "episode_reflections",
            "episode_emergency_triggers",
            "episode_collisions",
        ),
    )

    model = create_dqn(
        training_environment,
        verbose=1,
    )

    callback = DQNV2Callback()

    print("=" * 72)
    print("开始第三版DQN训练")
    print(
        f"总训练步数：{TOTAL_TIMESTEPS}"
    )
    print(
        f"每次评价Episode数："
        f"{EVAL_EPISODES}"
    )
    print(
        "最佳模型首先按照碰撞率选择"
    )
    print(
        "模型采用普通PyTorch .pt安全保存"
    )
    print("=" * 72)

    try:
        model.learn(
            total_timesteps=(
                TOTAL_TIMESTEPS
            ),

            callback=callback,

            log_interval=10,

            reset_num_timesteps=True,
        )

        final_metrics = (
            evaluate_current_policy(
                model
            )
        )

        safe_save_policy(
            model,
            FINAL_POLICY_PATH,
            label="final_v3_100000_steps",
            metrics=final_metrics,
        )

    finally:
        training_environment.close()

    print()
    print("=" * 72)
    print("第三版DQN训练完成")

    print(
        f"最佳策略：{BEST_POLICY_PATH}"
    )

    print(
        f"最终策略：{FINAL_POLICY_PATH}"
    )

    print(
        f"评价日志：{EVALUATION_LOG_PATH}"
    )

    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())