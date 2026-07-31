import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN

from robot_env_v3 import RobotReflectionEnv


# =========================================================
# 最终盲测设置
# =========================================================
EPISODE_NUMBER = 300

# 这批种子没有参与V1、V2、V3训练期间的选模
BASE_SEED = 15000

POLICY_KWARGS = {
    "net_arch": [64, 64],
}


# =========================================================
# 文件路径
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

V1_POLICY_PATH = (
    PROJECT_ROOT
    / "models"
    / "dqn_reflection_safe"
    / "best_policy.pt"
)

V3_POLICY_PATH = (
    PROJECT_ROOT
    / "models"
    / "dqn_reflection_v3_safe"
    / "best_policy.pt"
)

RESULT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "final_dqn_comparison"
)

EPISODE_RESULT_PATH = (
    RESULT_DIRECTORY
    / "episode_results.csv"
)

SUMMARY_PATH = (
    RESULT_DIRECTORY
    / "summary.csv"
)

PAIRED_RESULT_PATH = (
    RESULT_DIRECTORY
    / "v1_v3_paired_comparison.csv"
)


def create_dqn(
    environment,
) -> DQN:
    """创建与V1、V3完全相同结构的DQN。"""

    return DQN(
        policy="MlpPolicy",
        env=environment,
        policy_kwargs=POLICY_KWARGS,
        device="cpu",
        verbose=0,
        seed=2026,
    )


def load_safe_policy(
    policy_path: Path,
    allowed_formats: tuple[str, ...],
):
    """读取普通PyTorch格式保存的DQN策略。"""

    if not policy_path.exists():
        raise FileNotFoundError(
            f"找不到策略文件：{policy_path}"
        )

    payload = torch.load(
        policy_path,
        map_location="cpu",
        weights_only=True,
    )

    file_format = payload.get("format")

    if file_format not in allowed_formats:
        raise ValueError(
            f"不支持的模型格式：{file_format}"
        )

    if payload.get("observation_shape") != [14]:
        raise ValueError(
            f"{policy_path.name}的状态维数不是14。"
        )

    if payload.get("action_count") != 2:
        raise ValueError(
            f"{policy_path.name}的动作数不是2。"
        )

    construction_environment = (
        RobotReflectionEnv()
    )

    model = create_dqn(
        construction_environment
    )

    model.policy.load_state_dict(
        payload["policy_state_dict"],
        strict=True,
    )

    model.policy.set_training_mode(
        False
    )

    print()
    print(f"模型读取成功：{policy_path}")
    print(f"文件格式：{file_format}")
    print(f"保存标签：{payload.get('label')}")
    print(
        "保存时训练步数："
        f"{payload.get('num_timesteps')}"
    )

    return model


def wilson_interval(
    collision_number: int,
    episode_number: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """计算碰撞率的95% Wilson置信区间。"""

    if episode_number <= 0:
        return 0.0, 0.0

    proportion = (
        collision_number
        / episode_number
    )

    denominator = (
        1.0
        + z_value**2
        / episode_number
    )

    center = (
        proportion
        + z_value**2
        / (2.0 * episode_number)
    ) / denominator

    margin = (
        z_value
        * np.sqrt(
            (
                proportion
                * (1.0 - proportion)
                + z_value**2
                / (4.0 * episode_number)
            )
            / episode_number
        )
        / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def evaluate_policy(
    policy_name: str,
    model: Optional[DQN] = None,
    rule_name: Optional[str] = None,
) -> list[dict]:
    """使用相同的300个随机种子评价一种策略。"""

    environment = RobotReflectionEnv()

    records = []

    for episode_index in range(
        EPISODE_NUMBER
    ):
        episode_seed = (
            BASE_SEED
            + episode_index
        )

        random_generator = (
            np.random.default_rng(
                episode_seed + 100_000
            )
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
            if model is not None:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                action_value = int(
                    np.asarray(action).item()
                )

            elif rule_name == "continue":
                # 始终选择动作0，
                # 只依靠环境的紧急保护
                action_value = 0

            elif rule_name == "random":
                action_value = int(
                    random_generator.integers(
                        low=0,
                        high=2,
                    )
                )

            else:
                raise RuntimeError(
                    "没有提供有效策略。"
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

        records.append(
            {
                "policy": policy_name,
                "episode": episode_index + 1,
                "seed": episode_seed,
                "reward": total_reward,
                "episode_length": episode_length,
                "collision": collision,
                "survived_to_limit": survived,
                "reflections": int(
                    info["episode_reflections"]
                ),
                "emergency_triggers": int(
                    info[
                        "episode_emergency_triggers"
                    ]
                ),
                "average_speed": (
                    speed_sum
                    / max(
                        episode_length,
                        1,
                    )
                ),
                "action_1_count": (
                    action_one_count
                ),
                "action_1_ratio": (
                    action_one_count
                    / max(
                        episode_length,
                        1,
                    )
                ),
            }
        )

        if (
            episode_index + 1
        ) % 50 == 0:
            print(
                f"{policy_name}："
                f"已完成"
                f"{episode_index + 1}/"
                f"{EPISODE_NUMBER}个Episode"
            )

    environment.close()

    return records


def summarize_results(
    episode_data: pd.DataFrame,
) -> pd.DataFrame:
    """汇总每种策略的评价结果。"""

    summaries = []

    for (
        policy_name,
        group,
    ) in episode_data.groupby(
        "policy",
        sort=False,
    ):
        collision_number = int(
            group["collision"].sum()
        )

        (
            collision_ci_low,
            collision_ci_high,
        ) = wilson_interval(
            collision_number,
            len(group),
        )

        summaries.append(
            {
                "policy": policy_name,
                "episodes": len(group),

                "collision_episodes": (
                    collision_number
                ),

                "collision_rate": (
                    group["collision"].mean()
                ),

                "collision_ci95_low": (
                    collision_ci_low
                ),

                "collision_ci95_high": (
                    collision_ci_high
                ),

                "survival_rate": (
                    group[
                        "survived_to_limit"
                    ].mean()
                ),

                "mean_episode_length": (
                    group[
                        "episode_length"
                    ].mean()
                ),

                "episode_length_std": (
                    group[
                        "episode_length"
                    ].std(
                        ddof=0
                    )
                ),

                "mean_reward": (
                    group["reward"].mean()
                ),

                "reward_std": (
                    group["reward"].std(
                        ddof=0
                    )
                ),

                "mean_reflections": (
                    group[
                        "reflections"
                    ].mean()
                ),

                "mean_emergency_triggers": (
                    group[
                        "emergency_triggers"
                    ].mean()
                ),

                "mean_speed": (
                    group[
                        "average_speed"
                    ].mean()
                ),

                "mean_action_1_ratio": (
                    group[
                        "action_1_ratio"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        summaries
    )


def create_paired_comparison(
    episode_data: pd.DataFrame,
) -> pd.DataFrame:
    """比较V1和V3在相同初始状态下的碰撞结果。"""

    v1_data = (
        episode_data[
            episode_data["policy"]
            == "V1 Best DQN"
        ][
            [
                "seed",
                "collision",
                "episode_length",
            ]
        ]
        .rename(
            columns={
                "collision": "v1_collision",
                "episode_length": (
                    "v1_episode_length"
                ),
            }
        )
    )

    v3_data = (
        episode_data[
            episode_data["policy"]
            == "V3 Best DQN"
        ][
            [
                "seed",
                "collision",
                "episode_length",
            ]
        ]
        .rename(
            columns={
                "collision": "v3_collision",
                "episode_length": (
                    "v3_episode_length"
                ),
            }
        )
    )

    paired = v1_data.merge(
        v3_data,
        on="seed",
        how="inner",
        validate="one_to_one",
    )

    paired["result"] = np.select(
        [
            (
                (paired["v1_collision"] == 0)
                & (paired["v3_collision"] == 0)
            ),
            (
                (paired["v1_collision"] == 1)
                & (paired["v3_collision"] == 0)
            ),
            (
                (paired["v1_collision"] == 0)
                & (paired["v3_collision"] == 1)
            ),
            (
                (paired["v1_collision"] == 1)
                & (paired["v3_collision"] == 1)
            ),
        ],
        [
            "Both safe",
            "V3 safer",
            "V1 safer",
            "Both collided",
        ],
        default="Unknown",
    )

    return paired


def print_summary(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    """打印最终汇总结果。"""

    print()
    print("=" * 92)
    print("最终盲测结果")
    print("=" * 92)

    for _, row in summary.iterrows():
        print()
        print(
            f"策略：{row['policy']}"
        )

        print(
            "碰撞Episode："
            f"{int(row['collision_episodes'])}"
            f"/{int(row['episodes'])}"
        )

        print(
            "碰撞率："
            f"{row['collision_rate']:.2%}"
        )

        print(
            "碰撞率95%置信区间："
            f"[{row['collision_ci95_low']:.2%}, "
            f"{row['collision_ci95_high']:.2%}]"
        )

        print(
            "运行到600步比例："
            f"{row['survival_rate']:.2%}"
        )

        print(
            "平均Episode长度："
            f"{row['mean_episode_length']:.2f}"
        )

        print(
            "平均反射次数："
            f"{row['mean_reflections']:.2f}"
        )

        print(
            "平均紧急保护次数："
            f"{row['mean_emergency_triggers']:.2f}"
        )

        print(
            "平均运动速度："
            f"{row['mean_speed']:.2f}"
        )

        print(
            "动作1选择比例："
            f"{row['mean_action_1_ratio']:.2%}"
        )

    paired_counts = (
        paired["result"]
        .value_counts()
        .to_dict()
    )

    print()
    print("=" * 92)
    print("V1与V3相同初始状态的配对比较")
    print("=" * 92)

    print(
        "两者都未碰撞：",
        paired_counts.get(
            "Both safe",
            0,
        ),
    )

    print(
        "只有V1碰撞、V3安全：",
        paired_counts.get(
            "V3 safer",
            0,
        ),
    )

    print(
        "只有V3碰撞、V1安全：",
        paired_counts.get(
            "V1 safer",
            0,
        ),
    )

    print(
        "两者都碰撞：",
        paired_counts.get(
            "Both collided",
            0,
        ),
    )

    print("=" * 92)


def main() -> int:
    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("正在加载V1最佳策略……")

    v1_model = load_safe_policy(
        V1_POLICY_PATH,
        allowed_formats=(
            "safe_policy_state_dict_v1",
        ),
    )

    print()
    print("正在加载V3最佳策略……")

    v3_model = load_safe_policy(
        V3_POLICY_PATH,
        allowed_formats=(
            "safe_policy_state_dict_v3",
        ),
    )

    all_records = []

    print()
    print("开始评价仅紧急保护策略……")

    all_records.extend(
        evaluate_policy(
            policy_name=(
                "Emergency-only baseline"
            ),
            rule_name="continue",
        )
    )

    print()
    print("开始评价随机策略……")

    all_records.extend(
        evaluate_policy(
            policy_name="Random policy",
            rule_name="random",
        )
    )

    print()
    print("开始评价V1最佳DQN……")

    all_records.extend(
        evaluate_policy(
            policy_name="V1 Best DQN",
            model=v1_model,
        )
    )

    print()
    print("开始评价V3最佳DQN……")

    all_records.extend(
        evaluate_policy(
            policy_name="V3 Best DQN",
            model=v3_model,
        )
    )

    episode_data = pd.DataFrame(
        all_records
    )

    summary = summarize_results(
        episode_data
    )

    paired = create_paired_comparison(
        episode_data
    )

    episode_data.to_csv(
        EPISODE_RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    paired.to_csv(
        PAIRED_RESULT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        summary,
        paired,
    )

    print()
    print(
        "逐Episode结果：",
        EPISODE_RESULT_PATH,
    )

    print(
        "汇总结果：",
        SUMMARY_PATH,
    )

    print(
        "V1与V3配对结果：",
        PAIRED_RESULT_PATH,
    )

    v1_environment = v1_model.get_env()

    if v1_environment is not None:
        v1_environment.close()

    v3_environment = v3_model.get_env()

    if v3_environment is not None:
        v3_environment.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())