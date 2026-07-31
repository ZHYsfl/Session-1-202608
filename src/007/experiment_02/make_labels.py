from pathlib import Path

import numpy as np
import pandas as pd


# 预测未来多少帧内是否发生碰撞
PREDICTION_HORIZON = 30

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "robot_raw.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "robot_labeled_h30.csv"
)


def add_future_collision_label(
    group: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    对单个运行轮次生成未来碰撞标签。

    对第i行，检查第i+1行到第i+horizon行之间
    是否出现collision_event=1。
    """

    group = group.sort_values(
        "step"
    ).copy()

    collision_values = (
        group["collision_event"]
        .astype(int)
        .to_numpy()
    )

    labels = np.zeros(
        len(group),
        dtype=np.int64,
    )

    for current_index in range(len(group)):
        future_start = current_index + 1

        future_end = min(
            current_index + horizon + 1,
            len(group),
        )

        if future_start < future_end:
            future_events = collision_values[
                future_start:future_end
            ]

            labels[current_index] = int(
                future_events.max() == 1
            )

    group[
        f"future_collision_{horizon}"
    ] = labels

    return group


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到原始数据文件：{INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "run_id",
        "episode_id",
        "step",
        "collision_event",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "原始数据缺少以下字段："
            f"{sorted(missing_columns)}"
        )

    # 保证关键字段是整数
    data["episode_id"] = (
        data["episode_id"].astype(int)
    )

    data["step"] = (
        data["step"].astype(int)
    )

    data["collision_event"] = (
        data["collision_event"].astype(int)
    )

    # 不同程序运行可能从episode 1重新编号，
    # 因此必须同时按照run_id和episode_id分组。
    labeled_groups = []

    for _, episode_data in data.groupby(
        ["run_id", "episode_id"],
        sort=False,
    ):
        labeled_episode = (
            add_future_collision_label(
                episode_data,
                PREDICTION_HORIZON,
            )
        )

        labeled_groups.append(
            labeled_episode
        )

    labeled_data = pd.concat(
        labeled_groups,
        ignore_index=True,
    )

    label_column = (
        f"future_collision_"
        f"{PREDICTION_HORIZON}"
    )

    # 当前已经发生碰撞的事件帧，
    # 不作为神经网络的正常预测样本。
    model_data = labeled_data[
        labeled_data["collision_event"] == 0
    ].copy()

    model_data = model_data.sort_values(
        [
            "run_id",
            "episode_id",
            "step",
        ]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    label_counts = (
        model_data[label_column]
        .value_counts()
        .sort_index()
    )

    positive_count = int(
        (model_data[label_column] == 1).sum()
    )

    total_count = len(model_data)

    positive_ratio = (
        positive_count / total_count
        if total_count > 0
        else 0.0
    )

    episode_count = (
        model_data[
            ["run_id", "episode_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print("=" * 50)
    print("标签生成完成")
    print("=" * 50)
    print(f"原始数据行数：{len(data)}")
    print(f"删除的碰撞事件帧：{len(data) - len(model_data)}")
    print(f"模型数据行数：{total_count}")
    print(f"实验轮次数：{episode_count}")
    print()
    print("标签统计：")
    print(label_counts)
    print()
    print(
        "正样本比例："
        f"{positive_ratio:.2%}"
    )
    print()
    print(f"输出文件：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()