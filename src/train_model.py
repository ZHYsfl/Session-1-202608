import copy
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ==================================================
# 训练配置
# ==================================================
SEED = 2026

LABEL = "future_collision_30"

FEATURES = [
    "sensor_0",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "linear_speed",
    "angular_speed",
]

BATCH_SIZE = 64
MAX_EPOCHS = 200
LEARNING_RATE = 0.001
PATIENCE = 25


# ==================================================
# 文件路径
# ==================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "robot_labeled_h30.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "collision_mlp_h30.pt"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "metrics.json"
)

LOSS_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "loss_curve.png"
)

CONFUSION_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "confusion_matrix.png"
)

SPLIT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


# ==================================================
# 多层神经网络
# 11维输入 → 64 → 32 → 1
# ==================================================
class CollisionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs)


def set_random_seed() -> None:
    """固定随机种子。"""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def contains_two_classes(
    data: pd.DataFrame,
) -> bool:
    """判断数据是否同时包含标签0和标签1。"""

    return data[LABEL].nunique() == 2


def split_by_episode(
    data: pd.DataFrame,
):
    """
    按完整Episode划分数据：

    70%训练集
    15%验证集
    15%测试集
    """

    data = data.copy()

    # 同一次程序运行中的同一个episode视为一个分组
    data["_group"] = (
        data["run_id"].astype(str)
        + "__"
        + data["episode_id"].astype(str)
    )

    # 尝试多个随机划分，保证三个数据集都有0和1
    for attempt in range(100):
        random_state = SEED + attempt

        # 第一次：70%训练，30%临时数据
        first_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.30,
            random_state=random_state,
        )

        train_index, temporary_index = next(
            first_split.split(
                data,
                groups=data["_group"],
            )
        )

        train_data = data.iloc[
            train_index
        ].copy()

        temporary_data = data.iloc[
            temporary_index
        ].copy()

        # 第二次：临时数据平均分成验证和测试
        second_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=random_state + 1000,
        )

        valid_index, test_index = next(
            second_split.split(
                temporary_data,
                groups=temporary_data["_group"],
            )
        )

        valid_data = temporary_data.iloc[
            valid_index
        ].copy()

        test_data = temporary_data.iloc[
            test_index
        ].copy()

        if (
            contains_two_classes(train_data)
            and contains_two_classes(valid_data)
            and contains_two_classes(test_data)
        ):
            for part in (
                train_data,
                valid_data,
                test_data,
            ):
                part.drop(
                    columns="_group",
                    inplace=True,
                )

                part.sort_values(
                    [
                        "run_id",
                        "episode_id",
                        "step",
                    ],
                    inplace=True,
                )

                part.reset_index(
                    drop=True,
                    inplace=True,
                )

            return (
                train_data,
                valid_data,
                test_data,
            )

    raise RuntimeError(
        "无法划分出同时包含标签0和标签1的"
        "训练集、验证集和测试集。"
    )


def make_data_loader(
    features,
    labels,
    shuffle,
):
    """建立PyTorch数据加载器。"""

    dataset = TensorDataset(
        torch.tensor(
            features,
            dtype=torch.float32,
        ),
        torch.tensor(
            labels.reshape(-1, 1),
            dtype=torch.float32,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def calculate_average_loss(
    model,
    data_loader,
    loss_function,
    device,
):
    """计算整个数据集的平均损失。"""

    model.eval()

    total_loss = 0.0
    total_number = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)

            loss = loss_function(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * len(features)
            )

            total_number += len(features)

    return total_loss / total_number


def predict_probabilities(
    model,
    data_loader,
    device,
):
    """返回真实标签和预测碰撞概率。"""

    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)

            logits = model(features)

            probabilities = torch.sigmoid(
                logits
            )

            all_labels.append(
                labels.numpy().reshape(-1)
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
                .reshape(-1)
            )

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
    )


def choose_best_threshold(
    labels,
    probabilities,
):
    """
    在验证集上寻找F1最高的分类阈值。
    """

    selected_threshold = 0.50
    highest_f1 = -1.0

    for threshold in np.arange(
        0.05,
        0.96,
        0.01,
    ):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        current_f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        if current_f1 > highest_f1:
            highest_f1 = current_f1
            selected_threshold = float(
                threshold
            )

    return selected_threshold


def calculate_metrics(
    labels,
    probabilities,
    threshold,
):
    """计算测试集指标。"""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),

        "confusion_matrix": (
            matrix.tolist()
        ),
    }


def print_dataset_information(
    name,
    data,
):
    """显示数据划分结果。"""

    episode_number = (
        data[
            ["run_id", "episode_id"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    label_counts = (
        data[LABEL]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        f"{name}: "
        f"行数={len(data)}, "
        f"Episode={episode_number}, "
        f"标签={label_counts}, "
        f"正样本比例="
        f"{data[LABEL].mean():.2%}"
    )


def main():
    set_random_seed()

    # ==================================================
    # 1. 读取并检查数据
    # ==================================================
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = set(
        FEATURES
        + [
            "run_id",
            "episode_id",
            "step",
            LABEL,
        ]
    )

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "数据缺少字段："
            f"{sorted(missing_columns)}"
        )

    data[LABEL] = (
        data[LABEL].astype(int)
    )

    if (
        data[
            FEATURES + [LABEL]
        ]
        .isnull()
        .any()
        .any()
    ):
        raise ValueError(
            "训练数据中存在空值。"
        )

    # ==================================================
    # 2. 按Episode划分
    # ==================================================
    (
        train_data,
        valid_data,
        test_data,
    ) = split_by_episode(data)

    print("=" * 70)

    print_dataset_information(
        "训练集",
        train_data,
    )

    print_dataset_information(
        "验证集",
        valid_data,
    )

    print_dataset_information(
        "测试集",
        test_data,
    )

    print("=" * 70)

    SPLIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_data.to_csv(
        SPLIT_DIRECTORY / "train.csv",
        index=False,
        encoding="utf-8-sig",
    )

    valid_data.to_csv(
        SPLIT_DIRECTORY / "valid.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_data.to_csv(
        SPLIT_DIRECTORY / "test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ==================================================
    # 3. 标准化输入
    # ==================================================
    scaler = StandardScaler()

    # 只允许训练集参与拟合
    train_features = scaler.fit_transform(
        train_data[
            FEATURES
        ].to_numpy(
            dtype=np.float32
        )
    )

    valid_features = scaler.transform(
        valid_data[
            FEATURES
        ].to_numpy(
            dtype=np.float32
        )
    )

    test_features = scaler.transform(
        test_data[
            FEATURES
        ].to_numpy(
            dtype=np.float32
        )
    )

    train_labels = train_data[
        LABEL
    ].to_numpy(
        dtype=np.float32
    )

    valid_labels = valid_data[
        LABEL
    ].to_numpy(
        dtype=np.float32
    )

    test_labels = test_data[
        LABEL
    ].to_numpy(
        dtype=np.float32
    )

    # ==================================================
    # 4. 建立数据加载器
    # ==================================================
    train_loader = make_data_loader(
        train_features,
        train_labels,
        True,
    )

    train_evaluation_loader = (
        make_data_loader(
            train_features,
            train_labels,
            False,
        )
    )

    valid_loader = make_data_loader(
        valid_features,
        valid_labels,
        False,
    )

    test_loader = make_data_loader(
        test_features,
        test_labels,
        False,
    )

    # ==================================================
    # 5. 建立模型
    # ==================================================
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = CollisionMLP().to(
        device
    )

    positive_number = float(
        train_labels.sum()
    )

    negative_number = float(
        len(train_labels)
        - positive_number
    )

    if positive_number == 0:
        raise ValueError(
            "训练集中没有标签1。"
        )

    positive_weight = (
        negative_number
        / positive_number
    )

    loss_function = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [positive_weight],
                dtype=torch.float32,
                device=device,
            )
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    print(
        f"训练设备：{device}"
    )

    print(
        "正类权重："
        f"{positive_weight:.4f}"
    )

    # ==================================================
    # 6. 训练与早停
    # ==================================================
    train_losses = []
    valid_losses = []

    best_valid_loss = float(
        "inf"
    )

    best_model_state = None
    best_epoch = 0
    no_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        for (
            batch_features,
            batch_labels,
        ) in train_loader:
            batch_features = (
                batch_features.to(device)
            )

            batch_labels = (
                batch_labels.to(device)
            )

            optimizer.zero_grad()

            logits = model(
                batch_features
            )

            loss = loss_function(
                logits,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

        train_loss = (
            calculate_average_loss(
                model,
                train_evaluation_loader,
                loss_function,
                device,
            )
        )

        valid_loss = (
            calculate_average_loss(
                model,
                valid_loader,
                loss_function,
                device,
            )
        )

        train_losses.append(
            train_loss
        )

        valid_losses.append(
            valid_loss
        )

        if (
            valid_loss
            < best_valid_loss - 1e-6
        ):
            best_valid_loss = (
                valid_loss
            )

            best_epoch = epoch

            best_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            no_improvement = 0

        else:
            no_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss="
                f"{train_loss:.6f} | "
                f"valid_loss="
                f"{valid_loss:.6f}"
            )

        if (
            no_improvement
            >= PATIENCE
        ):
            print(
                "验证损失连续多轮"
                "未改善，提前停止。"
            )
            break

    if best_model_state is None:
        raise RuntimeError(
            "没有保存到有效模型。"
        )

    model.load_state_dict(
        best_model_state
    )

    # ==================================================
    # 7. 使用验证集确定阈值
    # ==================================================
    (
        valid_true,
        valid_probabilities,
    ) = predict_probabilities(
        model,
        valid_loader,
        device,
    )

    threshold = choose_best_threshold(
        valid_true,
        valid_probabilities,
    )

    # ==================================================
    # 8. 测试集评价
    # ==================================================
    (
        test_true,
        test_probabilities,
    ) = predict_probabilities(
        model,
        test_loader,
        device,
    )

    metrics = calculate_metrics(
        test_true,
        test_probabilities,
        threshold,
    )

    print()
    print("=" * 70)
    print(
        f"最佳Epoch：{best_epoch}"
    )
    print(
        f"分类阈值：{threshold:.2f}"
    )
    print(
        "Accuracy："
        f"{metrics['accuracy']:.4f}"
    )
    print(
        "Precision："
        f"{metrics['precision']:.4f}"
    )
    print(
        "Recall："
        f"{metrics['recall']:.4f}"
    )
    print(
        "F1："
        f"{metrics['f1']:.4f}"
    )
    print(
        "ROC-AUC："
        f"{metrics['roc_auc']:.4f}"
    )
    print(
        "混淆矩阵 "
        "[[TN, FP], [FN, TP]]："
    )
    print(
        np.asarray(
            metrics[
                "confusion_matrix"
            ]
        )
    )
    print("=" * 70)

    # ==================================================
    # 9. 保存模型与指标
    # ==================================================
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOSS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": (
            best_model_state
        ),

        "features": FEATURES,

        "label": LABEL,

        "input_size": 11,

        "hidden_sizes": [
            64,
            32,
        ],

        "threshold": threshold,

        "scaler_mean": (
            scaler.mean_.tolist()
        ),

        "scaler_scale": (
            scaler.scale_.tolist()
        ),

        "best_epoch": best_epoch,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ==================================================
    # 10. 绘制损失曲线
    # ==================================================
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        train_losses,
        label="Training loss",
    )

    plt.plot(
        valid_losses,
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        LOSS_PATH,
        dpi=200,
    )

    plt.close()

    # ==================================================
    # 11. 绘制混淆矩阵
    # ==================================================
    matrix = np.asarray(
        metrics[
            "confusion_matrix"
        ]
    )

    plt.figure(
        figsize=(5.5, 5)
    )

    plt.imshow(matrix)

    plt.xlabel(
        "Predicted label"
    )

    plt.ylabel(
        "True label"
    )

    plt.xticks(
        [0, 1],
        ["0", "1"],
    )

    plt.yticks(
        [0, 1],
        ["0", "1"],
    )

    plt.title(
        "Test confusion matrix"
    )

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        CONFUSION_PATH,
        dpi=200,
    )

    plt.close()

    print(
        f"模型：{MODEL_PATH}"
    )

    print(
        f"指标：{METRICS_PATH}"
    )

    print(
        f"损失曲线：{LOSS_PATH}"
    )

    print(
        f"混淆矩阵："
        f"{CONFUSION_PATH}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())