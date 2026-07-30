"""B3 Cart-Pole 实验的配置、路径和随机种子工具。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("配置文件根节点必须是映射")
    return config


def resolve_project_path(value: str) -> Path:
    """把配置中的相对路径解析到 B3 项目目录。"""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """auto 优先使用 CUDA，也允许显式选择 cpu。"""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA")
    return device
