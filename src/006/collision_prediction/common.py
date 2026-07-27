"""项目配置、设备选择和可复现性工具。"""

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
    """读取 YAML 配置并验证根节点类型。"""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("配置文件根节点必须是键值映射")
    return config


def resolve_project_path(value: str) -> Path:
    """将相对路径解析到当前实验目录。"""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def select_device(requested: str) -> torch.device:
    """选择训练设备；auto 优先使用 CUDA。"""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return device


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
