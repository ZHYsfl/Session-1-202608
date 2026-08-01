# -*- coding: utf-8 -*-
"""
models.py — 009 model.py 加载桥（进程内 import，api.md §1/§8）

009 交付 model.py 后，003 从此处统一获得网络结构与权重存取接口，
调用方（sac.py / train.py）不需要感知模型来自哪里。

加载优先级（自上而下第一个成功即用）：
  1. 环境变量 MODEL_MODULE=<path>：显式指定 009 交付的 model.py 路径
  2. 当前目录/上级目录存在 model.py（如把 009 交付文件拷进 003 根目录）
  3. 回退到 tools/model_stub.py 内置桩（开发期，接口与 api_doc.md §7 完全一致）

加载后校验 OBS_DIM / ACT_DIM 与协议一致（api.md §6.1）。
"""

import importlib.util
import os
import sys
from pathlib import Path

from config import ACT_DIM, OBS_DIM

_STUB_PATH = Path(__file__).parent / "tools" / "model_stub.py"

_model_module = None
_model_source = None  # 调试用：实际加载的模块路径


def _import_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("a2_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_model_module():
    """返回 009 兼容的 model 模块（惰性加载，幂等）。"""
    global _model_module, _model_source
    if _model_module is not None:
        return _model_module

    env = os.environ.get("MODEL_MODULE", "")
    candidates = []
    if env:  # 1) 显式指定
        candidates.append(("env:MODEL_MODULE", Path(env)))
    for d in (Path(__file__).parent, Path(__file__).parent.parent):  # 2) 就近查找
        p = d / "model.py"
        if p.exists():
            candidates.append(("本地 model.py", p))
            break
    candidates.append(("内置桩", _STUB_PATH))  # 3) 回退桩

    errors = []
    for label, path in candidates:
        if not path.exists():
            errors.append(f"[{label}] 文件不存在: {path}")
            continue
        try:
            _model_module = _import_from_path(path)
            _model_source = f"{label}: {path}"
            break
        except Exception as e:  # noqa: BLE001 — 某个候选失败就尝试下一个
            errors.append(f"[{label}] 加载失败: {e!r}")
    if _model_module is None:
        raise RuntimeError("无法加载 009 model 模块:\n" + "\n".join(errors))

    # 协议校验（api.md §6.1）：观测/动作维度必须与协议锁定值一致
    if _model_module.OBS_DIM != OBS_DIM:
        raise RuntimeError(
            f"模型 OBS_DIM={_model_module.OBS_DIM} 与协议 {OBS_DIM} 不一致"
            f"（{_model_source}），拒绝启动")
    if _model_module.ACT_DIM != ACT_DIM:
        raise RuntimeError(
            f"模型 ACT_DIM={_model_module.ACT_DIM} 与协议 {ACT_DIM} 不一致"
            f"（{_model_source}），拒绝启动")
    return _model_module


def model_source() -> str:
    """返回实际加载来源（日志用）。"""
    get_model_module()
    return _model_source
