from __future__ import annotations

import json
import logging
import math
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("yolo-road-segment")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = resolve_local_path(path, must_exist=True)
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")
    return data


def resolve_local_path(path: str | Path, *, must_exist: bool = False) -> Path:
    value = Path(path).expanduser()
    resolved = value if value.is_absolute() else ROOT / value
    resolved = resolved.resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def resolve_ultralytics_resource(value: str | Path) -> str:
    """Resolve project-local paths while preserving built-in model/data names and URLs."""
    text = str(value)
    if "://" in text:
        return text
    candidate = resolve_local_path(text)
    if candidate.exists() or any(sep in text for sep in ("/", "\\")):
        return str(candidate)
    return text


def resolve_dataset_config(value: str | Path) -> str:
    """Return a built-in dataset name or a generated YAML with an absolute dataset root."""
    text = str(value)
    if "://" in text:
        return text

    candidate = resolve_local_path(text)
    if not candidate.exists():
        if any(sep in text for sep in ("/", "\\")):
            raise FileNotFoundError(f"Dataset YAML does not exist: {candidate}")
        return text

    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        return str(candidate)

    with candidate.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML root must be a mapping: {candidate}")

    root_value = data.get("path")
    if root_value:
        dataset_root = Path(str(root_value)).expanduser()
        if not dataset_root.is_absolute():
            root_candidate = (ROOT / dataset_root).resolve()
            yaml_candidate = (candidate.parent / dataset_root).resolve()
            dataset_root = root_candidate if root_candidate.exists() else yaml_candidate
        data["path"] = str(dataset_root.resolve())

    runtime_dir = ROOT / "outputs" / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = runtime_dir / f"{candidate.stem}.resolved.yaml"
    with runtime_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
    return str(runtime_yaml.resolve())


def ensure_project_dirs() -> None:
    directories = [
        ROOT / "configs",
        ROOT / "models",
        ROOT / "outputs",
        ROOT / "test_images",
        ROOT / "datasets" / "road" / "images" / "train",
        ROOT / "datasets" / "road" / "images" / "val",
        ROOT / "datasets" / "road" / "images" / "test",
        ROOT / "datasets" / "road" / "labels" / "train",
        ROOT / "datasets" / "road" / "labels" / "val",
        ROOT / "datasets" / "road" / "labels" / "test",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def select_device(requested: Any = "auto") -> str | int:
    if requested is None:
        requested = "auto"
    if isinstance(requested, int):
        return requested

    value = str(requested).strip().lower()
    if value not in {"", "auto"}:
        if value.isdigit():
            return int(value)
        return value

    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_info(device: str | int) -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
        "selected_device": device,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = resolve_local_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, ensure_ascii=False, indent=2)
    return output_path


def clean_amp_artifact(root: Path | None = None) -> None:
    """Remove the temporary yolo26n.pt file Ultralytics downloads for AMP checks."""
    target = (root or ROOT) / "yolo26n.pt"
    if target.is_file():
        try:
            target.unlink()
            LOGGER.debug("Removed AMP artifact: %s", target)
        except OSError as exc:
            LOGGER.warning("Could not remove AMP artifact %s: %s", target, exc)


def timestamp_suffix(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return a timestamp string suitable for directory names."""
    return datetime.now().strftime(fmt)


def find_latest_train_dir(
    root: Path | None = None, base_name: str = "road_yolo26n_seg"
) -> Path | None:
    """Find the most recent training directory that contains a best.pt checkpoint."""
    root = root or ROOT
    train_root = root / "outputs" / "train"
    if not train_root.exists():
        return None
    candidates = sorted(
        (p for p in train_root.glob(f"{base_name}_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "weights" / "best.pt").exists():
            return candidate
    return None


def merge_not_none(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def print_runtime_info(info: dict[str, Any]) -> None:
    LOGGER.info(
        "Runtime: Python %s | PyTorch %s | device=%s | GPU=%s",
        info["python"],
        info["torch"],
        info["selected_device"],
        info["gpu"] or "none",
    )
