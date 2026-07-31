# src/validate.py -> 在验证集上评估模型

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.common import (
    LOGGER,
    ensure_project_dirs,
    find_latest_train_dir,
    load_yaml,
    merge_not_none,
    print_runtime_info,
    resolve_dataset_config,
    resolve_local_path,
    resolve_ultralytics_resource,
    runtime_info,
    select_device,
    setup_logging,
    timestamp_suffix,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a trained road segmentation model.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--split", choices=["val", "test", "train"])
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timestamp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def extract_metrics(metrics: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        result.update(results_dict)

    for family in ("seg", "box"):
        values = getattr(metrics, family, None)
        if values is None:
            continue
        for attr in ("map", "map50", "map75", "mp", "mr", "maps"):
            if hasattr(values, attr):
                result[f"{family}.{attr}"] = getattr(values, attr)
    speed = getattr(metrics, "speed", None)
    if speed is not None:
        result["speed"] = speed
    return result


def validate(args: argparse.Namespace) -> Path:
    setup_logging(args.verbose)
    ensure_project_dirs()
    config = load_yaml(args.config)
    project_cfg = config.get("project", {})
    val_cfg = config.get("val", {})

    overrides = {
        "name": args.name,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "plots": args.plots,
        "save_json": args.save_json,
        "exist_ok": args.exist_ok,
    }
    options = merge_not_none(val_cfg, overrides)

    # 为输出目录添加时间戳，避免多次运行相互覆盖
    if args.timestamp:
        base_name = options.get("name") or "val"
        options["name"] = f"{base_name}_{timestamp_suffix()}"

    model_value = args.model or options.pop("model", None) or project_cfg.get("model")
    data_value = args.data or project_cfg.get("data", "configs/road_seg.yaml")
    output_value = args.project or str(
        Path(project_cfg.get("output_dir", "outputs")) / "val"
    )

    if model_value is not None:
        model_source = resolve_ultralytics_resource(model_value)
        if not Path(model_source).exists() and args.model is None:
            # 配置文件中的路径已失效（例如训练目录已带时间戳），尝试自动定位最新训练结果
            model_value = None
    else:
        model_source = None

    if model_value is None:
        latest = find_latest_train_dir()
        if latest is None:
            raise FileNotFoundError(
                "No model specified and no training output found."
            )
        model_source = str(latest / "weights" / "best.pt")
        LOGGER.info("Auto-resolved latest training checkpoint: %s", model_source)

    data_source = resolve_dataset_config(data_value)
    if not Path(model_source).exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_source}")

    device = select_device(options.pop("device", "auto"))
    info = runtime_info(device)
    print_runtime_info(info)
    project_dir = resolve_local_path(output_value)
    project_dir.mkdir(parents=True, exist_ok=True)
    options.update(
        {
            "data": data_source,
            "project": str(project_dir),
            "device": device,
            "task": "segment",
            "verbose": True,
        }
    )

    LOGGER.info("Validating model: %s", model_source)
    model = YOLO(model_source, task="segment")
    metrics = model.val(**options)

    # Ultralytics 不保证 model.validator 在 val() 返回后仍然存在，
    # 优先从 metrics 取 save_dir，再回退到 model.validator 或按选项构造路径。
    save_dir: Path | None = None
    if hasattr(metrics, "save_dir") and metrics.save_dir:
        save_dir = Path(str(metrics.save_dir)).resolve()
    elif hasattr(model, "validator") and model.validator is not None:
        save_dir = Path(str(model.validator.save_dir)).resolve()

    if save_dir is None:
        save_dir = (
            Path(options.get("project", "outputs/val"))
            / str(options.get("name", "val"))
        ).resolve()

    summary = {
        "stage": "val",
        "model": model_source,
        "data": data_source,
        "options": options,
        "runtime": info,
        "metrics": extract_metrics(metrics),
        "save_dir": save_dir,
    }
    write_json(save_dir / "metrics_summary.json", summary)
    LOGGER.info("Validation complete: %s", save_dir)
    return save_dir


def main() -> None:
    args = build_parser().parse_args()
    validate(args)


if __name__ == "__main__":
    main()
