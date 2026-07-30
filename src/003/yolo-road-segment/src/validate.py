from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.common import (
    LOGGER,
    ensure_project_dirs,
    load_yaml,
    merge_not_none,
    print_runtime_info,
    resolve_dataset_config,
    resolve_local_path,
    resolve_ultralytics_resource,
    runtime_info,
    select_device,
    setup_logging,
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
    model_value = args.model or options.pop("model", None) or project_cfg.get("model")
    data_value = args.data or project_cfg.get("data", "configs/road_seg.yaml")
    output_value = args.project or str(
        Path(project_cfg.get("output_dir", "outputs")) / "val"
    )

    model_source = resolve_ultralytics_resource(model_value)
    data_source = resolve_dataset_config(data_value)
    if any(sep in str(model_value) for sep in ("/", "\\")) and not Path(model_source).exists():
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
    save_dir = Path(model.validator.save_dir).resolve()
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
