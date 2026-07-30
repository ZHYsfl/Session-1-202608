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
    resolve_local_path,
    resolve_ultralytics_resource,
    runtime_info,
    select_device,
    setup_logging,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run road segmentation inference.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--model")
    parser.add_argument("--source")
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--device")
    parser.add_argument("--max-det", type=int)
    parser.add_argument("--vid-stride", type=int)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-txt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-conf", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--retina-masks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def normalize_source(value: str) -> str | int:
    if value.isdigit() and not Path(value).exists():
        return int(value)
    if "://" in value:
        return value
    path = resolve_local_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Prediction source does not exist: {path}")
    return str(path)


def predict(args: argparse.Namespace) -> Path:
    setup_logging(args.verbose)
    ensure_project_dirs()
    config = load_yaml(args.config)
    project_cfg = config.get("project", {})
    predict_cfg = config.get("predict", {})

    overrides = {
        "name": args.name,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "max_det": args.max_det,
        "vid_stride": args.vid_stride,
        "save": args.save,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "retina_masks": args.retina_masks,
        "stream": args.stream,
        "exist_ok": args.exist_ok,
    }
    options: dict[str, Any] = merge_not_none(predict_cfg, overrides)
    model_value = args.model or options.pop("model", None) or project_cfg.get("model")
    source_value = args.source or options.pop("source", None)
    if not source_value:
        raise ValueError("Prediction source is required.")
    output_value = args.project or str(
        Path(project_cfg.get("output_dir", "outputs")) / "predict"
    )

    model_source = resolve_ultralytics_resource(model_value)
    if any(sep in str(model_value) for sep in ("/", "\\")) and not Path(model_source).exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_source}")
    source = normalize_source(str(source_value))
    device = select_device(options.pop("device", "auto"))
    info = runtime_info(device)
    print_runtime_info(info)

    project_dir = resolve_local_path(output_value)
    project_dir.mkdir(parents=True, exist_ok=True)
    options.update(
        {
            "source": source,
            "project": str(project_dir),
            "device": device,
            "task": "segment",
            "verbose": True,
        }
    )

    LOGGER.info("Running inference with: %s", model_source)
    LOGGER.info("Source: %s", source)
    model = YOLO(model_source, task="segment")
    results = model.predict(**options)

    count = 0
    first_save_dir: Path | None = None
    for result in results:
        count += 1
        if first_save_dir is None:
            first_save_dir = Path(result.save_dir).resolve()
    if first_save_dir is None:
        first_save_dir = project_dir / str(options.get("name", "predict"))

    summary = {
        "stage": "predict",
        "model": model_source,
        "source": source,
        "options": options,
        "runtime": info,
        "processed_items": count,
        "save_dir": first_save_dir,
    }
    write_json(first_save_dir / "prediction_summary.json", summary)
    LOGGER.info("Inference complete: %s item(s), output=%s", count, first_save_dir)
    return first_save_dir


def main() -> None:
    args = build_parser().parse_args()
    predict(args)


if __name__ == "__main__":
    main()
