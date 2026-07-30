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
    seed_everything,
    select_device,
    setup_logging,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO26 road segmentation model.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--name")
    parser.add_argument("--project")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=float)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", metavar="LAST_PT")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def train(args: argparse.Namespace) -> Path:
    setup_logging(args.verbose)
    ensure_project_dirs()
    config = load_yaml(args.config)
    project_cfg = config.get("project", {})
    train_cfg = config.get("train", {})

    overrides = {
        "name": args.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "cache": args.cache,
        "amp": args.amp,
        "deterministic": args.deterministic,
        "plots": args.plots,
        "exist_ok": args.exist_ok,
    }
    options: dict[str, Any] = merge_not_none(train_cfg, overrides)

    data_value = args.data or project_cfg.get("data", "configs/road_seg.yaml")
    model_value = args.model or project_cfg.get("model", "models/yolo26n-seg.pt")
    output_value = args.project or str(
        Path(project_cfg.get("output_dir", "outputs")) / "train"
    )

    device = select_device(options.pop("device", "auto"))
    seed = int(options.get("seed", 42))
    seed_everything(seed)
    info = runtime_info(device)
    print_runtime_info(info)

    if args.resume:
        model_source = resolve_ultralytics_resource(args.resume)
        options["resume"] = True
    else:
        model_source = resolve_ultralytics_resource(model_value)

    data_source = resolve_dataset_config(data_value)
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

    LOGGER.info("Loading model: %s", model_source)
    LOGGER.info("Dataset config: %s", data_source)
    LOGGER.info("Training output: %s/%s", project_dir, options.get("name"))

    model = YOLO(model_source, task="segment")
    model.train(**options)

    save_dir = Path(model.trainer.save_dir).resolve()
    metadata = {
        "stage": "train",
        "model": model_source,
        "data": data_source,
        "options": options,
        "runtime": info,
        "save_dir": save_dir,
    }
    write_json(save_dir / "run_metadata.json", metadata)
    LOGGER.info("Training complete: %s", save_dir)
    LOGGER.info("Best checkpoint: %s", save_dir / "weights" / "best.pt")
    return save_dir


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
