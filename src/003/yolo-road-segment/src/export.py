# src/export.py -> 导出训练好的模型为 ONNX 或其他格式

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.common import LOGGER, resolve_ultralytics_resource, select_device, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained segmentation model.")
    parser.add_argument(
        "--model",
        default="outputs/train/road_yolo26n_seg/weights/best.pt",
    )
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    model_source = resolve_ultralytics_resource(args.model)
    if not Path(model_source).exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_source}")
    device = select_device(args.device)
    model = YOLO(model_source, task="segment")
    output = model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=device,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    LOGGER.info("Export complete: %s", output)


if __name__ == "__main__":
    main()
