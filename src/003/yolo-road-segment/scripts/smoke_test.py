from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.common import (
    LOGGER,
    ROOT,
    clean_amp_artifact,
    ensure_project_dirs,
    print_runtime_info,
    runtime_info,
    select_device,
    setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short COCO8-Seg training smoke test.")
    parser.add_argument("--model", default="models/yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", action="store_true")
    return parser

# 定义命令行参数：
#   --model: 要加载的 YOLO 分割模型（默认 models/yolo26n-seg.pt）
#   --epochs: 训练轮数（默认 3）
#   --imgsz: 输入图像尺寸（默认 640）
#   --batch: batch size（默认 2）
#   --device: 计算设备（默认 auto，由 select_device 自动选择 CPU/GPU）
#   --verbose: 是否输出更详细的日志


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)
    ensure_project_dirs()
    model_path = (ROOT / args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run 'python scripts/bootstrap.py' first."
        )
    device = select_device(args.device)
    print_runtime_info(runtime_info(device))
    model = YOLO(str(model_path), task="segment")
    model.train(
        data="coco8-seg.yaml",  # 使用 Ultralytics 内置的 COCO8-Seg 数据集配置
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        project=str((ROOT / "outputs" / "smoke").resolve()),
        name="coco8_seg",
        exist_ok=True,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )
    clean_amp_artifact()
    LOGGER.info("Smoke test complete: %s", model.trainer.save_dir)


if __name__ == "__main__":
    main()
