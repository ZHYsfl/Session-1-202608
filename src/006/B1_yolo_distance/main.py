"""运行 YOLOv10 检测、单目测距、距离告警和结果保存。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import yaml

from detector import Detection, YoloV10Detector
from distance import DistanceEstimator, ExponentialSmoother, ObjectSpec

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置，并确保根节点为键值映射。"""

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def parse_source(value: int | str) -> int | str:
    """将数字解析为摄像头编号，并正确处理两种相对视频路径。"""

    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    path = Path(text)
    if path.is_absolute():
        return str(path)

    # 命令行通常传入仓库根目录相对路径，优先采用当前目录中存在的文件。
    if path.exists():
        return str(path.resolve())

    # config.yaml 中可直接写 data/video.avi，此时按本项目目录解析。
    return str((PROJECT_DIR / path).resolve())


def build_estimator(
    config: dict[str, Any],
    fx_override: float | None = None,
    fy_override: float | None = None,
) -> DistanceEstimator:
    """根据配置构建测距器，并允许命令行临时覆盖焦距。"""

    camera = config["camera"]
    specs = {
        class_name: ObjectSpec(
            real_size_m=float(values["real_size_m"]),
            dimension=values["dimension"],
        )
        for class_name, values in config["objects"].items()
    }
    return DistanceEstimator(
        fx_px=fx_override if fx_override is not None else camera.get("fx_px"),
        fy_px=fy_override if fy_override is not None else camera.get("fy_px"),
        object_specs=specs,
    )


def draw_detection(
    frame: Any,
    detection: Detection,
    distance_m: float | None,
    warning_distance_m: float,
) -> None:
    """绘制检测框、置信度、距离和近距离警告。"""

    x1, y1, x2, y2 = detection.xyxy
    warning = distance_m is not None and distance_m < warning_distance_m
    color = (0, 0, 255) if warning else (0, 200, 0)
    distance_text = "uncalibrated" if distance_m is None else f"{distance_m:.2f} m"
    label = (
        f"{detection.class_name} {detection.confidence:.2f} | {distance_text}"
    )
    if warning:
        label += " | WARNING"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = max(24, y1 - 8)
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def open_writer(path: Path, capture: Any, frame: Any) -> Any:
    """根据输入视频参数创建 MP4 输出对象。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 240:
        fps = 30.0
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {path}")
    return writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", help="camera index, image stream, or video path")
    parser.add_argument(
        "--save",
        nargs="?",
        const="outputs/result.mp4",
        help="save annotated video, optionally to a relative path",
    )
    parser.add_argument(
        "--fx-px",
        type=float,
        help="临时覆盖水平像素焦距，用于按检测框宽度测距",
    )
    parser.add_argument(
        "--fy-px",
        type=float,
        help="临时覆盖垂直像素焦距，用于按检测框高度测距",
    )
    parser.add_argument("--no-display", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    video_config = config["video"]
    weights_path = Path(model_config["weights"])
    if not weights_path.is_absolute():
        # 将自动下载的权重限制在当前 src 项目中，避免污染仓库根目录。
        weights_path = PROJECT_DIR / weights_path

    detector = YoloV10Detector(
        weights=str(weights_path),
        confidence=float(model_config["confidence"]),
        image_size=int(model_config["image_size"]),
        device=model_config.get("device"),
        allowed_classes=config["objects"].keys(),
    )
    estimator = build_estimator(
        config,
        fx_override=args.fx_px,
        fy_override=args.fy_px,
    )
    smoother = ExponentialSmoother(float(config["smoothing"]["alpha"]))
    warning_distance_m = float(config["warning"]["distance_m"])

    source_value = args.source if args.source is not None else video_config["source"]
    source = parse_source(source_value)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video source: {source}")

    display = bool(video_config.get("display", True)) and not args.no_display
    save_value = args.save if args.save is not None else video_config.get("save_path")
    save_path = None
    if save_value:
        candidate = Path(save_value)
        save_path = candidate if candidate.is_absolute() else PROJECT_DIR / candidate

    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # index 仅用于区分同一帧内的同类别目标，供基础平滑器使用。
            # 后续如引入目标跟踪，应改用跟踪器提供的稳定 track_id。
            class_counts: dict[str, int] = {}
            for detection in detector.predict(frame):
                index = class_counts.get(detection.class_name, 0)
                class_counts[detection.class_name] = index + 1
                distance_m = estimator.estimate(
                    detection.class_name,
                    detection.width_px,
                    detection.height_px,
                )
                if distance_m is not None:
                    # 对每个临时目标键分别平滑距离，减少边界框抖动。
                    key = f"{detection.class_name}:{index}"
                    distance_m = smoother.update(key, distance_m)
                draw_detection(
                    frame, detection, distance_m, warning_distance_m
                )

            if save_path is not None:
                # 读取首帧后才能确定输出视频的宽度和高度。
                if writer is None:
                    writer = open_writer(save_path, capture, frame)
                writer.write(frame)

            if display:
                cv2.imshow("YOLOv10 Distance Estimation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # 正常退出和发生异常时都释放摄像头、文件及窗口资源。
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
