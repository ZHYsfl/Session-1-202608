"""Ultralytics YOLOv10 检测器封装。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Detection:
    """与深度学习框架无关的单个检测结果。"""

    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]

    @property
    def width_px(self) -> int:
        return max(0, self.xyxy[2] - self.xyxy[0])

    @property
    def height_px(self) -> int:
        return max(0, self.xyxy[3] - self.xyxy[1])


class YoloV10Detector:
    """只加载一次 YOLO 模型，并将输出转换为普通数据类。"""

    def __init__(
        self,
        weights: str,
        confidence: float = 0.4,
        image_size: int = 640,
        device: str | int | None = None,
        allowed_classes: Iterable[str] | None = None,
    ) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: "
                "python -m pip install -r requirements.txt"
            ) from exc

        # 模型在初始化阶段加载，避免视频每一帧重复读取权重。
        self.model = YOLO(weights)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.allowed_classes = set(allowed_classes or [])

    def predict(self, frame: Any) -> list[Detection]:
        """检测一帧图像并返回经过类别过滤的结果。"""

        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        result = self.model.predict(**kwargs)[0]
        if result.boxes is None:
            return []

        # 先将张量移到 CPU，再转换为 Python 数据，隔离 PyTorch 细节。
        names = result.names
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()

        detections: list[Detection] = []
        for box, confidence, class_id in zip(
            boxes, confidences, class_ids, strict=True
        ):
            class_name = str(names[int(class_id)])
            if self.allowed_classes and class_name not in self.allowed_classes:
                continue
            # 坐标取整后供 OpenCV 绘图和像素尺寸计算使用。
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(confidence),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return detections
