"""单目几何测距与距离平滑模块。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Literal, Mapping

Dimension = Literal["width", "height"]


@dataclass(frozen=True)
class ObjectSpec:
    """目标的已知物理尺寸，用于给单目图像提供真实尺度。"""

    real_size_m: float
    dimension: Dimension

    def __post_init__(self) -> None:
        if self.real_size_m <= 0:
            raise ValueError("real_size_m must be positive")
        if self.dimension not in ("width", "height"):
            raise ValueError("dimension must be 'width' or 'height'")


class DistanceEstimator:
    """使用 D = 焦距像素值 × 真实尺寸 / 成像尺寸 估算目标距离。"""

    def __init__(
        self,
        fx_px: float | None,
        fy_px: float | None,
        object_specs: Mapping[str, ObjectSpec],
    ) -> None:
        self.fx_px = self._validate_optional_focal_length(fx_px, "fx_px")
        self.fy_px = self._validate_optional_focal_length(fy_px, "fy_px")
        self.object_specs = dict(object_specs)

    @staticmethod
    def _validate_optional_focal_length(
        value: float | None, name: str
    ) -> float | None:
        if value is None:
            return None
        value = float(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    def estimate(
        self, class_name: str, width_px: float, height_px: float
    ) -> float | None:
        """返回以米为单位的距离；目标类别或对应轴未标定时返回 None。"""

        spec = self.object_specs.get(class_name)
        if spec is None:
            return None

        # 车辆通常使用真实宽度和框宽，人员通常使用身高和框高。
        if spec.dimension == "width":
            focal_px, pixel_size = self.fx_px, width_px
        else:
            focal_px, pixel_size = self.fy_px, height_px

        # 不使用默认焦距猜测距离，避免显示没有标定依据的数值。
        if focal_px is None:
            return None
        if pixel_size <= 0:
            raise ValueError("bounding-box pixel size must be positive")
        return focal_px * spec.real_size_m / float(pixel_size)


class ExponentialSmoother:
    """为不同目标分别维护指数移动平均，减少检测框抖动。"""

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in the interval (0, 1]")
        self.alpha = alpha
        self._values: dict[str, float] = {}

    def update(self, key: str, value: float) -> float:
        if value < 0:
            raise ValueError("distance cannot be negative")
        previous = self._values.get(key)
        # 首次观测直接采用原值，后续观测再进行指数加权。
        smoothed = (
            value
            if previous is None
            else self.alpha * value + (1 - self.alpha) * previous
        )
        self._values[key] = smoothed
        return smoothed

    def clear(self) -> None:
        self._values.clear()


def estimate_focal_length(
    measurements: Iterable[tuple[float, float, float]],
) -> float:
    """根据（像素尺寸、已知距离、真实尺寸）计算焦距中位数。"""

    estimates: list[float] = []
    for pixel_size_px, known_distance_m, known_size_m in measurements:
        if pixel_size_px <= 0 or known_distance_m <= 0 or known_size_m <= 0:
            raise ValueError("all calibration measurements must be positive")
        # 由 D = fW / w 变换得到 f = wD / W。
        estimates.append(pixel_size_px * known_distance_m / known_size_m)
    if not estimates:
        raise ValueError("at least one calibration measurement is required")
    # 中位数比平均值更不容易受到单次框选异常的影响。
    return float(median(estimates))
