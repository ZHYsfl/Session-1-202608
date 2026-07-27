"""二维碰撞检测和射线测距几何函数。"""

from __future__ import annotations

import math
from typing import Iterable

Rect = tuple[float, float, float, float]


def expand_rect(rect: Rect, amount: float) -> Rect:
    """向四周扩展矩形，用于构造机器人路线安全走廊。"""

    x, y, width, height = rect
    return (
        x - amount,
        y - amount,
        width + 2 * amount,
        height + 2 * amount,
    )


def rects_overlap(first: Rect, second: Rect, spacing: float = 0.0) -> bool:
    """判断两个矩形是否重叠或小于指定间距。"""

    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return not (
        first_x + first_width + spacing <= second_x
        or second_x + second_width + spacing <= first_x
        or first_y + first_height + spacing <= second_y
        or second_y + second_height + spacing <= first_y
    )


def segment_intersects_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: Rect,
) -> bool:
    """使用参数裁剪判断线段是否经过轴对齐矩形。"""

    rect_x, rect_y, rect_width, rect_height = rect
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    minimum = 0.0
    maximum = 1.0

    for origin, delta, lower, upper in (
        (start[0], delta_x, rect_x, rect_x + rect_width),
        (start[1], delta_y, rect_y, rect_y + rect_height),
    ):
        if abs(delta) < 1e-12:
            if origin < lower or origin > upper:
                return False
            continue

        first = (lower - origin) / delta
        second = (upper - origin) / delta
        if first > second:
            first, second = second, first
        minimum = max(minimum, first)
        maximum = min(maximum, second)
        if maximum < minimum:
            return False
    return True


def circle_intersects_rect(
    center_x: float,
    center_y: float,
    radius: float,
    rect: Rect,
) -> bool:
    """判断圆形机器人是否与轴对齐矩形相交。"""

    rect_x, rect_y, rect_width, rect_height = rect
    closest_x = min(max(center_x, rect_x), rect_x + rect_width)
    closest_y = min(max(center_y, rect_y), rect_y + rect_height)
    delta_x = center_x - closest_x
    delta_y = center_y - closest_y
    return delta_x * delta_x + delta_y * delta_y <= radius * radius


def circle_outside_bounds(
    center_x: float,
    center_y: float,
    radius: float,
    width: float,
    height: float,
) -> bool:
    """判断机器人圆周是否越过场地边界。"""

    return (
        center_x - radius <= 0
        or center_x + radius >= width
        or center_y - radius <= 0
        or center_y + radius >= height
    )


def ray_rect_distance(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    rect: Rect,
) -> float | None:
    """使用 slab 算法计算射线到矩形的首次交点距离。"""

    rect_x, rect_y, rect_width, rect_height = rect
    minimum = -math.inf
    maximum = math.inf

    for origin, direction, lower, upper in (
        (origin_x, direction_x, rect_x, rect_x + rect_width),
        (origin_y, direction_y, rect_y, rect_y + rect_height),
    ):
        if abs(direction) < 1e-12:
            if origin < lower or origin > upper:
                return None
            continue

        first = (lower - origin) / direction
        second = (upper - origin) / direction
        if first > second:
            first, second = second, first
        minimum = max(minimum, first)
        maximum = min(maximum, second)
        if maximum < minimum:
            return None

    if maximum < 0:
        return None
    return max(0.0, minimum)


def raycast_distance(
    origin_x: float,
    origin_y: float,
    angle_rad: float,
    obstacles: Iterable[Rect],
    width: float,
    height: float,
    max_range: float,
) -> float:
    """返回射线到最近障碍物或墙壁的距离，并截断到最大量程。"""

    direction_x = math.cos(angle_rad)
    direction_y = math.sin(angle_rad)
    candidates = [max_range]

    # 场地墙壁相当于包围场景的矩形边界。
    if direction_x > 1e-12:
        candidates.append((width - origin_x) / direction_x)
    elif direction_x < -1e-12:
        candidates.append((0 - origin_x) / direction_x)
    if direction_y > 1e-12:
        candidates.append((height - origin_y) / direction_y)
    elif direction_y < -1e-12:
        candidates.append((0 - origin_y) / direction_y)

    for obstacle in obstacles:
        distance = ray_rect_distance(
            origin_x,
            origin_y,
            direction_x,
            direction_y,
            obstacle,
        )
        if distance is not None and distance >= 0:
            candidates.append(distance)

    return max(0.0, min(candidates))
