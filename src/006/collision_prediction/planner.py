"""仅用于离线可解性检查和路径效率参考的网格 A*。"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from itertools import count

Rect = tuple[float, float, float, float]
Cell = tuple[int, int]
Point = tuple[float, float]


def _occupied_cells(
    width: float,
    height: float,
    obstacles: Iterable[Rect],
    clearance: float,
    cell_size: float,
) -> tuple[set[Cell], int, int]:
    columns = math.ceil(width / cell_size)
    rows = math.ceil(height / cell_size)
    inflated = tuple(
        (
            x - clearance,
            y - clearance,
            obstacle_width + 2.0 * clearance,
            obstacle_height + 2.0 * clearance,
        )
        for x, y, obstacle_width, obstacle_height in obstacles
    )
    occupied: set[Cell] = set()
    for row in range(rows):
        for column in range(columns):
            x = (column + 0.5) * cell_size
            y = (row + 0.5) * cell_size
            blocked = (
                x <= clearance
                or x >= width - clearance
                or y <= clearance
                or y >= height - clearance
                or any(
                    left <= x <= left + rect_width and top <= y <= top + rect_height
                    for left, top, rect_width, rect_height in inflated
                )
            )
            if blocked:
                occupied.add((column, row))
    return occupied, columns, rows


def astar_path_length(
    width: float,
    height: float,
    obstacles: Iterable[Rect],
    start: Point,
    goal: Point,
    clearance: float,
    *,
    cell_size: float = 10.0,
) -> float | None:
    """返回安全网格上的最短路径长度；无解时返回 None。"""

    occupied, columns, rows = _occupied_cells(
        width, height, obstacles, clearance, cell_size
    )
    start_cell = (int(start[0] // cell_size), int(start[1] // cell_size))
    goal_cell = (int(goal[0] // cell_size), int(goal[1] // cell_size))
    if start_cell in occupied or goal_cell in occupied:
        return None

    queue: list[tuple[float, int, Cell]] = []
    sequence = count()
    heapq.heappush(queue, (0.0, next(sequence), start_cell))
    costs = {start_cell: 0.0}
    while queue:
        _, _, current = heapq.heappop(queue)
        if current == goal_cell:
            return costs[current]
        for dx, dy in (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ):
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < columns and 0 <= neighbor[1] < rows):
                continue
            if neighbor in occupied:
                continue
            if (
                dx
                and dy
                and (
                    (current[0] + dx, current[1]) in occupied
                    or (current[0], current[1] + dy) in occupied
                )
            ):
                continue
            step_cost = cell_size * (math.sqrt(2.0) if dx and dy else 1.0)
            candidate = costs[current] + step_cost
            if candidate >= costs.get(neighbor, float("inf")):
                continue
            costs[neighbor] = candidate
            heuristic = cell_size * math.hypot(
                goal_cell[0] - neighbor[0],
                goal_cell[1] - neighbor[1],
            )
            heapq.heappush(
                queue,
                (candidate + heuristic, next(sequence), neighbor),
            )
    return None
