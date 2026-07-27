"""随机挑战地形生成与二维网格 A* 路径规划。"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from itertools import count
from typing import Iterable

from geometry import (
    Rect,
    circle_intersects_rect,
    expand_rect,
    rects_overlap,
    segment_intersects_rect,
)
from simulation import World

GridCell = tuple[int, int]
Point = tuple[float, float]


@dataclass(frozen=True)
class PlannedLayout:
    """一次随机地形及其对应的安全导航路线。"""

    world: World
    route: tuple[Point, ...]
    path_ratio: float
    direct_blockers: int


def _generate_nonoverlapping_obstacles(
    rng: random.Random,
    width: float,
    height: float,
    count_value: int,
    start: Point,
    goal: Point,
    protected_radius: float,
    min_size: Point,
    max_size: Point,
    boundary_margin: float,
    spacing: float,
    max_attempts: int,
) -> tuple[Rect, ...]:
    """生成不重叠且不覆盖起终点的随机矩形障碍物。"""

    obstacles: list[Rect] = []
    for _ in range(max_attempts):
        if len(obstacles) >= count_value:
            break
        obstacle_width = rng.uniform(min_size[0], max_size[0])
        obstacle_height = rng.uniform(min_size[1], max_size[1])
        available_width = width - 2 * boundary_margin - obstacle_width
        available_height = height - 2 * boundary_margin - obstacle_height
        if available_width <= 0 or available_height <= 0:
            raise ValueError("场地尺寸不足以生成障碍物")

        candidate: Rect = (
            rng.uniform(boundary_margin, boundary_margin + available_width),
            rng.uniform(boundary_margin, boundary_margin + available_height),
            obstacle_width,
            obstacle_height,
        )
        if any(
            rects_overlap(candidate, obstacle, spacing)
            for obstacle in obstacles
        ):
            continue
        if circle_intersects_rect(*start, protected_radius, candidate):
            continue
        if circle_intersects_rect(*goal, protected_radius, candidate):
            continue
        obstacles.append(candidate)

    if len(obstacles) != count_value:
        raise RuntimeError("无法生成指定数量的不重叠障碍物")
    return tuple(obstacles)


def _occupied_cells(
    world: World,
    cell_size: float,
    clearance: float,
) -> tuple[set[GridCell], int, int]:
    """将膨胀后的障碍物投影到规则网格。"""

    columns = math.ceil(world.width / cell_size)
    rows = math.ceil(world.height / cell_size)
    inflated = tuple(
        expand_rect(obstacle, clearance)
        for obstacle in world.obstacles
    )
    occupied: set[GridCell] = set()
    for row in range(rows):
        for column in range(columns):
            x = min((column + 0.5) * cell_size, world.width)
            y = min((row + 0.5) * cell_size, world.height)
            if (
                x < clearance
                or x > world.width - clearance
                or y < clearance
                or y > world.height - clearance
                or any(
                    rect[0] <= x <= rect[0] + rect[2]
                    and rect[1] <= y <= rect[1] + rect[3]
                    for rect in inflated
                )
            ):
                occupied.add((column, row))
    return occupied, columns, rows


def _point_to_cell(point: Point, cell_size: float) -> GridCell:
    return (
        max(0, int(point[0] // cell_size)),
        max(0, int(point[1] // cell_size)),
    )


def _cell_to_point(cell: GridCell, cell_size: float) -> Point:
    return (
        (cell[0] + 0.5) * cell_size,
        (cell[1] + 0.5) * cell_size,
    )


def _astar_cells(
    start: GridCell,
    goal: GridCell,
    occupied: set[GridCell],
    columns: int,
    rows: int,
) -> list[GridCell] | None:
    """在八邻域网格上搜索路径，并禁止对角穿过墙角。"""

    if start in occupied or goal in occupied:
        return None
    queue: list[tuple[float, int, GridCell]] = []
    sequence = count()
    heapq.heappush(queue, (0.0, next(sequence), start))
    came_from: dict[GridCell, GridCell] = {}
    cost = {start: 0.0}

    while queue:
        _, _, current = heapq.heappop(queue)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for delta_x, delta_y in (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ):
            neighbor = (current[0] + delta_x, current[1] + delta_y)
            if not (
                0 <= neighbor[0] < columns
                and 0 <= neighbor[1] < rows
            ):
                continue
            if neighbor in occupied:
                continue
            if delta_x and delta_y:
                # 两个正交相邻格均可通行时才允许走对角线。
                if (
                    (current[0] + delta_x, current[1]) in occupied
                    or (current[0], current[1] + delta_y) in occupied
                ):
                    continue

            step_cost = math.sqrt(2) if delta_x and delta_y else 1.0
            new_cost = cost[current] + step_cost
            if new_cost >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = new_cost
            came_from[neighbor] = current
            heuristic = math.hypot(
                goal[0] - neighbor[0],
                goal[1] - neighbor[1],
            )
            heapq.heappush(
                queue,
                (new_cost + heuristic, next(sequence), neighbor),
            )
    return None


def _segment_is_clear(
    start: Point,
    end: Point,
    world: World,
    clearance: float,
) -> bool:
    """检查线段是否位于安全边界内且不穿过膨胀障碍物。"""

    if any(
        coordinate < clearance
        for coordinate in (start[0], start[1], end[0], end[1])
    ):
        return False
    if (
        start[0] > world.width - clearance
        or end[0] > world.width - clearance
        or start[1] > world.height - clearance
        or end[1] > world.height - clearance
    ):
        return False
    return not any(
        segment_intersects_rect(
            start,
            end,
            expand_rect(obstacle, clearance),
        )
        for obstacle in world.obstacles
    )


def _simplify_path(
    points: list[Point],
    world: World,
    clearance: float,
) -> tuple[Point, ...]:
    """使用视线检查删除多余网格点，保留必要转弯点。"""

    simplified = [points[0]]
    current = 0
    while current < len(points) - 1:
        next_index = len(points) - 1
        while next_index > current + 1:
            if _segment_is_clear(
                points[current],
                points[next_index],
                world,
                clearance,
            ):
                break
            next_index -= 1
        simplified.append(points[next_index])
        current = next_index
    return tuple(simplified)


def _route_length(route: Iterable[Point]) -> float:
    points = tuple(route)
    return sum(
        math.hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(points[:-1], points[1:], strict=True)
    )


def plan_route(
    world: World,
    start: Point,
    goal: Point,
    cell_size: float,
    clearance: float,
) -> tuple[Point, ...] | None:
    """在给定世界中规划并简化起点到终点的安全路线。"""

    occupied, columns, rows = _occupied_cells(
        world,
        cell_size,
        clearance,
    )
    cells = _astar_cells(
        _point_to_cell(start, cell_size),
        _point_to_cell(goal, cell_size),
        occupied,
        columns,
        rows,
    )
    if cells is None:
        return None
    points = [
        start,
        *(_cell_to_point(cell, cell_size) for cell in cells[1:-1]),
        goal,
    ]
    return _simplify_path(points, world, clearance)


def generate_challenging_layout(
    rng: random.Random,
    width: float,
    height: float,
    obstacle_count: int,
    start: Point,
    goal: Point,
    robot_radius: float,
    min_size: Point,
    max_size: Point,
    boundary_margin: float,
    obstacle_spacing: float,
    placement_attempts: int,
    layout_attempts: int,
    grid_size: float,
    extra_clearance: float,
    minimum_path_ratio: float,
    minimum_route_points: int,
) -> PlannedLayout:
    """反复生成地形，直到直线路径受阻且 A* 路线足够曲折。"""

    clearance = robot_radius + extra_clearance
    direct_distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
    for _ in range(layout_attempts):
        try:
            obstacles = _generate_nonoverlapping_obstacles(
                rng,
                width,
                height,
                obstacle_count,
                start,
                goal,
                clearance,
                min_size,
                max_size,
                boundary_margin,
                obstacle_spacing,
                placement_attempts,
            )
        except RuntimeError:
            continue
        world = World(width, height, obstacles)
        direct_blockers = sum(
            segment_intersects_rect(
                start,
                goal,
                expand_rect(obstacle, clearance),
            )
            for obstacle in obstacles
        )
        if direct_blockers == 0:
            continue

        route = plan_route(
            world,
            start,
            goal,
            grid_size,
            clearance,
        )
        if route is None:
            continue
        path_ratio = _route_length(route) / direct_distance
        if (
            len(route) >= minimum_route_points
            and path_ratio >= minimum_path_ratio
        ):
            return PlannedLayout(
                world=world,
                route=route,
                path_ratio=path_ratio,
                direct_blockers=direct_blockers,
            )
    raise RuntimeError("无法在限定次数内生成满足挑战条件的可通行地形")


def generate_partitioned_layout(
    rng: random.Random,
    width: float,
    height: float,
    barrier_count: int,
    start: Point,
    goal: Point,
    robot_radius: float,
    gap_size: Point,
    barrier_thickness: Point,
    x_jitter: float,
    layout_attempts: int,
    grid_size: float,
    extra_clearance: float,
    minimum_path_ratio: float,
    minimum_route_points: int,
) -> PlannedLayout:
    """生成带上下交错门洞的贯穿式分区墙，并规划内部路线。"""

    if barrier_count < 1:
        raise ValueError("分区墙数量必须至少为 1")
    clearance = robot_radius + extra_clearance
    direct_distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
    horizontal_step = width / (barrier_count + 1)

    for _ in range(layout_attempts):
        obstacles: list[Rect] = []
        # 随机决定第一道门在上方还是下方，但后续门洞必须交错。
        upper_first = bool(rng.getrandbits(1))
        for index in range(barrier_count):
            center_x = horizontal_step * (index + 1)
            center_x += rng.uniform(-x_jitter, x_jitter)
            thickness = rng.uniform(
                barrier_thickness[0],
                barrier_thickness[1],
            )
            gap_height = rng.uniform(gap_size[0], gap_size[1])
            upper_gap = (index % 2 == 0) == upper_first
            if upper_gap:
                gap_center = rng.uniform(height * 0.22, height * 0.34)
            else:
                gap_center = rng.uniform(height * 0.66, height * 0.78)
            gap_start = max(
                clearance + grid_size,
                gap_center - gap_height / 2,
            )
            gap_end = min(
                height - clearance - grid_size,
                gap_center + gap_height / 2,
            )
            x = center_x - thickness / 2

            # 上下两段都接触场地边界，机器人只能穿过门洞，
            # 无法从顶部或底部绕过整道隔墙。
            obstacles.append((x, 0.0, thickness, gap_start))
            obstacles.append(
                (x, gap_end, thickness, height - gap_end)
            )

        world = World(width, height, tuple(obstacles))
        route = plan_route(
            world,
            start,
            goal,
            grid_size,
            clearance,
        )
        if route is None:
            continue
        path_ratio = _route_length(route) / direct_distance
        direct_blockers = sum(
            segment_intersects_rect(
                start,
                goal,
                expand_rect(obstacle, clearance),
            )
            for obstacle in obstacles
        )
        if (
            direct_blockers >= 1
            and len(route) >= minimum_route_points
            and path_ratio >= minimum_path_ratio
        ):
            return PlannedLayout(
                world=world,
                route=route,
                path_ratio=path_ratio,
                direct_blockers=direct_blockers,
            )
    raise RuntimeError("无法生成满足挑战条件的交错门洞分区地形")
