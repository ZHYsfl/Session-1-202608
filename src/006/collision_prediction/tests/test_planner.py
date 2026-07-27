import random

from geometry import expand_rect, segment_intersects_rect
from planner import (
    generate_challenging_layout,
    generate_partitioned_layout,
    plan_route,
)
from simulation import World


def test_astar_routes_around_blocking_obstacle() -> None:
    world = World(300, 200, ((130, 40, 40, 120),))
    route = plan_route(
        world,
        start=(30, 100),
        goal=(270, 100),
        cell_size=10,
        clearance=15,
    )

    assert route is not None
    assert len(route) >= 3
    inflated = expand_rect(world.obstacles[0], 15)
    assert all(
        not segment_intersects_rect(start, end, inflated)
        for start, end in zip(route[:-1], route[1:], strict=True)
    )


def test_random_layout_is_blocked_but_reachable() -> None:
    layout = generate_challenging_layout(
        rng=random.Random(42),
        width=500,
        height=320,
        obstacle_count=6,
        start=(35, 35),
        goal=(465, 285),
        robot_radius=12,
        min_size=(35, 30),
        max_size=(80, 70),
        boundary_margin=15,
        obstacle_spacing=8,
        placement_attempts=1500,
        layout_attempts=100,
        grid_size=10,
        extra_clearance=10,
        minimum_path_ratio=1.05,
        minimum_route_points=3,
    )

    assert layout.direct_blockers >= 1
    assert layout.path_ratio >= 1.05
    assert len(layout.route) >= 3


def test_partitioned_layout_forces_zigzag_route() -> None:
    layout = generate_partitioned_layout(
        rng=random.Random(42),
        width=600,
        height=400,
        barrier_count=2,
        start=(40, 40),
        goal=(560, 360),
        robot_radius=12,
        gap_size=(120, 145),
        barrier_thickness=(24, 32),
        x_jitter=15,
        layout_attempts=50,
        grid_size=10,
        extra_clearance=12,
        minimum_path_ratio=1.15,
        minimum_route_points=4,
    )

    assert len(layout.world.obstacles) == 4
    assert layout.path_ratio >= 1.15
    assert layout.direct_blockers >= 1
