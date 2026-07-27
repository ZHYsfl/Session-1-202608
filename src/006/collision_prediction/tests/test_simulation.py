import math
import random

from geometry import expand_rect, segment_intersects_rect

from simulation import (
    RobotState,
    World,
    distance_to_goal,
    reached_goal,
    reasonable_random_obstacles,
)


def test_future_collision_is_labeled_positive() -> None:
    world = World(300, 200, ((150, 70, 40, 60),))
    state = RobotState(
        x=80,
        y=100,
        heading=0,
        speed=100,
        angular_speed=0,
    )
    assert world.will_collide(
        state,
        radius=15,
        horizon_sec=1,
        step_sec=0.05,
    )


def test_turning_path_can_avoid_obstacle() -> None:
    world = World(300, 200, ((150, 70, 40, 60),))
    state = RobotState(
        x=80,
        y=100,
        heading=0,
        speed=60,
        angular_speed=-math.pi,
    )
    assert not world.will_collide(
        state,
        radius=15,
        horizon_sec=1,
        step_sec=0.02,
    )


def test_goal_distance_and_reached_condition() -> None:
    state = RobotState(10, 10, 0, 0, 0)
    assert distance_to_goal(state, (13, 14)) == 5
    assert reached_goal(state, (13, 14), goal_radius=5)
    assert not reached_goal(state, (13, 14), goal_radius=4.9)


def test_reasonable_random_obstacles_keep_route_clear() -> None:
    route = ((30.0, 30.0), (200.0, 30.0), (270.0, 170.0))
    obstacles = reasonable_random_obstacles(
        rng=random.Random(42),
        width=300,
        height=200,
        count=4,
        route=route,
        min_size=(25, 20),
        max_size=(50, 45),
        boundary_margin=10,
        obstacle_spacing=5,
        route_clearance=15,
        max_attempts=2000,
    )

    assert len(obstacles) == 4
    for obstacle in obstacles:
        protected = expand_rect(obstacle, 15)
        assert not any(
            segment_intersects_rect(start, end, protected)
            for start, end in zip(route[:-1], route[1:], strict=True)
        )
