from controller import WaypointNavigator, waypoint_navigation_control
from simulation import RobotState

ROBOT_CONFIG = {
    "radius": 15,
    "max_speed": 100,
    "max_angular_speed": 2.5,
}
NAVIGATION_CONFIG = {
    "waypoint_radius": 20,
    "cruise_speed_ratio": 0.6,
    "sharp_turn_speed_ratio": 0.3,
}


def test_navigator_targets_first_waypoint() -> None:
    state = RobotState(0, 0, 0, 0, 0)
    route = ((100.0, 0.0), (200.0, 0.0))

    controlled, navigator = waypoint_navigation_control(
        state,
        route,
        ROBOT_CONFIG,
        NAVIGATION_CONFIG,
        WaypointNavigator(),
    )

    assert navigator.index == 0
    assert controlled.speed > 0
    assert controlled.angular_speed == 0


def test_navigator_advances_inside_waypoint_radius() -> None:
    state = RobotState(90, 0, 0, 0, 0)
    route = ((100.0, 0.0), (200.0, 0.0))

    _, navigator = waypoint_navigation_control(
        state,
        route,
        ROBOT_CONFIG,
        NAVIGATION_CONFIG,
        WaypointNavigator(),
    )

    assert navigator.index == 1


def test_navigator_keeps_final_goal_index() -> None:
    state = RobotState(200, 0, 0, 0, 0)
    route = ((100.0, 0.0), (200.0, 0.0))

    _, navigator = waypoint_navigation_control(
        state,
        route,
        ROBOT_CONFIG,
        NAVIGATION_CONFIG,
        WaypointNavigator(index=1),
    )

    assert navigator.index == 1
