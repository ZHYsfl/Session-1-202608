"""离线路径 Oracle 测试。"""

from planner import astar_path_length


def test_astar_reference_routes_around_rectangle() -> None:
    length = astar_path_length(
        width=300,
        height=200,
        obstacles=((135, 60, 30, 80),),
        start=(40, 100),
        goal=(260, 100),
        clearance=12,
        cell_size=10,
    )
    assert length is not None
    assert length > 220


def test_astar_returns_none_for_sealed_wall() -> None:
    length = astar_path_length(
        width=300,
        height=200,
        obstacles=((140, 0, 20, 200),),
        start=(40, 100),
        goal=(260, 100),
        clearance=10,
        cell_size=10,
    )
    assert length is None
