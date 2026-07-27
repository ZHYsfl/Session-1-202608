import math

import pytest

from geometry import (
    circle_intersects_rect,
    expand_rect,
    ray_rect_distance,
    raycast_distance,
    rects_overlap,
    segment_intersects_rect,
)


def test_circle_intersects_rectangle_edge() -> None:
    assert circle_intersects_rect(8, 15, 3, (10, 10, 20, 20))
    assert not circle_intersects_rect(5, 5, 2, (10, 10, 20, 20))


def test_ray_hits_rectangle() -> None:
    distance = ray_rect_distance(0, 15, 1, 0, (10, 10, 20, 20))
    assert distance == pytest.approx(10)


def test_raycast_uses_nearest_surface() -> None:
    distance = raycast_distance(
        20,
        50,
        0,
        [(80, 40, 20, 20)],
        width=200,
        height=100,
        max_range=180,
    )
    assert distance == pytest.approx(60)


def test_raycast_detects_wall() -> None:
    distance = raycast_distance(
        20,
        50,
        math.pi,
        [],
        width=200,
        height=100,
        max_range=180,
    )
    assert distance == pytest.approx(20)


def test_segment_intersection_and_rectangle_spacing() -> None:
    rect = (40, 40, 20, 20)
    assert segment_intersects_rect((0, 50), (100, 50), rect)
    assert not segment_intersects_rect((0, 10), (100, 10), rect)
    assert rects_overlap(rect, (65, 40, 20, 20), spacing=6)
    assert not rects_overlap(rect, (70, 40, 20, 20), spacing=6)
    assert expand_rect(rect, 5) == (35, 35, 30, 30)
