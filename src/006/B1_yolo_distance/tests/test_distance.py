import pytest

from distance import (
    DistanceEstimator,
    ExponentialSmoother,
    ObjectSpec,
    estimate_focal_length,
)


def test_width_based_distance() -> None:
    estimator = DistanceEstimator(
        fx_px=800,
        fy_px=None,
        object_specs={"car": ObjectSpec(1.8, "width")},
    )
    assert estimator.estimate("car", width_px=360, height_px=200) == pytest.approx(4)


def test_height_based_distance() -> None:
    estimator = DistanceEstimator(
        fx_px=None,
        fy_px=900,
        object_specs={"person": ObjectSpec(1.7, "height")},
    )
    assert estimator.estimate("person", width_px=100, height_px=510) == pytest.approx(3)


def test_missing_calibration_returns_none() -> None:
    estimator = DistanceEstimator(
        fx_px=None,
        fy_px=None,
        object_specs={"car": ObjectSpec(1.8, "width")},
    )
    assert estimator.estimate("car", width_px=300, height_px=200) is None
    assert estimator.estimate("unknown", width_px=300, height_px=200) is None


def test_focal_length_uses_median() -> None:
    measurements = [
        (400, 2, 1),
        (200, 4, 1),
        (500, 2, 1),
    ]
    assert estimate_focal_length(measurements) == 800


def test_exponential_smoothing() -> None:
    smoother = ExponentialSmoother(alpha=0.25)
    assert smoother.update("car:0", 4) == 4
    assert smoother.update("car:0", 8) == 5
