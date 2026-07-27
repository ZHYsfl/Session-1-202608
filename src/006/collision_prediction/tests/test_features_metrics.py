import numpy as np
import pytest

from features import build_feature_vector
from metrics import classification_metrics
from simulation import RobotState


def test_feature_normalization() -> None:
    state = RobotState(0, 0, 0, speed=50, angular_speed=-1)
    features = build_feature_vector(
        [0, 50, 150],
        state,
        max_range=100,
        max_speed=100,
        max_angular_speed=2,
    )
    assert np.allclose(features, [0, 0.5, 1, 0.5, -0.5])


def test_classification_metrics() -> None:
    metrics = classification_metrics(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.7, 0.9],
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1)
