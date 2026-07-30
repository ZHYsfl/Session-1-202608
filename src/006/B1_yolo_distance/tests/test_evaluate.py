import pytest

from evaluate import calculate_metrics


def test_metrics() -> None:
    metrics = calculate_metrics([(2.0, 2.2), (4.0, 3.6)])
    assert metrics["count"] == 2
    assert metrics["mae_m"] == pytest.approx(0.3)
    assert metrics["rmse_m"] == pytest.approx(0.3162277)
    assert metrics["mape_percent"] == pytest.approx(10.0)
    assert metrics["bias_m"] == pytest.approx(-0.1)


def test_metrics_reject_zero_ground_truth() -> None:
    with pytest.raises(ValueError):
        calculate_metrics([(0.0, 0.1)])
