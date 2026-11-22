
import numpy as np
import pandas as pd
import pytest

from src.fusion.sensor_fusion import (
    calculate_sensor_weights,
    weighted_sensor_fusion,
    estimate_sensor_weights,
    calculate_metrics,
)


# ==========================================
# 1. SENSOR WEIGHT TESTS
# ==========================================

def test_sensor_weights_sum_to_one():
    variances = {
        "sensor_1": 1.0,
        "sensor_2": 2.0,
        "sensor_3": 4.0,
    }

    weights = calculate_sensor_weights(variances)

    assert sum(weights.values()) == pytest.approx(1.0)


def test_lower_variance_gets_higher_weight():
    variances = {
        "sensor_1": 1.0,
        "sensor_2": 4.0,
    }

    weights = calculate_sensor_weights(variances)

    assert weights["sensor_1"] > weights["sensor_2"]


def test_empty_variances_raise_error():
    with pytest.raises(ValueError):
        calculate_sensor_weights({})


@pytest.mark.parametrize("invalid_variance", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_variance_raises_error(invalid_variance):
    with pytest.raises(ValueError):
        calculate_sensor_weights({
            "sensor_1": invalid_variance
        })


# ==========================================
# 2. WEIGHTED SENSOR FUSION TESTS
# ==========================================

def test_weighted_fusion_calculates_correct_value():
    values = {
        "sensor_1": 20.0,
        "sensor_2": 30.0,
    }

    weights = {
        "sensor_1": 0.75,
        "sensor_2": 0.25,
    }

    result = weighted_sensor_fusion(values, weights)

    assert result == pytest.approx(22.5)


def test_fusion_renormalizes_weights_when_sensor_missing():
    values = {
        "sensor_1": 20.0,
        "sensor_2": np.nan,
    }

    weights = {
        "sensor_1": 0.75,
        "sensor_2": 0.25,
    }

    result = weighted_sensor_fusion(values, weights)

    assert result == pytest.approx(20.0)


def test_fusion_ignores_unknown_sensor():
    values = {
        "sensor_1": 20.0,
        "sensor_unknown": 1000.0,
    }

    weights = {
        "sensor_1": 1.0,
    }

    result = weighted_sensor_fusion(values, weights)

    assert result == pytest.approx(20.0)


def test_fusion_returns_nan_when_all_readings_missing():
    values = {
        "sensor_1": np.nan,
        "sensor_2": np.nan,
    }

    weights = {
        "sensor_1": 0.5,
        "sensor_2": 0.5,
    }

    result = weighted_sensor_fusion(values, weights)

    assert np.isnan(result)


def test_fusion_ignores_infinite_readings():
    values = {
        "sensor_1": np.inf,
        "sensor_2": 25.0,
    }

    weights = {
        "sensor_1": 0.5,
        "sensor_2": 0.5,
    }

    result = weighted_sensor_fusion(values, weights)

    assert result == pytest.approx(25.0)


# ==========================================
# 3. SENSOR VARIANCE ESTIMATION TESTS
# ==========================================

def test_estimate_sensor_weights_from_calibration_data():
    calibration_data = pd.DataFrame({
        "temperature_true": [10.0, 20.0, 30.0],
        "temperature_sensor_1": [11.0, 19.0, 32.0],
        "temperature_sensor_2": [12.0, 18.0, 34.0],
    })

    weights, variances = estimate_sensor_weights(
        calibration_data,
        "temperature",
        "temperature_true",
    )

    assert "temperature_sensor_1" in weights
    assert "temperature_sensor_2" in weights

    assert sum(weights.values()) == pytest.approx(1.0)

    assert variances["temperature_sensor_1"] == pytest.approx(2.3333333333)
    assert variances["temperature_sensor_2"] == pytest.approx(9.3333333333)


def test_estimation_raises_error_without_valid_variance():
    calibration_data = pd.DataFrame({
        "temperature_true": [10.0, 20.0, 30.0],
        "temperature_sensor_1": [10.0, 20.0, 30.0],
    })

    with pytest.raises(ValueError):
        estimate_sensor_weights(
            calibration_data,
            "temperature",
            "temperature_true",
        )


# ==========================================
# 4. PERFORMANCE METRICS TESTS
# ==========================================

def test_calculate_metrics_returns_correct_values():
    actual = pd.Series([10.0, 20.0, 30.0])
    predicted = pd.Series([12.0, 18.0, 33.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["Valid Samples"] == 3
    assert metrics["MAE"] == pytest.approx(2.3333333333)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(17 / 3))
    assert metrics["Bias"] == pytest.approx(1.0)


def test_metrics_ignore_missing_values():
    actual = pd.Series([10.0, 20.0, np.nan])
    predicted = pd.Series([12.0, np.nan, 30.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["Valid Samples"] == 1
    assert metrics["MAE"] == pytest.approx(2.0)
    assert metrics["RMSE"] == pytest.approx(2.0)
    assert metrics["Bias"] == pytest.approx(2.0)


def test_metrics_return_nan_when_no_valid_samples():
    actual = pd.Series([np.nan, np.nan])
    predicted = pd.Series([np.nan, np.nan])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["Valid Samples"] == 0
    assert np.isnan(metrics["MAE"])
    assert np.isnan(metrics["RMSE"])
    assert np.isnan(metrics["Bias"])
