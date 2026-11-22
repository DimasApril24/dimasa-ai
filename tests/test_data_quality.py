
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "simulated"
    / "environmental_sensors.csv"
)


@pytest.fixture
def sensor_data():
    """Load the simulated environmental sensor dataset."""
    assert DATASET_PATH.exists(), (
        f"Dataset not found: {DATASET_PATH}"
    )

    return pd.read_csv(DATASET_PATH)


def test_dataset_has_expected_number_of_rows(sensor_data):
    assert len(sensor_data) == 1000


def test_dataset_has_timestamp_column(sensor_data):
    assert "timestamp" in sensor_data.columns


def test_timestamps_are_valid_and_unique(sensor_data):
    timestamps = pd.to_datetime(
        sensor_data["timestamp"],
        errors="coerce"
    )

    assert timestamps.notna().all()
    assert timestamps.is_unique


def test_timestamps_are_chronological(sensor_data):
    timestamps = pd.to_datetime(sensor_data["timestamp"])

    assert timestamps.is_monotonic_increasing


def test_ground_truth_columns_exist(sensor_data):
    expected_columns = [
        "temperature_true",
        "humidity_true",
        "pressure_true",
        "pm25_true",
    ]

    for column in expected_columns:
        assert column in sensor_data.columns


def test_all_sensor_columns_exist(sensor_data):
    variables = [
        "temperature",
        "humidity",
        "pressure",
        "pm25",
    ]

    expected_columns = [
        f"{variable}_sensor_{number}"
        for variable in variables
        for number in range(1, 4)
    ]

    for column in expected_columns:
        assert column in sensor_data.columns


def test_ground_truth_values_are_complete(sensor_data):
    ground_truth_columns = [
        "temperature_true",
        "humidity_true",
        "pressure_true",
        "pm25_true",
    ]

    assert sensor_data[ground_truth_columns].notna().all().all()


def test_ground_truth_values_are_finite(sensor_data):
    ground_truth_columns = [
        "temperature_true",
        "humidity_true",
        "pressure_true",
        "pm25_true",
    ]

    values = sensor_data[ground_truth_columns].to_numpy()

    assert np.isfinite(values).all()


def test_sensor_columns_have_no_infinite_values(sensor_data):
    sensor_columns = [
        column
        for column in sensor_data.columns
        if "_sensor_" in column
    ]

    values = sensor_data[sensor_columns].to_numpy(
        dtype=float
    )

    # NaN is allowed because simulated sensors can have
    # missing readings. Infinite values are not allowed.
    assert not np.isinf(values).any()


def test_dataset_has_twelve_sensor_channels(sensor_data):
    sensor_columns = [
        column
        for column in sensor_data.columns
        if "_sensor_" in column
    ]

    assert len(sensor_columns) == 12
