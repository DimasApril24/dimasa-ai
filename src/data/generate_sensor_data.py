
"""
DIMASA AI
Multi-Sensor Environmental Data Generator

Tagline: Where Hope Meets Intelligence.

Generates:
1. Environmental ground-truth conditions
2. Multiple noisy sensor observations
3. Different sensor noise levels
4. Independent missing sensor readings
"""

from pathlib import Path

import numpy as np
import pandas as pd


SENSOR_CONFIG = {
    "temperature": {
        "noise": [0.5, 0.8, 1.2],
        "missing": [0.03, 0.05, 0.08],
    },
    "humidity": {
        "noise": [1.0, 2.0, 3.0],
        "missing": [0.04, 0.08, 0.12],
    },
    "pressure": {
        "noise": [0.3, 0.6, 1.0],
        "missing": [0.02, 0.03, 0.07],
    },
    "pm25": {
        "noise": [3.0, 5.0, 7.0],
        "missing": [0.05, 0.10, 0.15],
    },
}


def generate_sensor_data(
    n_samples: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:

    rng = np.random.default_rng(random_seed)

    timestamp = pd.date_range(
        start="2026-01-01",
        periods=n_samples,
        freq="h"
    )

    time = np.arange(n_samples)

    # Ground-truth environmental conditions
    temperature_true = (
        28
        + 4 * np.sin(2 * np.pi * time / 24)
        + rng.normal(0, 0.5, n_samples)
    )

    humidity_true = (
        70
        - 12 * np.sin(2 * np.pi * time / 24)
        + rng.normal(0, 1.5, n_samples)
    )

    pressure_true = (
        1010
        + 3 * np.sin(2 * np.pi * time / 168)
        + rng.normal(0, 0.4, n_samples)
    )

    pm25_true = (
        25
        + 10 * np.sin(2 * np.pi * time / 24 + 1)
        + rng.normal(0, 3, n_samples)
    )

    pm25_true = np.maximum(pm25_true, 2)

    ground_truth = {
        "temperature": temperature_true,
        "humidity": humidity_true,
        "pressure": pressure_true,
        "pm25": pm25_true,
    }

    data = {
        "timestamp": timestamp,
        "temperature_true": temperature_true,
        "humidity_true": humidity_true,
        "pressure_true": pressure_true,
        "pm25_true": pm25_true,
    }

    # Generate three virtual sensors per variable
    for variable, config in SENSOR_CONFIG.items():

        true_values = ground_truth[variable]

        for index, (noise_std, missing_rate) in enumerate(
            zip(config["noise"], config["missing"]),
            start=1
        ):

            sensor_name = f"{variable}_sensor_{index}"

            # Simulate measurement noise
            sensor_values = (
                true_values
                + rng.normal(0, noise_std, n_samples)
            )

            # Simulate missing observations
            missing_mask = rng.random(n_samples) < missing_rate
            sensor_values[missing_mask] = np.nan

            data[sensor_name] = sensor_values

    return pd.DataFrame(data)


def main():

    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / "data" / "simulated"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "environmental_sensors.csv"

    dataset = generate_sensor_data()

    dataset.to_csv(output_file, index=False)

    sensor_columns = [
        column for column in dataset.columns
        if "_sensor_" in column
    ]

    print("=" * 60)
    print("DIMASA AI - MULTI-SENSOR DATA GENERATOR")
    print("=" * 60)

    print(f"Total samples : {len(dataset)}")
    print(f"Total columns : {len(dataset.columns)}")
    print(f"Total sensors : {len(sensor_columns)}")
    print(f"Output file   : {output_file}")

    print("\nMISSING VALUES PER SENSOR")
    print("-" * 60)

    print(dataset[sensor_columns].isna().sum().to_string())

    print("\nMISSING VALUE PERCENTAGE")
    print("-" * 60)

    print(
        (dataset[sensor_columns].isna().mean() * 100)
        .round(2)
        .to_string()
    )

    print("\nDATASET PREVIEW")
    print("-" * 60)
    print(dataset.head().to_string(index=False))

    print("\nDIMASA AI - MULTI-SENSOR DATA GENERATION COMPLETED!")


if __name__ == "__main__":
    main()
