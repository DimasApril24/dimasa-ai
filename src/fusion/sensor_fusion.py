
"""
DIMASA AI
Multi-Sensor Data Fusion Engine

Method:
Inverse-Variance Weighted Fusion (IVWF)

Workflow:
1. Load environmental sensor dataset.
2. Split data chronologically into 60:20:20.
3. Estimate sensor error variance using calibration data only.
4. Calculate normalized inverse-variance sensor weights.
5. Perform multi-sensor data fusion using fixed weights.
6. Evaluate fusion performance on the independent test set.
7. Export fusion results, weights, and performance reports.

Data policy:
- Calibration: first 60% of observations.
- Validation: next 20% of observations.
- Testing: final 20% of observations.
- Validation and test ground truth are never used to estimate weights.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

VARIABLES = {
    "temperature": "temperature_true",
    "humidity": "humidity_true",
    "pressure": "pressure_true",
    "pm25": "pm25_true",
}

CALIBRATION_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

RANDOMNESS_POLICY = "Chronological split; no random shuffling"


# =========================================================
# SENSOR WEIGHT CALCULATION
# =========================================================

def calculate_sensor_weights(error_variances: dict) -> dict:
    """
    Calculate normalized inverse-variance sensor weights.

    Lower error variance produces a higher reliability weight.
    """

    if not error_variances:
        raise ValueError("Sensor variance cannot be empty.")

    for sensor, variance in error_variances.items():
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError(
                f"Invalid variance for {sensor}: {variance}"
            )

    inverse_variances = {
        sensor: 1.0 / variance
        for sensor, variance in error_variances.items()
    }

    total_weight = sum(inverse_variances.values())

    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("Total inverse variance must be positive.")

    return {
        sensor: weight / total_weight
        for sensor, weight in inverse_variances.items()
    }


# =========================================================
# WEIGHTED SENSOR FUSION
# =========================================================

def weighted_sensor_fusion(
    sensor_values: dict,
    sensor_weights: dict,
) -> float:
    """
    Fuse available sensor readings.

    Missing or non-finite readings are ignored.
    Available weights are renormalized.
    Returns NaN if all readings are unavailable.
    """

    valid_readings = {}

    for sensor, value in sensor_values.items():

        if sensor not in sensor_weights:
            continue

        if pd.isna(value) or not np.isfinite(value):
            continue

        valid_readings[sensor] = float(value)

    if not valid_readings:
        return np.nan

    available_weight = sum(
        sensor_weights[sensor]
        for sensor in valid_readings
    )

    if available_weight <= 0:
        return np.nan

    fused_value = sum(
        value * sensor_weights[sensor]
        for sensor, value in valid_readings.items()
    ) / available_weight

    return float(fused_value)


# =========================================================
# SENSOR WEIGHT ESTIMATION
# =========================================================

def estimate_sensor_weights(
    calibration_df: pd.DataFrame,
    variable: str,
    true_column: str,
) -> tuple:
    """
    Estimate sensor error variance using calibration data only.
    """

    sensor_columns = [
        column
        for column in calibration_df.columns
        if column.startswith(f"{variable}_sensor_")
    ]

    if not sensor_columns:
        raise ValueError(
            f"No sensor columns found for {variable}."
        )

    if true_column not in calibration_df.columns:
        raise ValueError(
            f"Ground truth column not found: {true_column}"
        )

    error_variances = {}

    for sensor in sensor_columns:

        valid_data = calibration_df[
            [true_column, sensor]
        ].dropna()

        if len(valid_data) < 2:
            continue

        errors = (
            valid_data[sensor] - valid_data[true_column]
        )

        variance = errors.var(ddof=1)

        if np.isfinite(variance) and variance > 0:
            error_variances[sensor] = float(variance)

    if not error_variances:
        raise ValueError(
            f"No valid sensor variance for {variable}."
        )

    weights = calculate_sensor_weights(error_variances)

    return weights, error_variances


# =========================================================
# METRIC CALCULATION
# =========================================================

def calculate_metrics(
    true_values: pd.Series,
    predicted_values: pd.Series,
) -> dict:
    """
    Calculate MAE, RMSE, Bias, and valid sample count.
    """

    valid_data = pd.DataFrame({
        "true": true_values,
        "predicted": predicted_values,
    })

    valid_data = valid_data.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if valid_data.empty:
        return {
            "Valid Samples": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
        }

    errors = (
        valid_data["predicted"] - valid_data["true"]
    ).to_numpy()

    return {
        "Valid Samples": len(valid_data),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Bias": float(np.mean(errors)),
    }


# =========================================================
# CHRONOLOGICAL DATA SPLIT
# =========================================================

def chronological_split(
    df: pd.DataFrame,
) -> tuple:
    """
    Split observations chronologically into 60:20:20.

    No random shuffling is performed.
    """

    total_samples = len(df)

    if total_samples < 10:
        raise ValueError(
            "Dataset must contain at least 10 observations."
        )

    total_ratio = (
        CALIBRATION_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            "Calibration, validation, and test ratios must sum to 1."
        )

    calibration_end = int(
        total_samples * CALIBRATION_RATIO
    )

    validation_end = int(
        total_samples
        * (CALIBRATION_RATIO + VALIDATION_RATIO)
    )

    calibration_df = df.iloc[:calibration_end].copy()

    validation_df = df.iloc[
        calibration_end:validation_end
    ].copy()

    test_df = df.iloc[validation_end:].copy()

    if (
        calibration_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise ValueError(
            "One or more chronological data splits are empty."
        )

    return calibration_df, validation_df, test_df


# =========================================================
# DATASET VALIDATION
# =========================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate required ground truth and sensor columns.
    """

    if df.empty:
        raise ValueError("Environmental dataset is empty.")

    required_columns = list(VARIABLES.values())

    for variable in VARIABLES:
        sensor_columns = [
            column
            for column in df.columns
            if column.startswith(f"{variable}_sensor_")
        ]

        if not sensor_columns:
            raise ValueError(
                f"No sensor columns found for {variable}."
            )

        required_columns.extend(sensor_columns)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for true_column in VARIABLES.values():
        if not np.isfinite(
            pd.to_numeric(
                df[true_column],
                errors="coerce",
            )
        ).all():
            raise ValueError(
                f"Ground truth contains missing or invalid values: "
                f"{true_column}"
            )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # -----------------------------------------------------
    # PROJECT PATHS
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "data"
        / "simulated"
        / "environmental_sensors.csv"
    )

    output_dir = (
        project_root
        / "data"
        / "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # -----------------------------------------------------
    # LOAD AND VALIDATE DATA
    # -----------------------------------------------------

    df = pd.read_csv(data_path)

    validate_dataset(df)

    # -----------------------------------------------------
    # CHRONOLOGICAL 60:20:20 SPLIT
    # -----------------------------------------------------

    calibration_df, validation_df, test_df = (
        chronological_split(df)
    )

    print("=" * 75)
    print("DIMASA AI - MULTI-SENSOR DATA FUSION")
    print("=" * 75)

    print(f"Total observations : {len(df)}")
    print(f"Calibration data   : {len(calibration_df)}")
    print(f"Validation data    : {len(validation_df)}")
    print(f"Test data          : {len(test_df)}")
    print("Split ratio        : 60% / 20% / 20%")
    print("Split strategy     : Chronological")
    print("Weight estimation  : Calibration data only")

    # -----------------------------------------------------
    # INITIALIZE OUTPUTS
    # -----------------------------------------------------

    fusion_results = df.copy()

    weight_results = []
    performance_results = []

    # -----------------------------------------------------
    # PROCESS EACH ENVIRONMENTAL VARIABLE
    # -----------------------------------------------------

    for variable, true_column in VARIABLES.items():

        print(f"\nPROCESSING: {variable.upper()}")
        print("-" * 75)

        # Estimate weights using calibration data only
        weights, variances = estimate_sensor_weights(
            calibration_df,
            variable,
            true_column,
        )

        # Save and display sensor weights
        for sensor, weight in weights.items():

            weight_results.append({
                "Variable": variable,
                "Sensor": sensor,
                "Error Variance": variances[sensor],
                "Weight": weight,
                "Calibration Samples": len(calibration_df),
            })

            print(
                f"{sensor:<25} "
                f"Variance: {variances[sensor]:.6f} "
                f"Weight: {weight:.4f}"
            )

        # -------------------------------------------------
        # APPLY FIXED WEIGHTS TO ALL OBSERVATIONS
        # -------------------------------------------------

        sensor_columns = list(weights.keys())

        fused_column = f"{variable}_fused"

        fusion_results[fused_column] = (
            fusion_results[sensor_columns].apply(
                lambda row: weighted_sensor_fusion(
                    row.to_dict(),
                    weights,
                ),
                axis=1,
            )
        )

        # -------------------------------------------------
        # EVALUATE ON TEST SET ONLY
        # -------------------------------------------------

        test_metrics = calculate_metrics(
            test_df[true_column],
            fusion_results.loc[
                test_df.index,
                fused_column,
            ],
        )

        performance_results.append({
            "Variable": variable,
            **test_metrics,
        })

    # -----------------------------------------------------
    # PREPARE REPORTS
    # -----------------------------------------------------

    weights_df = pd.DataFrame(weight_results)

    performance_df = pd.DataFrame(
        performance_results
    )

    performance_df[
        ["MAE", "RMSE", "Bias"]
    ] = performance_df[
        ["MAE", "RMSE", "Bias"]
    ].round(4)

    # -----------------------------------------------------
    # EXPORT FUSED DATASET
    # -----------------------------------------------------

    fused_output = (
        output_dir
        / "fused_environmental_data.csv"
    )

    fusion_results.to_csv(
        fused_output,
        index=False,
    )

    # -----------------------------------------------------
    # EXPORT SENSOR WEIGHTS
    # -----------------------------------------------------

    weights_output = (
        output_dir
        / "sensor_fusion_weights.csv"
    )

    weights_df.to_csv(
        weights_output,
        index=False,
    )

    # -----------------------------------------------------
    # EXPORT PERFORMANCE REPORT
    # -----------------------------------------------------

    performance_output = (
        output_dir
        / "fusion_performance.csv"
    )

    performance_df.to_csv(
        performance_output,
        index=False,
    )

    # -----------------------------------------------------
    # DISPLAY PERFORMANCE REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 75)
    print("FUSION PERFORMANCE - INDEPENDENT TEST DATA")
    print("=" * 75)

    print(
        performance_df.to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("OUTPUT FILES")
    print("=" * 75)

    print(f"Fused dataset : {fused_output}")
    print(f"Sensor weights: {weights_output}")
    print(f"Performance   : {performance_output}")

    print("\nDIMASA AI - DATA FUSION COMPLETED!")


if __name__ == "__main__":
    main()
