
"""
DIMASA AI
Fusion Performance Comparison

Compares:
1. Individual Sensor 1
2. Individual Sensor 2
3. Individual Sensor 3
4. Inverse-Variance Weighted Sensor Fusion

Evaluation:
- MAE
- RMSE
- Bias

Data split:
- Calibration: 60%
- Validation: 20%
- Testing: 20%

Evaluation policy:
- All methods use the same independent test period.
- All methods are evaluated on identical valid observations.
- No sensor weights are estimated in this script.
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

    comparison = pd.DataFrame({
        "true": true_values,
        "predicted": predicted_values,
    })

    comparison = comparison.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if comparison.empty:
        return {
            "Valid Samples": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
        }

    errors = (
        comparison["predicted"]
        - comparison["true"]
    ).to_numpy()

    return {
        "Valid Samples": len(errors),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Bias": float(np.mean(errors)),
    }


# =========================================================
# CHRONOLOGICAL TEST SPLIT
# =========================================================

def get_test_data(
    original_df: pd.DataFrame,
    fused_df: pd.DataFrame,
) -> tuple:
    """
    Extract the final 20% test period.

    Uses the same chronological boundaries as the
    sensor fusion and machine learning pipelines.
    """

    if len(original_df) != len(fused_df):
        raise ValueError(
            "Original and fused datasets have different row counts."
        )

    total_ratio = (
        CALIBRATION_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            "Data split ratios must sum to 1."
        )

    if len(original_df) < 10:
        raise ValueError(
            "Dataset must contain at least 10 observations."
        )

    test_start = int(
        len(original_df)
        * (CALIBRATION_RATIO + VALIDATION_RATIO)
    )

    test_original = (
        original_df.iloc[test_start:]
        .reset_index(drop=True)
        .copy()
    )

    test_fused = (
        fused_df.iloc[test_start:]
        .reset_index(drop=True)
        .copy()
    )

    if test_original.empty or test_fused.empty:
        raise ValueError(
            "Test dataset is empty."
        )

    # Verify chronological alignment when timestamps exist.
    if (
        "timestamp" in test_original.columns
        and "timestamp" in test_fused.columns
    ):
        original_timestamps = (
            test_original["timestamp"]
            .astype(str)
            .reset_index(drop=True)
        )

        fused_timestamps = (
            test_fused["timestamp"]
            .astype(str)
            .reset_index(drop=True)
        )

        if not original_timestamps.equals(fused_timestamps):
            raise ValueError(
                "Original and fused timestamps are not aligned."
            )

    return test_original, test_fused


# =========================================================
# DATA VALIDATION
# =========================================================

def validate_columns(
    original_df: pd.DataFrame,
    fused_df: pd.DataFrame,
) -> None:
    """
    Validate required sensor, ground truth,
    and fusion columns.
    """

    missing_original = []
    missing_fused = []

    for variable, true_column in VARIABLES.items():

        sensor_columns = [
            f"{variable}_sensor_1",
            f"{variable}_sensor_2",
            f"{variable}_sensor_3",
        ]

        required_original = [
            true_column,
            *sensor_columns,
        ]

        required_fused = [
            f"{variable}_fused",
        ]

        missing_original.extend([
            column
            for column in required_original
            if column not in original_df.columns
        ])

        missing_fused.extend([
            column
            for column in required_fused
            if column not in fused_df.columns
        ])

    if missing_original:
        raise ValueError(
            f"Missing original dataset columns: {missing_original}"
        )

    if missing_fused:
        raise ValueError(
            f"Missing fused dataset columns: {missing_fused}"
        )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # -----------------------------------------------------
    # PROJECT PATHS
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    data_dir = (
        project_root
        / "data"
        / "processed"
    )

    original_path = (
        project_root
        / "data"
        / "simulated"
        / "environmental_sensors.csv"
    )

    fused_path = (
        data_dir
        / "fused_environmental_data.csv"
    )

    if not original_path.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {original_path}"
        )

    if not fused_path.exists():
        raise FileNotFoundError(
            f"Fused dataset not found: {fused_path}"
        )

    # -----------------------------------------------------
    # LOAD DATASETS
    # -----------------------------------------------------

    original_df = pd.read_csv(original_path)
    fused_df = pd.read_csv(fused_path)

    validate_columns(
        original_df,
        fused_df,
    )

    # -----------------------------------------------------
    # EXTRACT INDEPENDENT TEST DATA
    # -----------------------------------------------------

    test_original, test_fused = get_test_data(
        original_df,
        fused_df,
    )

    results = []

    print("=" * 80)
    print("DIMASA AI - FUSION PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"Total observations : {len(original_df)}")
    print(
        f"Calibration data   : "
        f"{int(len(original_df) * CALIBRATION_RATIO)}"
    )
    print(
        f"Validation data    : "
        f"{int(len(original_df) * VALIDATION_RATIO)}"
    )
    print(
        f"Test observations  : {len(test_original)}"
    )
    print("Split ratio        : 60% / 20% / 20%")
    print("Evaluation period  : Final 20%")
    print("Comparison policy  : Common valid observations")

    # -----------------------------------------------------
    # COMPARE EACH ENVIRONMENTAL VARIABLE
    # -----------------------------------------------------

    for variable, true_column in VARIABLES.items():

        sensor_columns = [
            f"{variable}_sensor_1",
            f"{variable}_sensor_2",
            f"{variable}_sensor_3",
        ]

        fused_column = f"{variable}_fused"

        print(f"\nVARIABLE: {variable.upper()}")
        print("-" * 80)

        # Build aligned comparison table.
        comparison_df = pd.DataFrame({
            "Ground Truth": test_original[true_column],
            "Sensor 1": test_original[sensor_columns[0]],
            "Sensor 2": test_original[sensor_columns[1]],
            "Sensor 3": test_original[sensor_columns[2]],
            "Weighted Fusion": test_fused[fused_column],
        })

        # Exclude invalid observations.
        comparison_df = comparison_df.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if comparison_df.empty:
            print(
                "Skipping variable: no common valid test observations."
            )
            continue

        true_values = comparison_df["Ground Truth"]

        print(
            f"Common valid test observations: "
            f"{len(comparison_df)}"
        )

        methods = [
            "Sensor 1",
            "Sensor 2",
            "Sensor 3",
            "Weighted Fusion",
        ]

        variable_results = []

        for method in methods:

            metrics = calculate_metrics(
                true_values,
                comparison_df[method],
            )

            result = {
                "Variable": variable,
                "Method": method,
                **metrics,
            }

            variable_results.append(result)

        variable_df = pd.DataFrame(
            variable_results
        )

        # Rank methods by MAE.
        variable_df["MAE Rank"] = (
            variable_df["MAE"]
            .rank(method="min")
            .astype(int)
        )

        variable_df[
            ["MAE", "RMSE", "Bias"]
        ] = variable_df[
            ["MAE", "RMSE", "Bias"]
        ].round(4)

        print(
            variable_df.to_string(index=False)
        )

        # Identify the best method.
        best_method = variable_df.loc[
            variable_df["MAE"].idxmin()
        ]

        print(
            f"\nBest method: {best_method['Method']}"
        )

        results.extend(variable_results)

    # -----------------------------------------------------
    # EXPORT COMPARISON REPORT
    # -----------------------------------------------------

    if not results:
        raise RuntimeError(
            "No valid fusion comparison results were generated."
        )

    results_df = pd.DataFrame(results)

    results_df["MAE Rank"] = (
        results_df.groupby("Variable")["MAE"]
        .rank(method="min")
        .astype(int)
    )

    results_df[
        ["MAE", "RMSE", "Bias"]
    ] = results_df[
        ["MAE", "RMSE", "Bias"]
    ].round(4)

    output_file = (
        data_dir
        / "fusion_comparison.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    for variable in VARIABLES:

        variable_results = results_df[
            results_df["Variable"] == variable
        ]

        if variable_results.empty:
            continue

        best = variable_results.loc[
            variable_results["MAE"].idxmin()
        ]

        print(
            f"{variable:<15} "
            f"Best: {best['Method']:<18} "
            f"MAE: {best['MAE']:.4f}"
        )

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(f"Comparison report: {output_file}")

    print("\nDIMASA AI - COMPARISON COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
