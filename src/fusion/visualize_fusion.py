
"""
DIMASA AI
Sensor Fusion Visualization

Generates:
1. MAE and RMSE comparison charts.
2. Ground truth vs weighted fusion time series.
3. Sensor reliability weight charts.
4. Sensor missing-value analysis.

Data configuration:
- Calibration: 60%
- Validation: 20%
- Testing: 20%

Visualization policy:
- Performance comparison uses fusion_comparison.csv.
- Time series evaluation displays the independent test period only.
- Sensor weights come from calibration data only.
- Missing-value analysis covers the complete dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

VARIABLES = {
    "temperature": ("Temperature", "°C"),
    "humidity": ("Humidity", "%"),
    "pressure": ("Pressure", "hPa"),
    "pm25": ("PM2.5", "µg/m³"),
}

CALIBRATION_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

METRICS = ["MAE", "RMSE"]


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

def setup_output_directory(project_root: Path) -> Path:
    """
    Create the visualization output directory.
    """

    output_dir = (
        project_root
        / "data"
        / "processed"
        / "visualizations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# =========================================================
# DATA VALIDATION
# =========================================================

def validate_datasets(
    original_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    fused_df: pd.DataFrame,
) -> None:
    """
    Validate required columns and dataset alignment.
    """

    if original_df.empty:
        raise ValueError("Original dataset is empty.")

    if fused_df.empty:
        raise ValueError("Fused dataset is empty.")

    if comparison_df.empty:
        raise ValueError("Fusion comparison dataset is empty.")

    if weights_df.empty:
        raise ValueError("Sensor weights dataset is empty.")

    if len(original_df) != len(fused_df):
        raise ValueError(
            "Original and fused datasets have different row counts."
        )

    required_comparison_columns = {
        "Variable",
        "Method",
        "MAE",
        "RMSE",
    }

    required_weight_columns = {
        "Variable",
        "Sensor",
        "Weight",
    }

    missing_comparison = (
        required_comparison_columns
        - set(comparison_df.columns)
    )

    missing_weights = (
        required_weight_columns
        - set(weights_df.columns)
    )

    if missing_comparison:
        raise ValueError(
            f"Missing comparison columns: {missing_comparison}"
        )

    if missing_weights:
        raise ValueError(
            f"Missing weight columns: {missing_weights}"
        )

    for variable in VARIABLES:

        required_original = [
            f"{variable}_true",
        ]

        required_fused = [
            f"{variable}_fused",
        ]

        for sensor_number in range(1, 4):
            required_original.append(
                f"{variable}_sensor_{sensor_number}"
            )

        for column in required_original:
            if column not in original_df.columns:
                raise ValueError(
                    f"Missing original column: {column}"
                )

        for column in required_fused:
            if column not in fused_df.columns:
                raise ValueError(
                    f"Missing fused column: {column}"
                )

    # Verify timestamp alignment.
    if (
        "timestamp" in original_df.columns
        and "timestamp" in fused_df.columns
    ):
        original_timestamps = (
            original_df["timestamp"]
            .astype(str)
            .reset_index(drop=True)
        )

        fused_timestamps = (
            fused_df["timestamp"]
            .astype(str)
            .reset_index(drop=True)
        )

        if not original_timestamps.equals(fused_timestamps):
            raise ValueError(
                "Original and fused timestamps are not aligned."
            )


# =========================================================
# CHRONOLOGICAL TEST DATA
# =========================================================

def extract_test_data(
    original_df: pd.DataFrame,
    fused_df: pd.DataFrame,
) -> tuple:
    """
    Extract the final 20% of observations.

    Uses the same chronological split as the
    sensor fusion and machine learning pipelines.
    """

    total_ratio = (
        CALIBRATION_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            "Data split ratios must sum to 1."
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
            "Test visualization data is empty."
        )

    return test_original, test_fused


# =========================================================
# PERFORMANCE COMPARISON
# =========================================================

def plot_error_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
) -> int:
    """
    Compare MAE and RMSE across sensor methods.
    """

    generated = 0

    for metric in METRICS:

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        pivot = comparison_df.pivot(
            index="Variable",
            columns="Method",
            values=metric,
        )

        pivot = pivot.reindex(
            list(VARIABLES.keys())
        )

        pivot.plot(
            kind="bar",
            ax=ax,
            width=0.8,
        )

        ax.set_title(
            f"DIMASA AI | {metric} Performance Comparison",
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Environmental Variable"
        )

        ax.set_ylabel(metric)

        ax.tick_params(
            axis="x",
            rotation=0,
        )

        ax.legend(
            title="Method",
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        filename = f"{metric.lower()}_comparison.png"

        fig.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        generated += 1

        print(f"Saved: {filename}")

    return generated


# =========================================================
# FUSION TIME SERIES
# =========================================================

def plot_fusion_timeseries(
    test_original: pd.DataFrame,
    test_fused: pd.DataFrame,
    output_dir: Path,
) -> int:
    """
    Plot ground truth and weighted fusion
    on the independent test period.
    """

    generated = 0

    if "timestamp" not in test_original.columns:
        raise ValueError(
            "Timestamp column is required for time series visualization."
        )

    timestamps = pd.to_datetime(
        test_original["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Invalid timestamp values detected."
        )

    for variable, (label, unit) in VARIABLES.items():

        true_column = f"{variable}_true"
        fused_column = f"{variable}_fused"

        true_values = pd.to_numeric(
            test_original[true_column],
            errors="coerce",
        )

        fused_values = pd.to_numeric(
            test_fused[fused_column],
            errors="coerce",
        )

        fig, ax = plt.subplots(
            figsize=(13, 5)
        )

        ax.plot(
            timestamps,
            true_values,
            label="Ground Truth",
            linewidth=1.8,
        )

        ax.plot(
            timestamps,
            fused_values,
            label="Weighted Fusion",
            linewidth=1.2,
            alpha=0.85,
        )

        ax.set_title(
            f"DIMASA AI | {label}: Ground Truth vs Fusion",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel("Timestamp")

        ax.set_ylabel(
            f"{label} ({unit})"
        )

        ax.legend()

        ax.grid(
            alpha=0.3,
        )

        fig.autofmt_xdate()

        fig.tight_layout()

        filename = (
            f"{variable}_fusion_timeseries.png"
        )

        fig.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        generated += 1

        print(f"Saved: {filename}")

    return generated


# =========================================================
# SENSOR RELIABILITY WEIGHTS
# =========================================================

def plot_sensor_weights(
    weights_df: pd.DataFrame,
    output_dir: Path,
) -> int:
    """
    Visualize sensor reliability weights.
    """

    generated = 0

    for variable, (label, _) in VARIABLES.items():

        variable_weights = weights_df[
            weights_df["Variable"] == variable
        ].copy()

        if variable_weights.empty:
            print(
                f"Skipped weights: {variable}"
            )
            continue

        variable_weights["Weight"] = pd.to_numeric(
            variable_weights["Weight"],
            errors="coerce",
        )

        variable_weights = variable_weights.dropna(
            subset=["Weight"]
        )

        if variable_weights.empty:
            continue

        variable_weights = variable_weights.sort_values(
            "Sensor"
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            variable_weights["Sensor"],
            variable_weights["Weight"] * 100,
        )

        ax.set_title(
            f"DIMASA AI | {label} Sensor Reliability Weights",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel("Sensor")

        ax.set_ylabel("Weight (%)")

        ax.tick_params(
            axis="x",
            rotation=15,
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        filename = (
            f"{variable}_sensor_weights.png"
        )

        fig.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        generated += 1

        print(f"Saved: {filename}")

    return generated


# =========================================================
# SENSOR MISSING-VALUE ANALYSIS
# =========================================================

def plot_missing_values(
    original_df: pd.DataFrame,
    output_dir: Path,
) -> int:
    """
    Visualize missing-value percentages
    for all sensor channels.
    """

    sensor_columns = [
        column
        for column in original_df.columns
        if "_sensor_" in column
    ]

    if not sensor_columns:
        raise ValueError(
            "No sensor columns available for missing-value analysis."
        )

    missing_percentages = (
        original_df[sensor_columns]
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    missing_percentages.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "DIMASA AI | Sensor Missing-Value Analysis",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Sensor")

    ax.set_ylabel("Missing Values (%)")

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    filename = "sensor_missing_values.png"

    fig.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")

    return 1


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

    comparison_path = (
        data_dir
        / "fusion_comparison.csv"
    )

    weights_path = (
        data_dir
        / "sensor_fusion_weights.csv"
    )

    fused_path = (
        data_dir
        / "fused_environmental_data.csv"
    )

    required_files = [
        original_path,
        comparison_path,
        weights_path,
        fused_path,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # -----------------------------------------------------
    # LOAD DATASETS
    # -----------------------------------------------------

    original_df = pd.read_csv(
        original_path
    )

    comparison_df = pd.read_csv(
        comparison_path
    )

    weights_df = pd.read_csv(
        weights_path
    )

    fused_df = pd.read_csv(
        fused_path
    )

    validate_datasets(
        original_df,
        comparison_df,
        weights_df,
        fused_df,
    )

    # -----------------------------------------------------
    # EXTRACT TEST PERIOD
    # -----------------------------------------------------

    test_original, test_fused = extract_test_data(
        original_df,
        fused_df,
    )

    # -----------------------------------------------------
    # PREPARE OUTPUT DIRECTORY
    # -----------------------------------------------------

    output_dir = setup_output_directory(
        project_root
    )

    # -----------------------------------------------------
    # DISPLAY PIPELINE INFORMATION
    # -----------------------------------------------------

    print("=" * 70)
    print("DIMASA AI - FUSION VISUALIZATION")
    print("=" * 70)

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
    print("Visualization      : Independent test period")

    # -----------------------------------------------------
    # GENERATE VISUALIZATIONS
    # -----------------------------------------------------

    generated_count = 0

    generated_count += plot_error_comparison(
        comparison_df,
        output_dir,
    )

    generated_count += plot_fusion_timeseries(
        test_original,
        test_fused,
        output_dir,
    )

    generated_count += plot_sensor_weights(
        weights_df,
        output_dir,
    )

    generated_count += plot_missing_values(
        original_df,
        output_dir,
    )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)

    print(f"Generated charts: {generated_count}")
    print(f"Output directory: {output_dir}")

    print("\nDIMASA AI - VISUALIZATION COMPLETED!")

    print("=" * 70)


if __name__ == "__main__":
    main()
