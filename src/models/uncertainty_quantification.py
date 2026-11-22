
"""
DIMASA AI
Uncertainty Quantification

Method:
Split Conformal Prediction

Data split:
- 60% Training
- 20% Calibration
- 20% Testing

Outputs:
- Prediction intervals
- Empirical coverage
- Mean interval width
- Conformal model artifacts

Note:
The dataset is synthetic and chronological.
Classical split conformal coverage guarantees rely on
exchangeability, which may not hold for time-dependent data.
"""

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = {
    "temperature": "temperature_true",
    "humidity": "humidity_true",
    "pressure": "pressure_true",
    "pm25": "pm25_true",
}

FUSION_FEATURES = [
    "temperature_fused",
    "humidity_fused",
    "pressure_fused",
    "pm25_fused",
]

TRAIN_RATIO = 0.60
CALIBRATION_RATIO = 0.20
TEST_RATIO = 0.20

COVERAGE_LEVEL = 0.90
RANDOM_STATE = 42


def create_time_features(df):
    """Create cyclical time features."""

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Invalid timestamp values detected."
        )

    features = pd.DataFrame(index=df.index)

    features["hour_sin"] = np.sin(
        2 * np.pi * timestamps.dt.hour / 24
    )
    features["hour_cos"] = np.cos(
        2 * np.pi * timestamps.dt.hour / 24
    )

    features["day_sin"] = np.sin(
        2 * np.pi * timestamps.dt.day_of_week / 7
    )
    features["day_cos"] = np.cos(
        2 * np.pi * timestamps.dt.day_of_week / 7
    )

    features["month_sin"] = np.sin(
        2 * np.pi * (timestamps.dt.month - 1) / 12
    )
    features["month_cos"] = np.cos(
        2 * np.pi * (timestamps.dt.month - 1) / 12
    )

    return features


def create_model(model_name):
    """Create a regression model matching the ML experiment."""

    if model_name == "Linear Regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])

    if model_name == "Random Forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

    if model_name == "Gradient Boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )),
        ])

    raise ValueError(
        f"Unsupported selected model: {model_name}"
    )


def load_selected_models(processed_dir):
    """
    Read the selected models from the latest ML test report.
    """

    results_path = (
        processed_dir / "machine_learning_test_results.csv"
    )

    if not results_path.exists():
        raise FileNotFoundError(
            f"ML test results not found: {results_path}"
        )

    results = pd.read_csv(results_path)

    required_columns = {
        "Target",
        "Selected_Model",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns in ML results: {sorted(missing)}"
        )

    selected_models = {}

    for target in TARGETS:
        matching = results[
            results["Target"].astype(str).str.lower() == target
        ]

        if len(matching) != 1:
            raise ValueError(
                f"Expected exactly one selected model for {target}."
            )

        selected_models[target] = str(
            matching.iloc[0]["Selected_Model"]
        )

        # Validate that the model can be constructed.
        create_model(selected_models[target])

    return selected_models


def calculate_conformal_quantile(
    calibration_residuals,
    coverage_level,
):
    """
    Calculate the finite-sample conformal quantile
    using absolute calibration residuals.
    """

    residuals = np.asarray(
        calibration_residuals,
        dtype=float,
    )

    residuals = residuals[np.isfinite(residuals)]

    n = len(residuals)

    if n == 0:
        raise ValueError(
            "No valid calibration residuals."
        )

    alpha = 1.0 - coverage_level

    rank = math.ceil((n + 1) * (1.0 - alpha))

    if rank > n:
        return float("inf")

    quantile = np.sort(residuals)[rank - 1]

    return float(quantile)


def validate_dataset(df):
    """Validate the input dataset before training."""

    required_columns = (
        ["timestamp"]
        + list(TARGETS.values())
        + FUSION_FEATURES
    )

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    if len(df) < 10:
        raise ValueError(
            "Dataset contains too few observations."
        )

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Dataset contains invalid timestamps."
        )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Dataset must be sorted chronologically."
        )

    for column in TARGETS.values():
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(
                f"Target column contains missing or invalid values: {column}"
            )

    return df


def main():

    project_root = Path(__file__).resolve().parents[2]

    processed_dir = (
        project_root / "data" / "processed"
    )

    model_dir = project_root / "models"

    data_path = (
        processed_dir / "fused_environmental_data.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(data_path)

    validate_dataset(df)

    total_samples = len(df)

    train_end = int(
        total_samples * TRAIN_RATIO
    )

    calibration_end = int(
        total_samples
        * (TRAIN_RATIO + CALIBRATION_RATIO)
    )

    train_df = df.iloc[:train_end]
    calibration_df = df.iloc[train_end:calibration_end]
    test_df = df.iloc[calibration_end:]

    if (
        len(train_df) == 0
        or len(calibration_df) == 0
        or len(test_df) == 0
    ):
        raise ValueError(
            "One or more data splits are empty."
        )

    # Load model choices from the latest ML experiment.
    selected_models = load_selected_models(
        processed_dir
    )

    # Generate time features.
    time_features = create_time_features(df)

    prediction_results = pd.DataFrame({
        "timestamp": test_df["timestamp"].values
    })

    metric_results = []

    print("=" * 80)
    print("DIMASA AI - UNCERTAINTY QUANTIFICATION")
    print("=" * 80)

    print(f"Total observations : {total_samples}")
    print(f"Training samples   : {len(train_df)}")
    print(f"Calibration samples: {len(calibration_df)}")
    print(f"Testing samples    : {len(test_df)}")
    print(f"Target coverage    : {COVERAGE_LEVEL:.0%}")

    print("\nSELECTED MODELS FROM MACHINE LEARNING")
    print("-" * 80)

    for target, model_name in selected_models.items():
        print(f"{target:<15} {model_name}")

    for target_name, target_column in TARGETS.items():

        print(f"\nTARGET: {target_name.upper()}")
        print("-" * 80)

        model_name = selected_models[target_name]

        # Exclude the target's own fused measurement.
        features = [
            column
            for column in FUSION_FEATURES
            if column != f"{target_name}_fused"
        ]

        X = pd.concat(
            [
                df[features],
                time_features,
            ],
            axis=1,
        )

        X_train = X.iloc[:train_end]
        X_calibration = X.iloc[
            train_end:calibration_end
        ]
        X_test = X.iloc[calibration_end:]

        y_train = train_df[target_column].to_numpy(
            dtype=float
        )

        y_calibration = calibration_df[
            target_column
        ].to_numpy(dtype=float)

        y_test = test_df[target_column].to_numpy(
            dtype=float
        )

        # Fit exclusively on training data.
        model = create_model(model_name)

        model.fit(
            X_train,
            y_train,
        )

        # Calibration predictions.
        calibration_predictions = model.predict(
            X_calibration
        )

        calibration_residuals = np.abs(
            y_calibration - calibration_predictions
        )

        q = calculate_conformal_quantile(
            calibration_residuals,
            COVERAGE_LEVEL,
        )

        # Final test predictions.
        test_predictions = model.predict(X_test)

        lower_bounds = test_predictions - q
        upper_bounds = test_predictions + q

        # Empirical test coverage.
        covered = (
            (y_test >= lower_bounds)
            & (y_test <= upper_bounds)
        )

        coverage = float(np.mean(covered))

        interval_width = float(
            np.mean(upper_bounds - lower_bounds)
        )

        mae = float(
            np.mean(
                np.abs(y_test - test_predictions)
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (y_test - test_predictions) ** 2
                )
            )
        )

        metric_results.append({
            "Target": target_name,
            "Model": model_name,
            "Training Samples": len(train_df),
            "Calibration Samples": len(calibration_residuals),
            "Test Samples": len(test_df),
            "Conformal Quantile": q,
            "Target Coverage": COVERAGE_LEVEL,
            "Actual Coverage": coverage,
            "Mean Interval Width": interval_width,
            "MAE": mae,
            "RMSE": rmse,
        })

        prediction_results[
            f"{target_name}_actual"
        ] = y_test

        prediction_results[
            f"{target_name}_prediction"
        ] = test_predictions

        prediction_results[
            f"{target_name}_lower"
        ] = lower_bounds

        prediction_results[
            f"{target_name}_upper"
        ] = upper_bounds

        # Save the trained conformal model.
        model_path = (
            model_dir
            / f"{target_name}_conformal_model.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "features": list(X.columns),
                "target": target_column,
                "selected_model": model_name,
                "conformal_quantile": q,
                "coverage_level": COVERAGE_LEVEL,
                "training_samples": len(train_df),
                "calibration_samples": len(calibration_residuals),
                "test_samples": len(test_df),
            },
            model_path,
        )

        print(f"Selected model      : {model_name}")
        print(f"Calibration samples : {len(calibration_residuals)}")
        print(f"Conformal quantile  : {q:.4f}")
        print(f"Actual coverage     : {coverage:.2%}")
        print(f"Mean interval width : {interval_width:.4f}")
        print(f"Test MAE            : {mae:.4f}")
        print(f"Test RMSE           : {rmse:.4f}")
        print(f"Saved model         : {model_path.name}")

    # Save evaluation metrics.
    metrics_df = pd.DataFrame(metric_results)

    metrics_path = (
        processed_dir / "uncertainty_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # Save test prediction intervals.
    predictions_path = (
        processed_dir / "uncertainty_predictions.csv"
    )

    prediction_results.to_csv(
        predictions_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("UNCERTAINTY EVALUATION SUMMARY")
    print("=" * 80)

    print(
        metrics_df.round(4).to_string(index=False)
    )

    print("\nOUTPUT FILES")
    print("-" * 80)
    print(f"Metrics     : {metrics_path}")
    print(f"Predictions : {predictions_path}")
    print(f"Models      : {model_dir}")

    print(
        "\nDIMASA AI - UNCERTAINTY QUANTIFICATION COMPLETED!"
    )


if __name__ == "__main__":
    main()
