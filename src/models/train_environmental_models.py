
"""
DIMASA AI
Environmental Machine Learning Pipeline

Models:
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Targets:
- Temperature
- Humidity
- Pressure
- PM2.5

Evaluation:
- MAE
- RMSE
- R2 Score

Data split:
- First 60%: Training
- Next 20%: Validation
- Last 20%: Testing

Model selection:
- Best model selected using Validation MAE.
- Selected model retrained on Training + Validation.
- Final evaluation performed on Test set only.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

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
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

RANDOM_STATE = 42


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_time_features(df):
    """Extract cyclical time features from timestamps."""

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    result = pd.DataFrame(index=df.index)

    result["hour_sin"] = np.sin(
        2 * np.pi * timestamps.dt.hour / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * timestamps.dt.hour / 24
    )

    result["day_sin"] = np.sin(
        2 * np.pi * timestamps.dt.day_of_week / 7
    )

    result["day_cos"] = np.cos(
        2 * np.pi * timestamps.dt.day_of_week / 7
    )

    result["month_sin"] = np.sin(
        2 * np.pi * (timestamps.dt.month - 1) / 12
    )

    result["month_cos"] = np.cos(
        2 * np.pi * (timestamps.dt.month - 1) / 12
    )

    return result


# ============================================================
# MODEL FACTORY
# ============================================================

def create_models():
    """Create the three regression pipelines."""

    return {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_eval, y_eval):
    """Calculate regression evaluation metrics."""

    predictions = model.predict(X_eval)

    return {
        "MAE": mean_absolute_error(
            y_eval,
            predictions,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_eval,
                predictions,
            )
        ),
        "R2": r2_score(
            y_eval,
            predictions,
        ),
        "Predictions": predictions,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. PROJECT PATHS
    # --------------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "data"
        / "processed"
        / "fused_environmental_data.csv"
    )

    model_dir = project_root / "models"
    output_dir = project_root / "data" / "processed"

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Fused dataset not found: {data_path}"
        )

    # --------------------------------------------------------
    # 2. LOAD AND VALIDATE DATASET
    # --------------------------------------------------------

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("The fused dataset is empty.")

    if not np.isclose(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO,
        1.0,
    ):
        raise ValueError(
            "Train, validation, and test ratios must sum to 1."
        )

    required_columns = [
        "timestamp",
        *TARGETS.values(),
        *FUSION_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 3. CHRONOLOGICAL TRAIN-VALIDATION-TEST SPLIT
    # --------------------------------------------------------

    total_samples = len(df)

    train_end = int(total_samples * TRAIN_RATIO)

    validation_end = int(
        total_samples * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[validation_end:].copy()

    if (
        train_df.empty
        or validation_df.empty
        or test_df.empty
    ):
        raise ValueError(
            "One or more data splits are empty."
        )

    assert (
        len(train_df)
        + len(validation_df)
        + len(test_df)
        == total_samples
    )

    # --------------------------------------------------------
    # 4. DATA SPLIT REPORT
    # --------------------------------------------------------

    print("=" * 80)
    print("DIMASA AI - ENVIRONMENTAL MACHINE LEARNING")
    print("=" * 80)

    print(f"Total observations : {total_samples}")
    print(f"Training samples   : {len(train_df)}")
    print(f"Validation samples : {len(validation_df)}")
    print(f"Testing samples    : {len(test_df)}")
    print("Algorithms         : 3")
    print(f"Prediction targets : {len(TARGETS)}")
    print("Data split         : Chronological 60/20/20")

    # --------------------------------------------------------
    # 5. TIME FEATURE ENGINEERING
    # --------------------------------------------------------

    time_features = create_time_features(df)

    # --------------------------------------------------------
    # 6. EXPERIMENT STORAGE
    # --------------------------------------------------------

    validation_results = []
    test_results = []
    selected_models = {}

    prediction_data = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
    })

    # --------------------------------------------------------
    # 7. MODEL TRAINING AND SELECTION
    # --------------------------------------------------------

    for target_name, target_column in TARGETS.items():

        print("\n" + "=" * 80)
        print(f"TARGET: {target_name.upper()}")
        print("=" * 80)

        # Exclude the target's own fused value.
        features = [
            column
            for column in FUSION_FEATURES
            if column != f"{target_name}_fused"
        ]

        # Combine sensor fusion and time features.
        X = pd.concat(
            [
                df[features],
                time_features,
            ],
            axis=1,
        )

        # Chronological feature split.
        X_train = X.iloc[:train_end]

        X_validation = X.iloc[
            train_end:validation_end
        ]

        X_test = X.iloc[validation_end:]

        # Target split.
        y_train = train_df[target_column]
        y_validation = validation_df[target_column]
        y_test = test_df[target_column]

        # Verify feature and target alignment.
        assert len(X_train) == len(y_train)
        assert len(X_validation) == len(y_validation)
        assert len(X_test) == len(y_test)

        # ----------------------------------------------------
        # 7A. TRAIN CANDIDATE MODELS
        # ----------------------------------------------------

        models = create_models()
        candidate_results = []

        print("\nVALIDATION MODEL COMPARISON")
        print("-" * 80)

        for model_name, model in models.items():

            # Train using training data only.
            model.fit(
                X_train,
                y_train,
            )

            # Evaluate using validation data.
            evaluation = evaluate_model(
                model,
                X_validation,
                y_validation,
            )

            validation_predictions = evaluation.pop(
                "Predictions"
            )

            validation_mae = evaluation["MAE"]

            candidate_results.append({
                "Target": target_name,
                "Model": model_name,
                "MAE": validation_mae,
                "RMSE": evaluation["RMSE"],
                "R2": evaluation["R2"],
            })

            validation_results.append({
                "Target": target_name,
                "Model": model_name,
                "Validation_MAE": evaluation["MAE"],
                "Validation_RMSE": evaluation["RMSE"],
                "Validation_R2": evaluation["R2"],
            })

            print(
                f"{model_name:<22}"
                f"MAE: {evaluation['MAE']:.4f} | "
                f"RMSE: {evaluation['RMSE']:.4f} | "
                f"R2: {evaluation['R2']:.4f}"
            )

        # ----------------------------------------------------
        # 7B. SELECT BEST MODEL USING VALIDATION MAE
        # ----------------------------------------------------

        candidate_df = pd.DataFrame(candidate_results)

        best_candidate = candidate_df.loc[
            candidate_df["MAE"].idxmin()
        ]

        best_model_name = best_candidate["Model"]

        print("\nSELECTED MODEL")
        print("-" * 80)

        print(f"Target       : {target_name}")
        print(f"Best model   : {best_model_name}")
        print(
            f"Validation MAE: "
            f"{best_candidate['MAE']:.4f}"
        )

        selected_models[target_name] = best_model_name

        # ----------------------------------------------------
        # 7C. RETRAIN SELECTED MODEL
        #     USING TRAINING + VALIDATION DATA
        # ----------------------------------------------------

        X_train_validation = pd.concat(
            [
                X_train,
                X_validation,
            ],
            axis=0,
        )

        y_train_validation = pd.concat(
            [
                y_train,
                y_validation,
            ],
            axis=0,
        )

        final_model = clone(
            models[best_model_name]
        )

        final_model.fit(
            X_train_validation,
            y_train_validation,
        )

        # ----------------------------------------------------
        # 7D. FINAL EVALUATION ON TEST DATA
        # ----------------------------------------------------

        final_evaluation = evaluate_model(
            final_model,
            X_test,
            y_test,
        )

        test_predictions = final_evaluation.pop(
            "Predictions"
        )

        test_results.append({
            "Target": target_name,
            "Selected_Model": best_model_name,
            "Validation_MAE": best_candidate["MAE"],
            "MAE": final_evaluation["MAE"],
            "RMSE": final_evaluation["RMSE"],
            "R2": final_evaluation["R2"],
        })

        # Store actual and predicted test values.
        prediction_data[
            f"{target_name}_actual"
        ] = y_test.to_numpy()

        prediction_data[
            f"{target_name}_prediction"
        ] = test_predictions

        # ----------------------------------------------------
        # 7E. SAVE FINAL SELECTED MODEL
        # ----------------------------------------------------

        model_path = (
            model_dir
            / f"{target_name}_best_model.joblib"
        )

        joblib.dump(
            {
                "model": final_model,
                "features": list(X.columns),
                "target": target_column,
                "selected_model": best_model_name,
                "selection_metric": "Validation MAE",
                "train_samples": len(X_train),
                "validation_samples": len(X_validation),
                "final_training_samples": len(
                    X_train_validation
                ),
                "test_samples": len(X_test),
            },
            model_path,
        )

        print("\nFINAL TEST PERFORMANCE")
        print("-" * 80)

        print(
            f"MAE  : {final_evaluation['MAE']:.4f}"
        )

        print(
            f"RMSE : {final_evaluation['RMSE']:.4f}"
        )

        print(
            f"R2   : {final_evaluation['R2']:.4f}"
        )

        print(f"Saved model: {model_path.name}")

    # --------------------------------------------------------
    # 8. SAVE VALIDATION RESULTS
    # --------------------------------------------------------

    validation_results_df = pd.DataFrame(
        validation_results
    )

    validation_results_df[
        [
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_R2",
        ]
    ] = validation_results_df[
        [
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_R2",
        ]
    ].round(4)

    validation_results_path = (
        output_dir / "machine_learning_results.csv"
    )

    validation_results_df.to_csv(
        validation_results_path,
        index=False,
    )

    # --------------------------------------------------------
    # 9. SAVE FINAL TEST RESULTS
    # --------------------------------------------------------

    test_results_df = pd.DataFrame(test_results)

    numeric_columns = [
        "Validation_MAE",
        "MAE",
        "RMSE",
        "R2",
    ]

    test_results_df[numeric_columns] = (
        test_results_df[numeric_columns].round(4)
    )

    test_results_path = (
        output_dir / "machine_learning_test_results.csv"
    )

    test_results_df.to_csv(
        test_results_path,
        index=False,
    )

    # --------------------------------------------------------
    # 10. SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    predictions_path = (
        output_dir / "machine_learning_predictions.csv"
    )

    prediction_data.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # 11. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("DIMASA AI - FINAL MODEL SUMMARY")
    print("=" * 80)

    for _, row in test_results_df.iterrows():

        print(
            f"{row['Target']:<15}"
            f"{row['Selected_Model']:<22}"
            f"MAE: {row['MAE']:.4f} | "
            f"RMSE: {row['RMSE']:.4f} | "
            f"R2: {row['R2']:.4f}"
        )

    # --------------------------------------------------------
    # 12. OUTPUT FILES
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(f"Model directory       : {model_dir}")
    print(f"Validation results    : {validation_results_path}")
    print(f"Final test results    : {test_results_path}")
    print(f"Test predictions      : {predictions_path}")

    print("\nDIMASA AI - MACHINE LEARNING COMPLETED!")


if __name__ == "__main__":
    main()
