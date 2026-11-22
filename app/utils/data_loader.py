
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# =========================================================
# DATASET REGISTRY
# =========================================================

DATASETS = {
    "fused": "fused_environmental_data.csv",
    "fusion_comparison": "fusion_comparison.csv",
    "fusion_performance": "fusion_performance.csv",
    "sensor_errors": "sensor_error_analysis.csv",
    "sensor_weights": "sensor_fusion_weights.csv",
    "ml_results": "machine_learning_results.csv",
    "ml_predictions": "machine_learning_predictions.csv",
    "uncertainty_metrics": "uncertainty_metrics.csv",
    "uncertainty_predictions": "uncertainty_predictions.csv",
}


# =========================================================
# LOAD SINGLE DATASET
# =========================================================

@st.cache_data
def load_dataset(filename):
    """Load a CSV dataset from the processed data directory."""

    file_path = PROCESSED_DATA_DIR / filename

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
        st.error(f"Unable to read {filename}: {error}")
        return None


# =========================================================
# LOAD ALL DATASETS
# =========================================================

def load_all_datasets():
    """Load all registered datasets into a dictionary."""

    return {
        key: load_dataset(filename)
        for key, filename in DATASETS.items()
    }


# =========================================================
# DATA AVAILABILITY
# =========================================================

def get_dataset_status(datasets):
    """Return dataset availability information."""

    return {
        key: {
            "available": data is not None,
            "rows": len(data) if data is not None else 0,
            "columns": len(data.columns) if data is not None else 0,
        }
        for key, data in datasets.items()
    }


# =========================================================
# TIMESTAMP PREPROCESSING
# =========================================================

def prepare_timestamp(data):
    """Convert timestamp column into pandas datetime."""

    if data is None or "timestamp" not in data.columns:
        return data

    result = data.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    return result
