
"""
DIMASA AI
Multi-Sensor Error Analysis

Evaluates the measurement quality of 12 virtual sensors.

Metrics:
1. Mean Absolute Error (MAE)
2. Root Mean Squared Error (RMSE)
3. Measurement Bias
4. Number of valid observations
5. MAE ranking within each variable
"""

from pathlib import Path

import numpy as np
import pandas as pd


VARIABLES = {
    "temperature": "temperature_true",
    "humidity": "humidity_true",
    "pressure": "pressure_true",
    "pm25": "pm25_true",
}


def calculate_sensor_metrics(
    df: pd.DataFrame,
    variable: str,
    true_column: str,
    sensor_column: str,
) -> dict:

    # Mengambil pasangan data yang tersedia
    valid_data = df[[true_column, sensor_column]].dropna()

    if valid_data.empty:
        return {
            "Variable": variable,
            "Sensor": sensor_column,
            "Valid Samples": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
        }

    y_true = valid_data[true_column].to_numpy()
    y_sensor = valid_data[sensor_column].to_numpy()

    # Error = hasil sensor - nilai sebenarnya
    errors = y_sensor - y_true

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)

    return {
        "Variable": variable,
        "Sensor": sensor_column,
        "Valid Samples": len(valid_data),
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
    }


def main():

    # Menentukan lokasi project
    project_root = Path(__file__).resolve().parents[2]

    # Membaca dataset
    data_path = (
        project_root
        / "data"
        / "simulated"
        / "environmental_sensors.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {data_path}"
        )

    df = pd.read_csv(data_path)

    # Memeriksa kolom ground truth
    required_columns = list(VARIABLES.values())

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom ground truth tidak ditemukan: {missing_columns}"
        )

    # Menghitung metrik seluruh sensor
    results = []

    for variable, true_column in VARIABLES.items():

        sensor_columns = [
            column
            for column in df.columns
            if column.startswith(f"{variable}_sensor_")
        ]

        if not sensor_columns:
            raise ValueError(
                f"Tidak ada sensor untuk variabel: {variable}"
            )

        for sensor_column in sensor_columns:

            metrics = calculate_sensor_metrics(
                df=df,
                variable=variable,
                true_column=true_column,
                sensor_column=sensor_column,
            )

            results.append(metrics)

    results_df = pd.DataFrame(results)

    # Ranking MAE dalam kelompok variabel yang sama
    results_df["MAE Rank"] = (
        results_df.groupby("Variable")["MAE"]
        .rank(method="min", ascending=True)
    )

    results_df["MAE Rank"] = (
        results_df["MAE Rank"].astype("Int64")
    )

    # Membulatkan nilai metrik
    results_df[["MAE", "RMSE", "Bias"]] = (
        results_df[["MAE", "RMSE", "Bias"]].round(4)
    )

    # Mengurutkan laporan
    results_df = results_df.sort_values(
        ["Variable", "MAE Rank"]
    ).reset_index(drop=True)

    # Menampilkan laporan
    print("=" * 75)
    print("DIMASA AI - MULTI-SENSOR ERROR ANALYSIS")
    print("=" * 75)

    print(f"\nTotal observations : {len(df)}")
    print(f"Total sensors      : {len(results_df)}")
    print(f"Variables analyzed : {len(VARIABLES)}")

    print("\nSENSOR PERFORMANCE REPORT")
    print("-" * 75)
    print(results_df.to_string(index=False))

    # Menentukan sensor terbaik untuk setiap variabel
    print("\nBEST SENSOR PER VARIABLE")
    print("-" * 75)

    for variable in VARIABLES:

        variable_results = results_df[
            results_df["Variable"] == variable
        ].dropna(subset=["MAE"])

        if variable_results.empty:
            print(f"{variable}: No valid sensor measurements")
            continue

        best_sensor = variable_results.loc[
            variable_results["MAE"].idxmin()
        ]

        print(f"\nVariable : {variable}")
        print(f"Sensor   : {best_sensor['Sensor']}")
        print(f"MAE      : {best_sensor['MAE']}")
        print(f"RMSE     : {best_sensor['RMSE']}")
        print(f"Bias     : {best_sensor['Bias']}")

    # Menyimpan laporan
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "sensor_error_analysis.csv"

    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 75)
    print(f"Report saved: {output_file}")
    print("DIMASA AI - ERROR ANALYSIS COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    main()
