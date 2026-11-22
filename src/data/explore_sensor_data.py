
"""
DIMASA AI
Environmental Sensor Data Exploration

Analyzes:
1. Dataset structure and information
2. Data types
3. Descriptive statistics
4. Missing values
5. Duplicate records
6. Timestamp range
"""

from pathlib import Path
import pandas as pd


def main():
    # Menentukan lokasi project
    project_root = Path(__file__).resolve().parents[2]

    # Membaca dataset hasil simulasi
    data_path = (
        project_root
        / "data"
        / "simulated"
        / "environmental_sensors.csv"
    )

    df = pd.read_csv(data_path)

    print("=" * 60)
    print("DIMASA AI - DATA EXPLORATION")
    print("=" * 60)

    # 1. Informasi dataset
    print("\n[1] DATASET INFORMATION")
    print("-" * 40)
    print(f"Total rows    : {df.shape[0]}")
    print(f"Total columns : {df.shape[1]}")
    print(f"Memory usage  : {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # 2. Tipe data
    print("\n[2] DATA TYPES")
    print("-" * 40)
    print(df.dtypes)

    # 3. Statistik deskriptif
    print("\n[3] DESCRIPTIVE STATISTICS")
    print("-" * 40)
    print(df.describe().round(2).to_string())

    # 4. Missing values
    print("\n[4] MISSING VALUES")
    print("-" * 40)

    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100

    missing_report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage.round(2)
    })

    print(missing_report.to_string())

    # 5. Duplikasi data
    print("\n[5] DUPLICATE DATA")
    print("-" * 40)
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # 6. Rentang waktu data
    print("\n[6] TIMESTAMP RANGE")
    print("-" * 40)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Start : {df['timestamp'].min()}")
    print(f"End   : {df['timestamp'].max()}")

    print("\n" + "=" * 60)
    print("DIMASA AI - DATA EXPLORATION COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
