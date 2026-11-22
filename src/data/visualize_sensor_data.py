
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # DIMASA AI - Project Configuration
    project_root = Path(__file__).resolve().parents[2]

    # Membaca dataset
    data_path = (
        project_root
        / "data"
        / "simulated"
        / "environmental_sensors.csv"
    )

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Menyiapkan folder output grafik
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Daftar sensor dan ground truth
    sensors = [
        ("temperature_true", "temperature_sensor", "Temperature (°C)"),
        ("humidity_true", "humidity_sensor", "Humidity (%)"),
        ("pressure_true", "pressure_sensor", "Pressure (hPa)"),
        ("pm25_true", "pm25_sensor", "PM2.5 (µg/m³)")
    ]

    # Membuat satu grafik untuk setiap sensor
    for true_col, sensor_col, title in sensors:
        plt.figure(figsize=(12, 5))

        plt.plot(
            df["timestamp"],
            df[true_col],
            label="Ground Truth",
            linewidth=1.5
        )

        plt.plot(
            df["timestamp"],
            df[sensor_col],
            label="Sensor Reading",
            alpha=0.7,
            linewidth=1
        )

        plt.title(f"DIMASA AI | {title} - Ground Truth vs Sensor")
        plt.xlabel("Timestamp")
        plt.ylabel(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Menyimpan grafik
        filename = f"{sensor_col}_comparison.png"
        output_path = output_dir / filename

        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"[DIMASA AI] Saved: {output_path}")

    print("\nDIMASA AI - All sensor visualizations completed!")


if __name__ == "__main__":
    main()
