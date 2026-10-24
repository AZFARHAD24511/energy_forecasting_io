from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.energy_io.data_loader import load_hourly_data, demo_io_matrix
from src.energy_io.forecasting import fit_and_evaluate
from src.energy_io.io_model import LeontiefEconomicModel
from src.energy_io.visualize import save_forecast_plot


def main() -> None:
    reports = ROOT / "reports"
    processed = ROOT / "data" / "processed"
    figures = reports / "figures"
    for folder in [reports, processed, figures]:
        folder.mkdir(parents=True, exist_ok=True)

    hourly, data_source = load_hourly_data()
    metrics, predictions, model, model_name = fit_and_evaluate(hourly)
    A, sectors = demo_io_matrix()
    io_model = LeontiefEconomicModel(A, sectors)

    # Scenario input is deliberately explicit and editable; not automatically equated
    # to the next-hour forecast because the IO matrix is annual and synthetic by default.
    scenario_change_pct = 0.20
    io_report = io_model.electricity_cost_scenario("Electricity", scenario_change_pct)

    hourly.to_csv(processed / "demo_hourly.csv", index=True)
    metrics.insert(0, "data_source", data_source)
    metrics.to_csv(reports / "metrics.csv", index=False)
    predictions.to_csv(reports / "price_forecast.csv", index=True)
    io_report.to_csv(reports / "io_price_scenario.csv", index=False)
    try:
        save_forecast_plot(predictions.tail(240), figures / "price_forecast_test.png")
    except ImportError as exc:
        print(f"Plot skipped because matplotlib is unavailable: {exc}")

    print(f"Data source: {data_source}")
    print(f"Forecast model: {model_name}")
    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print(f"\nIO scenario: electricity price shock = +{scenario_change_pct * 100:.1f}%")
    print(io_report.to_string(index=False))
    print(f"\nReports saved under: {reports}")


if __name__ == "__main__":
    main()
