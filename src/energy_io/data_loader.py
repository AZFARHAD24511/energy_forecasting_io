from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["Temperature", "WindSpeed", "Load", "Price_Lag1", "Load_Lag1"]


def make_demo_data(periods: int = 24 * 90, seed: int = 2026) -> pd.DataFrame:
    """Create clearly synthetic hourly data for software testing, not empirical analysis."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=periods, freq="h")
    hour = idx.hour.to_numpy()
    day = idx.dayofyear.to_numpy()
    temperature = 10 + 9 * np.sin(2 * np.pi * (day - 100) / 365) + rng.normal(0, 3, periods)
    temperature += 2 * np.sin(2 * np.pi * hour / 24)
    wind = np.clip(rng.weibull(2.0, periods) * 7, 0, 25)
    load = (
        3900 + 35 * (temperature - 16) ** 2
        + 170 * np.sin(2 * np.pi * (hour - 8) / 24)
        + rng.normal(0, 100, periods)
    )
    price = 22 + 0.018 * load - 0.8 * wind + 0.15 * np.maximum(load - 4700, 0)
    price += rng.normal(0, 5, periods)
    frame = pd.DataFrame(
        {"Temperature": temperature, "WindSpeed": wind, "Load": load, "Price": price},
        index=idx,
    )
    frame.index.name = "timestamp"
    return add_lag_features(frame)


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Temperature", "WindSpeed", "Load", "Price"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Hourly data is missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["Price_Lag1"] = out["Price"].shift(1)
    out["Load_Lag1"] = out["Load"].shift(1)
    return out.dropna().sort_index()


def load_hourly_data(raw_path: str | Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load a real CSV when present; otherwise return labelled synthetic demo data."""
    if raw_path is None:
        raw_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "hourly_energy.csv"
    path = Path(raw_path)
    if not path.exists():
        return make_demo_data(), "synthetic_demo"
    frame = pd.read_csv(path)
    timestamp_col = next((c for c in frame.columns if c.lower() in {"timestamp", "datetime", "date_time"}), None)
    if timestamp_col is not None:
        frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="raise")
        frame = frame.set_index(timestamp_col)
    else:
        first = frame.columns[0]
        parsed = pd.to_datetime(frame[first], errors="coerce")
        if parsed.notna().all():
            frame[first] = parsed
            frame = frame.set_index(first)
        else:
            raise ValueError("CSV needs a timestamp/datetime/date_time column or a parseable first column.")
    for col in ["Temperature", "WindSpeed", "Load", "Price"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame.index.name = "timestamp"
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Duplicate timestamps found in hourly_energy.csv.")
    return add_lag_features(frame), "user_csv"


def demo_io_matrix() -> tuple[np.ndarray, list[str]]:
    """Synthetic A matrix: rows are input sectors, columns are producing sectors."""
    sectors = ["Agriculture", "Heavy_Industry", "Services", "Electricity"]
    A = np.array([
        [0.10, 0.08, 0.03, 0.02],
        [0.12, 0.20, 0.07, 0.10],
        [0.06, 0.09, 0.18, 0.04],
        [0.04, 0.12, 0.05, 0.08],
    ], dtype=float)
    return A, sectors
