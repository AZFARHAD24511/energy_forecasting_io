from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "germany_electricity_hourly.csv"
OUT = ROOT / "outputs" / "hourly_real_model"
TARGET = "price_eur_mwh"
HOLDOUT_DAYS = 30

EXOG = [
    "grid_load", "residual_load", "wind_offshore", "hydro", "biomass",
    "wind_onshore", "solar", "hard_coal", "pumped_storage",
    "natural_gas", "other_conventional", "other_renewables",
]


def metrics(y, prediction):
    return {
        "MAE_EUR_MWh": mean_absolute_error(y, prediction),
        "RMSE_EUR_MWh": np.sqrt(mean_squared_error(y, prediction)),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT)
    original_rows = len(raw)

    required = {"timestamp_utc", TARGET}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    raw["timestamp_utc"] = pd.to_datetime(
        raw["timestamp_utc"], errors="coerce", utc=True
    )
    raw[TARGET] = pd.to_numeric(raw[TARGET], errors="coerce")

    for col in EXOG:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    invalid_timestamps = int(raw["timestamp_utc"].isna().sum())
    raw = raw.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    before_dedup = len(raw)
    raw = raw.drop_duplicates("timestamp_utc", keep="last")
    duplicates_removed = before_dedup - len(raw)
    raw = raw.set_index("timestamp_utc")

    # Use a regular UTC hourly timeline to handle daylight-saving changes.
    hourly_index = pd.date_range(
        start=raw.index.min().floor("h"),
        end=raw.index.max().floor("h"),
        freq="h",
    )
    df = raw.reindex(hourly_index)
    df.index.name = "timestamp_utc"

    local = df.index.tz_convert("Europe/Berlin")
    df["hour"] = local.hour
    df["dayofweek"] = local.dayofweek
    df["month"] = local.month
    df["dayofyear"] = local.dayofyear
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    features = []

    # Historical price lags: all are strictly in the past.
    for lag in [1, 2, 3, 24, 48, 168, 336]:
        name = f"{TARGET}_lag_{lag}"
        df[name] = df[TARGET].shift(lag)
        features.append(name)

    # Lagged system load and generation variables.
    for col in [c for c in EXOG if c in df.columns]:
        for lag in [1, 24, 168]:
            name = f"{col}_lag_{lag}"
            df[name] = df[col].shift(lag)
            features.append(name)

    # Rolling statistics use observations only up to the previous hour.
    for window in [24, 168]:
        shifted_price = df[TARGET].shift(1)
        for stat in ["mean", "std"]:
            name = f"price_roll_{stat}_{window}"
            rolling = shifted_price.rolling(window, min_periods=window)
            df[name] = getattr(rolling, stat)()
            features.append(name)

    features += [
        "hour", "dayofweek", "month", "dayofyear", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "doy_sin", "doy_cos",
    ]

    cutoff = df.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train = df.loc[(df.index < cutoff) & df[TARGET].notna()].copy()
    test = df.loc[(df.index >= cutoff) & df[TARGET].notna()].copy()

    if len(train) < 24 * 30 or len(test) < 24 * 7:
        raise ValueError(
            f"Insufficient data: train={len(train)}, test={len(test)}"
        )

    X_train = train[features].replace([np.inf, -np.inf], np.nan)
    X_test = test[features].replace([np.inf, -np.inf], np.nan)
    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    model_predictions = model.predict(X_test)

    # Compare all methods on identical timestamps.
    common = (
        test[f"{TARGET}_lag_24"].notna()
        & test[f"{TARGET}_lag_168"].notna()
    )
    y_eval = y_test.loc[common]
    prediction_series = pd.Series(model_predictions, index=test.index)

    candidates = {
        "seasonal_naive_24h":
            test.loc[common, f"{TARGET}_lag_24"].to_numpy(dtype=float),
        "seasonal_naive_168h":
            test.loc[common, f"{TARGET}_lag_168"].to_numpy(dtype=float),
        "hist_gradient_boosting":
            prediction_series.loc[common].to_numpy(),
    }

    results = []
    for name, prediction in candidates.items():
        results.append({
            "model": name,
            "n_test_common": len(y_eval),
            **metrics(y_eval, prediction),
        })

    metrics_df = pd.DataFrame(results).sort_values("MAE_EUR_MWh")
    metrics_df.to_csv(OUT / "metrics.csv", index=False)

    predictions = pd.DataFrame({
        "actual_price_eur_mwh": y_test,
        "seasonal_naive_24h": test[f"{TARGET}_lag_24"],
        "seasonal_naive_168h": test[f"{TARGET}_lag_168"],
        "hist_gradient_boosting": model_predictions,
    }, index=test.index)
    predictions.index.name = "timestamp_utc"
    predictions.to_csv(OUT / "holdout_predictions.csv")

    joblib.dump({
        "pipeline": model,
        "feature_columns": features,
        "target": TARGET,
        "holdout_days": HOLDOUT_DAYS,
        "train_end_exclusive_utc": cutoff.isoformat(),
    }, OUT / "model.joblib")

    report = {
        "input_file": str(INPUT),
        "original_rows": original_rows,
        "invalid_timestamps": invalid_timestamps,
        "duplicate_timestamps_removed": duplicates_removed,
        "hourly_grid_rows": len(df),
        "observed_price_rows": int(df[TARGET].notna().sum()),
        "missing_price_hours": int(df[TARGET].isna().sum()),
        "start_utc": df.index.min().isoformat(),
        "end_utc": df.index.max().isoformat(),
        "test_start_utc": test.index.min().isoformat(),
        "test_end_utc": test.index.max().isoformat(),
        "train_rows": len(train),
        "test_rows": len(test),
        "common_comparison_rows": len(y_eval),
        "feature_count": len(features),
        "evaluation": (
            "Rolling one-step-ahead holdout. Actual prior observations "
            "within the test period are available for later predictions; "
            "this is not a recursive 30-day forecast."
        ),
        "caveats": [
            "Baseline model; further backtesting and tuning remain.",
            "Check whether the latest SMARD observations are complete.",
            "Only lagged exogenous variables are used.",
            "The annual FIGARO IO model is not yet connected to this forecast.",
        ],
    }
    (OUT / "run_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== REAL SMARD HOURLY PRICE MODEL ===")
    print(f"Training observations: {len(train):,}")
    print(f"Test observations: {len(test):,}")
    print(f"Features: {len(features)}")
    print(f"Test period UTC: {test.index.min()} -> {test.index.max()}")
    print("\nModel comparison (lower is better):")
    print(metrics_df.to_string(
        index=False, float_format=lambda value: f"{value:,.3f}"
    ))
    print(f"\nOutputs: {OUT}")
    print("Created: metrics.csv, holdout_predictions.csv, model.joblib, run_report.json")


if __name__ == "__main__":
    main()
