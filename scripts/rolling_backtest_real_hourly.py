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
DATA_PATH = ROOT / "data" / "raw" / "germany_electricity_hourly.csv"
OLD_HOLDOUT = ROOT / "outputs" / "hourly_real_model" / "holdout_predictions.csv"
OUT = ROOT / "outputs" / "hourly_real_model"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 4
TEST_DAYS = 30

# Read source data; preserve the original file.
df = pd.read_csv(DATA_PATH)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.drop_duplicates("timestamp_utc", keep="last")
df = df.sort_values("timestamp_utc").set_index("timestamp_utc")

target = "price_eur_mwh"
if target not in df.columns:
    raise ValueError(f"Target column not found: {target}")

# Convert numeric data and create a regular UTC hourly index.
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

hourly_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="h",
    tz="UTC",
)
df = df.reindex(hourly_index)
df.index.name = "timestamp_utc"

# Determine the beginning of the existing final holdout.
old = pd.read_csv(OLD_HOLDOUT, usecols=["timestamp_utc"])
old_start = pd.to_datetime(old["timestamp_utc"], utc=True).min()

# All rolling-backtest windows must end no later than old_start.
# Each fold tests 720 hours; training expands over time.
test_length = pd.Timedelta(days=TEST_DAYS)
fold_starts = [
    old_start - test_length * n
    for n in range(N_FOLDS, 0, -1)
]

# Features use past observations only.
features = pd.DataFrame(index=df.index)
price = df[target]

for lag in [1, 2, 3, 24, 48, 168, 336]:
    features[f"price_lag_{lag}"] = price.shift(lag)

shifted_price = price.shift(1)
for window in [24, 168]:
    features[f"price_roll_mean_{window}"] = (
        shifted_price.rolling(window, min_periods=window).mean()
    )
    features[f"price_roll_std_{window}"] = (
        shifted_price.rolling(window, min_periods=window).std()
    )

exogenous = [
    c for c in df.columns
    if c not in [target, "datetime_berlin"]
    and pd.api.types.is_numeric_dtype(df[c])
]
for col in exogenous:
    for lag in [1, 24, 168]:
        features[f"{col}_lag_{lag}"] = df[col].shift(lag)

local_time = df.index.tz_convert("Europe/Berlin")
hour = local_time.hour
dow = local_time.dayofweek
month = local_time.month

features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
features["dow_cos"] = np.cos(2 * np.pi * dow / 7)
features["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
features["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
features["is_weekend"] = (dow >= 5).astype(int)

# Compare all models on exactly the same test hours.
all_predictions = []
all_metrics = []

for fold_no, start in enumerate(fold_starts, start=1):
    end = start + test_length

    train_mask = (features.index < start) & price.notna()
    test_mask = (
        (features.index >= start)
        & (features.index < end)
        & price.notna()
    )

    train_idx = features.index[train_mask]
    test_idx = features.index[test_mask]

    # Naive baselines require both historical prices to be available.
    common = (
        price.loc[test_idx].notna()
        & price.shift(24).loc[test_idx].notna()
        & price.shift(168).loc[test_idx].notna()
    )
    test_idx = test_idx[common.to_numpy()]

    if len(train_idx) < 1000 or len(test_idx) < 500:
        raise ValueError(
            f"Insufficient data in fold {fold_no}: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])

    model.fit(features.loc[train_idx], price.loc[train_idx])

    actual = price.loc[test_idx]
    predictions = {
        "hist_gradient_boosting": model.predict(features.loc[test_idx]),
        "seasonal_naive_24h": price.shift(24).loc[test_idx].to_numpy(),
        "seasonal_naive_168h": price.shift(168).loc[test_idx].to_numpy(),
    }

    local_test = test_idx.tz_convert("Europe/Berlin")
    print(
        f"Fold {fold_no}: train through {start}, "
        f"test [{start}, {end}), train rows={len(train_idx)}, "
        f"test hours={len(test_idx)}"
    )

    for model_name, pred in predictions.items():
        error = pred - actual.to_numpy()
        metrics = {
            "fold": fold_no,
            "test_start_utc": start.isoformat(),
            "test_end_utc_exclusive": end.isoformat(),
            "model": model_name,
            "n_test": len(actual),
            "MAE_EUR_MWh": mean_absolute_error(actual, pred),
            "RMSE_EUR_MWh": np.sqrt(mean_squared_error(actual, pred)),
            "Bias_EUR_MWh": float(np.mean(error)),
        }
        all_metrics.append(metrics)

        all_predictions.append(pd.DataFrame({
            "timestamp_utc": test_idx,
            "datetime_berlin": local_test.astype(str),
            "fold": fold_no,
            "actual_price_eur_mwh": actual.to_numpy(),
            "model": model_name,
            "prediction_eur_mwh": pred,
        }))

metrics_df = pd.DataFrame(all_metrics)
predictions_df = pd.concat(all_predictions, ignore_index=True)

metrics_path = OUT / "rolling_backtest_metrics.csv"
predictions_path = OUT / "rolling_backtest_predictions.csv"
report_path = OUT / "rolling_backtest_report.json"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

summary = (
    metrics_df.groupby("model")
    .agg(
        folds=("fold", "nunique"),
        total_test_hours=("n_test", "sum"),
        mean_fold_MAE=("MAE_EUR_MWh", "mean"),
        mean_fold_RMSE=("RMSE_EUR_MWh", "mean"),
        mean_fold_Bias=("Bias_EUR_MWh", "mean"),
    )
    .sort_values("mean_fold_MAE")
    .round(3)
)

report = {
    "data_file": str(DATA_PATH),
    "existing_holdout_start_utc": old_start.isoformat(),
    "method": "Expanding-window, rolling one-step-ahead backtest",
    "test_days_per_fold": TEST_DAYS,
    "number_of_folds": N_FOLDS,
    "feature_count": int(features.shape[1]),
    "feature_policy": (
        "Price and exogenous variables use lags only; "
        "rolling price statistics are shifted by one hour."
    ),
    "summary": summary.reset_index().to_dict(orient="records"),
}

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\nSUMMARY: mean metrics across folds")
print(summary.to_string())
print(f"\nSaved metrics:     {metrics_path}")
print(f"Saved predictions: {predictions_path}")
print(f"Saved report:      {report_path}")
