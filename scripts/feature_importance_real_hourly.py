
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "germany_electricity_hourly.csv"
OLD_HOLDOUT = ROOT / "outputs" / "hourly_real_model" / "holdout_predictions.csv"
OUT = ROOT / "outputs" / "hourly_real_model"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "price_eur_mwh"
TEST_DAYS = 30
N_REPEATS = 5

# Read original data without modifying it.
df = pd.read_csv(DATA_PATH)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.drop_duplicates("timestamp_utc", keep="last")
df = df.sort_values("timestamp_utc").set_index("timestamp_utc")

for col in df.columns:
    if col != "datetime_berlin":
        df[col] = pd.to_numeric(df[col], errors="coerce")

hourly_index = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="h",
    tz="UTC",
)
df = df.reindex(hourly_index)
df.index.name = "timestamp_utc"

price = df[TARGET]

# Reconstruct the same features used by the rolling backtest.
features = pd.DataFrame(index=df.index)

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
    if c not in [TARGET, "datetime_berlin"]
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

# Reproduce Fold 4: the 30 days immediately before the old holdout.
old = pd.read_csv(OLD_HOLDOUT, usecols=["timestamp_utc"])
old_start = pd.to_datetime(old["timestamp_utc"], utc=True).min()
test_start = old_start - pd.Timedelta(days=TEST_DAYS)
test_end = old_start

train_idx = features.index[
    (features.index < test_start) & price.notna()
]
test_idx = features.index[
    (features.index >= test_start)
    & (features.index < test_end)
    & price.notna()
]

# Match the same hours available to both seasonal-naive baselines.
valid = (
    price.loc[test_idx].notna()
    & price.shift(24).loc[test_idx].notna()
    & price.shift(168).loc[test_idx].notna()
)
test_idx = test_idx[valid.to_numpy()]

X_train = features.loc[train_idx]
y_train = price.loc[train_idx]
X_test = features.loc[test_idx]
y_test = price.loc[test_idx]

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

model.fit(X_train, y_train)
pred = model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, pred)

print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Test period: {test_start} to {test_end}")
print(f"Fold 4 baseline MAE: {baseline_mae:.3f} EUR/MWh")
print("Calculating permutation importance...")

result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=N_REPEATS,
    random_state=42,
    n_jobs=-1,
)

importance = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
})

importance = importance.sort_values(
    "importance_mean", ascending=False
).reset_index(drop=True)

importance["importance_mean"] = importance["importance_mean"].round(4)
importance["importance_std"] = importance["importance_std"].round(4)

out_path = OUT / "permutation_importance_fold4.csv"
importance.to_csv(out_path, index=False)

report = {
    "fold": 4,
    "test_start_utc": test_start.isoformat(),
    "test_end_utc_exclusive": test_end.isoformat(),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "baseline_MAE_EUR_MWh": float(baseline_mae),
    "method": "Permutation importance on Fold 4 test data",
    "n_repeats": N_REPEATS,
    "top_20_features": importance.head(20).to_dict(orient="records"),
}

report_path = OUT / "permutation_importance_fold4_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\nTOP 20 FEATURES")
print(importance.head(20).to_string(index=False))
print(f"\nSaved importance: {out_path}")
print(f"Saved report:     {report_path}")
