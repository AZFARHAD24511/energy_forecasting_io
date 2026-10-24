from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_loader import FEATURE_COLUMNS


def _new_model():
    """Prefer XGBoost when installed; otherwise use a scikit-learn model."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=2026,
            n_jobs=-1,
        ), "XGBoost"
    except ImportError:
        return HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.06, max_leaf_nodes=20, random_state=2026
        ), "HistGradientBoostingRegressor"


def chronological_split(frame: pd.DataFrame):
    """60% train, 20% validation, 20% test; never shuffle time."""
    n = len(frame)
    if n < 50:
        raise ValueError("At least 50 hourly rows are required for a time-based split.")
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return frame.iloc[:train_end], frame.iloc[train_end:val_end], frame.iloc[val_end:]


def fit_and_evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, object, str]:
    required = FEATURE_COLUMNS + ["Price"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for forecasting: {sorted(missing)}")
    train, valid, test = chronological_split(frame)
    model, model_name = _new_model()
    model.fit(train[FEATURE_COLUMNS], train["Price"])

    # Validation is reported separately; final test is not used for model fitting.
    valid_pred = model.predict(valid[FEATURE_COLUMNS])
    test_pred = model.predict(test[FEATURE_COLUMNS])

    def metrics(actual, pred, split):
        return {
            "split": split,
            "n": len(actual),
            "MAE": float(mean_absolute_error(actual, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "R2": float(r2_score(actual, pred)),
        }

    metrics_frame = pd.DataFrame([
        {"model": model_name, **metrics(valid["Price"], valid_pred, "validation")},
        {"model": model_name, **metrics(test["Price"], test_pred, "test")},
    ])
    predictions = pd.DataFrame(
        {
            "actual_price": test["Price"].to_numpy(),
            "predicted_price": test_pred,
        },
        index=test.index,
    )
    predictions.index.name = "timestamp"
    return metrics_frame, predictions, model, model_name
