from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "hourly_real_model"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Consistent labels for publication-quality charts.
LABELS = {
    "hist_gradient_boosting": "Gradient Boosting",
    "seasonal_naive_24h": "Seasonal Naive (24h)",
    "seasonal_naive_168h": "Seasonal Naive (168h)",
}

COLORS = {
    "hist_gradient_boosting": "#2878B5",
    "seasonal_naive_24h": "#E69F00",
    "seasonal_naive_168h": "#777777",
}

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Created: {FIGURES / filename}")


# 1. MAE across the four rolling backtest folds.
metrics = pd.read_csv(RESULTS / "rolling_backtest_metrics.csv")
metrics = metrics.sort_values(["fold", "model"])

fig, ax = plt.subplots(figsize=(10, 5.5))
folds = sorted(metrics["fold"].unique())
models = [
    "hist_gradient_boosting",
    "seasonal_naive_24h",
    "seasonal_naive_168h",
]
width = 0.24
positions = list(range(len(folds)))

for i, model in enumerate(models):
    subset = (
        metrics[metrics["model"] == model]
        .set_index("fold")
        .reindex(folds)
    )
    offsets = [p + (i - 1) * width for p in positions]
    bars = ax.bar(
        offsets,
        subset["MAE_EUR_MWh"],
        width,
        label=LABELS[model],
        color=COLORS[model],
    )
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

ax.set_title("Rolling Backtest: MAE by 30-Day Fold")
ax.set_xlabel("Test fold")
ax.set_ylabel("MAE (€/MWh)")
ax.set_xticks(positions)
ax.set_xticklabels([f"Fold {f}" for f in folds])
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
save_figure("rolling_backtest_mae.png")


# 2. RMSE across folds.
fig, ax = plt.subplots(figsize=(10, 5.5))

for i, model in enumerate(models):
    subset = (
        metrics[metrics["model"] == model]
        .set_index("fold")
        .reindex(folds)
    )
    offsets = [p + (i - 1) * width for p in positions]
    bars = ax.bar(
        offsets,
        subset["RMSE_EUR_MWh"],
        width,
        label=LABELS[model],
        color=COLORS[model],
    )
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

ax.set_title("Rolling Backtest: RMSE by 30-Day Fold")
ax.set_xlabel("Test fold")
ax.set_ylabel("RMSE (€/MWh)")
ax.set_xticks(positions)
ax.set_xticklabels([f"Fold {f}" for f in folds])
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
save_figure("rolling_backtest_rmse.png")


# 3. Top 15 permutation-importance features on Fold 4.
importance = pd.read_csv(
    RESULTS / "permutation_importance_fold4.csv"
)
importance = importance.sort_values(
    "importance_mean", ascending=False
).head(15).sort_values("importance_mean")

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(
    importance["feature"],
    importance["importance_mean"],
    xerr=importance["importance_std"],
    color="#2878B5",
    alpha=0.9,
    capsize=3,
)
ax.set_title("Top 15 Features: Permutation Importance (Fold 4)")
ax.set_xlabel("Increase in MAE after permutation (€/MWh)")
ax.set_ylabel("Feature")
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)
save_figure("permutation_importance_fold4.png")


# 4. Actual vs predicted prices over the last seven days of final holdout.
predictions = pd.read_csv(RESULTS / "holdout_predictions.csv")
predictions["timestamp_utc"] = pd.to_datetime(
    predictions["timestamp_utc"], utc=True
)
predictions = predictions.sort_values("timestamp_utc")
end_time = predictions["timestamp_utc"].max()
start_time = end_time - pd.Timedelta(days=7)
recent = predictions[predictions["timestamp_utc"] >= start_time].copy()

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(
    recent["timestamp_utc"],
    recent["actual_price_eur_mwh"],
    label="Actual price",
    color="#222222",
    linewidth=1.8,
)
ax.plot(
    recent["timestamp_utc"],
    recent["hist_gradient_boosting"],
    label="Gradient Boosting",
    color="#2878B5",
    linewidth=1.5,
)
ax.plot(
    recent["timestamp_utc"],
    recent["seasonal_naive_24h"],
    label="Seasonal Naive (24h)",
    color="#E69F00",
    linewidth=1.0,
    alpha=0.85,
)
ax.set_title("Actual vs Predicted Electricity Prices: Final Holdout")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Wholesale price (€/MWh)")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
ax.tick_params(axis="x", rotation=25)
save_figure("price_forecast_holdout.png")


print("\nAll README figures were generated successfully.")
