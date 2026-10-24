from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_forecast_plot(predictions: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    predictions[["actual_price", "predicted_price"]].plot(ax=ax)
    ax.set_title("Electricity price forecast — test period")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price (source units)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
