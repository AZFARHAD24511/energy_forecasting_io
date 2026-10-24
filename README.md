# German Electricity Price Forecasting & Input–Output Analysis

A data-driven research project combining **hourly German wholesale electricity price forecasting** with a separate **input–output (IO) economic analysis**. The forecasting component uses real hourly electricity-market data, time-dependent features, gradient boosting, and chronological backtesting.

> **Current status:** The real-data forecasting pipeline and four-fold rolling backtest have been evaluated. The reported results represent rolling one-step-ahead forecasting, not a fully independent day-ahead market forecast.

## Project Overview

The project investigates how historical electricity prices, intraday patterns, and lagged electricity-system variables can support hourly wholesale price forecasting in Germany.

The repository also contains a separate input–output analysis framework. The hourly forecasting and annual IO components are distinct analytical tasks; their outputs should not be interpreted as a validated causal link between hourly electricity prices and annual macroeconomic effects.

### Main objectives

- Build an hourly electricity-price forecasting pipeline using real data.
- Compare a machine-learning model against seasonal-naive benchmarks.
- Evaluate performance using chronological, expanding-window backtesting.
- Interpret predictive features using permutation importance.
- Maintain a reproducible workflow for further electricity-market and IO research.

## Data

The real hourly dataset contains **59,735 observations and 15 columns**, covering approximately December 2019 to October 2026.

The dataset includes:

- Wholesale electricity price (`price_eur_mwh`)
- Grid load and residual load
- Wind, solar, hydro, and biomass variables
- Conventional generation variables, including hard coal and natural gas
- UTC timestamps and Berlin-local timestamps

The forecasting pipeline uses UTC timestamps to maintain a consistent hourly timeline across daylight-saving-time transitions.

**Data handling:** Raw and processed datasets are excluded from Git by default. Users should obtain the data from its original provider and follow the applicable terms of use. The repository should document the exact source URL and retrieval date when these details have been verified.

## Methodology

### Feature engineering

The model uses 54 features, including:

- Electricity-price lags: 1, 2, 3, 24, 48, 168, and 336 hours
- Lagged electricity-system variables
- Past-only rolling price statistics
- Cyclical calendar features for hour of day, day of week, and month

Price and exogenous variables are lagged, and rolling price statistics are shifted to avoid directly incorporating the target hour's price into its own predictors.

### Models compared

1. **HistGradientBoostingRegressor** — the machine-learning model.
2. **Seasonal Naive (24h)** — uses the price from 24 hours earlier.
3. **Seasonal Naive (168h)** — uses the price from 168 hours earlier.

### Evaluation design

An expanding-window rolling backtest evaluates four consecutive 30-day test periods. Each fold contains 720 hourly observations, giving **2,880 test observations across four folds**.

The model is evaluated using:

- **MAE (Mean Absolute Error):** average absolute forecasting error.
- **RMSE (Root Mean Squared Error):** an error measure that penalizes large errors more heavily.
- **Bias:** the signed average prediction error, subject to the sign convention used in the evaluation code.

All reported errors are in euros per megawatt-hour (€/MWh).

## Results

### Four-fold rolling backtest

| Model | Mean fold MAE (€/MWh) | Mean fold RMSE (€/MWh) |
|---|---:|---:|
| HistGradientBoostingRegressor | **9.231** | **16.025** |
| Seasonal Naive (24h) | 33.525 | 52.361 |
| Seasonal Naive (168h) | 39.273 | 60.281 |

Across these four evaluation windows, the gradient-boosting model achieved lower MAE and RMSE than both seasonal-naive benchmarks.

Its mean fold MAE was approximately **72.5% lower** than the 24-hour seasonal-naive benchmark. This is a relative reduction in MAE on these evaluation windows, not a guarantee of equivalent improvement in live trading or day-ahead forecasting.

![Rolling backtest model comparison](docs/figures/rolling_backtest_mae.png)

### Forecasting performance across time

The four folds help assess whether model performance is stable across different periods rather than relying on one test window alone.

The fourth fold, covering 25 August to 24 September 2026 (UTC), had an MAE of **11.468 €/MWh**. This was higher than the overall mean fold MAE, illustrating that forecasting accuracy varies over time.

![Rolling backtest performance by fold](docs/figures/rolling_backtest_by_fold.png)

### Feature importance

Permutation importance was calculated on the fourth fold's test set, containing 720 hours. The baseline MAE for that fold was approximately 11.468 €/MWh.

The most influential features were:

| Feature | Mean permutation importance |
|---|---:|
| `price_lag_1` | 76.446 |
| `hour_sin` | 6.122 |
| `price_lag_2` | 3.390 |
| `hour_cos` | 2.197 |
| `solar_lag_1` | 2.041 |
| `solar_lag_24` | 1.485 |

Importance values represent the increase in MAE after a feature is permuted; they are not coefficients, percentages, or causal effects.

![Permutation feature importance](docs/figures/permutation_importance_fold4.png)

#### Interpretation

**1. Strong short-term price persistence**

The previous hour's price (`price_lag_1`) dominates the permutation-importance results. Recent prices therefore carry substantial predictive information in this evaluation setting.

**2. Intraday patterns matter**

The hour-of-day features (`hour_sin` and `hour_cos`) are the next most important group, indicating that the model makes substantial use of recurring hourly patterns.

**3. Solar generation contributes predictive information**

Lagged solar variables rank among the more important system features. This is consistent with the possibility that solar-generation patterns help explain price variation, but permutation importance alone cannot establish a causal relationship.

**4. Correlated features require caution**

When predictors contain overlapping information, permutation importance may redistribute or understate their individual contributions. These results should be interpreted as model-specific predictive importance, not as a ranking of causal drivers of electricity prices.

## Limitations and next steps

- **Rolling one-step-ahead evaluation:** Actual prices from earlier hours in each test period are available as lagged inputs for subsequent predictions. This is not equivalent to forecasting the next 24 hours without observing any prices during that horizon.
- **Day-ahead forecasting:** A production-oriented day-ahead evaluation should reproduce the actual forecast-issuance time and restrict every input to information available at that time.
- **Extreme prices:** Large price spikes are harder to predict accurately. Performance should also be reported separately for negative prices and high-price periods.
- **Exogenous data availability:** Future load and generation values cannot be used unless they would have been available as forecasts at the time of prediction.
- **Permutation importance:** Importance is model- and evaluation-period-specific and does not establish causality.
- **Input–output analysis:** Annual IO results require separate assumptions and validation. They should not be presented as causal estimates derived directly from hourly price predictions.

## Reproducibility

The real-data workflow is implemented in Python. It includes data-validation, model-training, rolling-backtest, and feature-importance scripts.

Use the project's documented Python environment and install the dependencies listed in `requirements.txt`.

Example commands from the repository root in PowerShell:

```powershell
python .\scripts\validate_real_data.py
python .\scripts\train_real_hourly_model.py
python .\scripts\rolling_backtest_real_hourly.py
python .\scripts\feature_importance_real_hourly.py
```

These commands assume that the scripts' configured input paths and dependencies are available. Review each script before running it, especially if your local data paths differ.

## Repository structure

```text
energy_forecasting_io/
├── data/                 # Raw and processed data; excluded by default
├── docs/
│   └── figures/          # README figures
├── models/               # Model artifacts; excluded by default
├── notebooks/
├── outputs/              # Generated forecasting outputs; excluded by default
├── reports/
├── scripts/
│   ├── train_real_hourly_model.py
│   ├── rolling_backtest_real_hourly.py
│   ├── feature_importance_real_hourly.py
│   ├── validate_real_data.py
│   └── run_pipeline.py
├── src/energy_io/
├── tests/
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Conclusion

The current results show that a gradient-boosting model can outperform simple seasonal-naive benchmarks in the evaluated German hourly electricity-price forecasting windows. Recent prices and intraday timing provide the strongest predictive signals in the feature-importance analysis.

The next research step is to evaluate a realistic day-ahead forecasting setup, improve the treatment of extreme prices, and document data provenance and forecast-time information constraints. The separate IO framework can then be developed and validated on its own terms.
