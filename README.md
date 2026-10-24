# Energy Forecasting + Input–Output Analysis

A reproducible starter repository for hourly electricity-price forecasting and a separate annual input–output (IO) scenario analysis.

> **Data status:** The default run uses explicitly labelled synthetic demonstration data. It does not download real ENTSO-E, ERA5/Copernicus, or Eurostat FIGARO data. Replace the CSV inputs with real data before interpreting results economically.

## Project structure

```text
energy_forecasting_io/
├── data/
│   ├── raw/                 # User-provided source files (not committed)
│   └── processed/           # Generated/cleaned files (not committed by default)
├── models/                  # Trained model artifacts (not committed)
├── notebooks/
├── reports/
│   ├── figures/
│   └── metrics.csv
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── energy_io/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── forecasting.py
│       ├── io_model.py
│       └── visualize.py
├── tests/
│   └── test_io_model.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 1. Use an existing virtual environment (no automatic installation)

The PowerShell helper `setup_project.ps1` checks `D:\projects\.venv\Scripts\python.exe` first and reports which core packages are available. It does **not** install packages and does not change any other project. If that environment is unavailable, it prints a clear message instead of creating another environment.

Run from PowerShell after extracting this project into `D:\projects\energy_forecasting_io`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_project.ps1
```

To inspect the interpreter and packages directly:

```powershell
& D:\projects\.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.version)"
& D:\projects\.venv\Scripts\python.exe -m pip list
```

If the shared environment is missing a dependency, do not install blindly. Review the report first. Shared environments avoid repeated installations but can create dependency-version conflicts between projects.

## 2. Run the demonstration pipeline

```powershell
& D:\projects\.venv\Scripts\python.exe .\scripts\run_pipeline.py
```

The script writes:
- `reports/metrics.csv`
- `reports/price_forecast.csv`
- `reports/io_price_scenario.csv`
- `data/processed/demo_hourly.csv`

The forecaster uses a chronological train/validation/test split. The validation set is used for model choice; the test set is reserved for final evaluation. Lag features are constructed without shuffling time.

The model uses XGBoost if installed, otherwise scikit-learn's `HistGradientBoostingRegressor`. No dependency is installed automatically.

## 3. Supply real hourly data

Place an hourly CSV at `data/raw/hourly_energy.csv` with a timestamp column and these columns:

- `Temperature`
- `WindSpeed`
- `Load`
- `Price`

The timestamp column may be named `timestamp` or be a parseable first column. Units and timezone must be documented by the data provider. The current starter expects one row per hour and numeric values.

## 4. Supply a real IO matrix

The default IO matrix is a small synthetic example. To do a defensible analysis, load a square, consistently ordered technical-coefficients matrix `A` from a documented IO source (e.g. Eurostat FIGARO), and record:
- country/region, year, currency and price basis;
- sector concordance and ordering;
- whether rows are inputs and columns are producing sectors;
- treatment of imports, taxes, margins and electricity sector aggregation.

The IO matrix is annual while the forecast is hourly. This project does **not** pretend that hourly price forecasts directly produce a validated annual macroeconomic effect. Scenario inputs and temporal aggregation need to be specified before interpreting results.

## 5. Model interpretation

The Leontief quantity model `x = (I - A)^(-1) y` describes gross output associated with a final-demand vector `y`. It is not automatically a price model.

The separate price-cost scenario in `io_model.py` is an explicitly simplified, fixed-coefficient cost-propagation calculation. It assumes a defined upstream electricity-price shock and a stable technical-coefficients matrix. Treat results as scenario analysis, not causal estimates.

## 6. Tests

```powershell
& D:\projects\.venv\Scripts\python.exe -m pytest
```

## 7. GitHub

This folder is intended to be a **new, separate repository**. Do not reuse another project's remote. After creating a new empty GitHub repository, add its exact URL:

```powershell
git init
git add .
git commit -m "Initial energy forecasting and IO scaffold"
git branch -M main
git remote add origin <YOUR_NEW_REPOSITORY_URL>
git push -u origin main
```

Never commit `.venv`, API keys, tokens, large raw datasets, or private source files.
