# FinCast Inference Pipeline

A complete toolkit for fetching financial data, optimizing forecast models, and visualizing predictions.

## 🚀 Quick Start

The main entry point is `Inference/run.py`. It automates the entire flow:

1. **Fetch** data from Refinitiv Workspace.
2. **Predict** future prices using FinCast.
3. **Visualize** results (terminal table + plots).

```powershell
# Basic: Forecast Apple (Daily, last 5 years)
uv run Inference\run.py AAPL.O
```

---

## 🧪 Usage Examples

### 1. Intraday & High-Frequency

```powershell
# 1-minute bars for Apple, last 10 days
uv run Inference\run.py AAPL.O --freq 1min --days 10

# 5-minute bars for EUR/USD, last 5 days
uv run Inference\run.py EUR= --freq 5min --days 5

# Hourly bars for Microsoft, last 90 days
uv run Inference\run.py MSFT.O --freq 1h --days 90
```

### 2. Custom Model Settings

Override the Context Length (L) and Forecast Horizon (H):

```powershell
# Look back 256 days (L), predict next 32 days (H)
uv run Inference\run.py NVDA.O -L 256 -H 32
```

### 3. "Power User" Optimization

Run a rigorous optimization search on a specific date range:

```powershell
# Optimize model for Google (Daily), using 50 trials and 5-fold backtesting
uv run Inference\run.py GOOG.O --optimize --trials 50 --folds 5 --start 2020-01-01 --end 2024-12-31
```

### 4. Skip Data Fetching (Re-run Analysis)

If you already fetched the data and just want to tweak the inference:

```powershell
# Reuse existing data.csv, but change Context Length to 64
uv run Inference\run.py AAPL.O --skip-fetch -L 64 -H 16
```

### 5. Visualization Only

Just show me the plots for the last run again (no new inference):

```powershell
uv run Inference\run.py AAPL.O --skip-fetch --skip-inference
```

---

## 📖 Flag Dictionary

### Main Arguments

| Flag    | Value   | Default            | Description                                                           |
| ------- | ------- | ------------------ | --------------------------------------------------------------------- |
| `ric` | `str` | **Required** | The Refinitiv Instrument Code (e.g.,`AAPL.O`, `EUR=`, `VOD.L`). |

### Data Fetching Control

| Flag             | Value          | Default   | Description                                                                                |
| ---------------- | -------------- | --------- | ------------------------------------------------------------------------------------------ |
| `--freq`       | `str`        | `d`     | Data frequency. Options:`tick`, `1min` (t), `5min` (t), `1h`, `d`, `w`, `m`. |
| `--days`       | `int`        | `None`  | Fetch last N days of data.                                                                 |
| `--years`      | `int`        | `5`     | Fetch last N years of data.                                                                |
| `--start`      | `YYYY-MM-DD` | `None`  | Specific start date.                                                                       |
| `--end`        | `YYYY-MM-DD` | `Today` | Specific end date.                                                                         |
| `--skip-fetch` | `bool`       | `False` | Reuse existing `Inference/data.csv` instead of downloading new data.                     |

### Model Configuration

| Flag                 | Value    | Default           | Description                                                                                 |
| -------------------- | -------- | ----------------- | ------------------------------------------------------------------------------------------- |
| `-L`               | `int`  | `128 (32-1024)` | **Context Length**: Number of past data points the model "sees" to make a prediction. |
| `-H`               | `int`  | `16 (1-256)`    | **Forecast Horizon**: Number of future steps to predict.                              |
| `--optimize`       | `bool` | `False`         | Enable hyperparameter optimization (automl) to find best L and H.                           |
| `--trials`         | `int`  | `20`            | Number of Optuna trials to run during optimization. Higher = better results but slower.     |
| `--folds`          | `int`  | `3`             | Number of cross-validation folds during optimization.                                       |
| `--cpu`            | `bool` | `False`         | Force using CPU even if a GPU is available.                                                 |
| `--skip-inference` | `bool` | `False`         | Skip the model run entirely; primarily used to re-visualize old results.                    |

---

## 📂 Output Files

All generated outputs are saved in `Inference/output/[Ric]_[Frequency]_[Date]` (auto-created), keeping the root `Inference/` directory clean:

```
Inference/
├── data.csv                              ← Input data from Refinitiv
├── v1.pth                                ← Model weights
├── run.py                                ← Main executor
└── output/[Ric]_[Frequency]_[Date]
    ├── fincast_mean_YYYYMMDD_HHMMSS.csv  ← Mean forecast values
    ├── fincast_full_YYYYMMDD_HHMMSS.csv  ← Full forecast (Mean + Q1-Q9)
    ├── fincast_full_..._forecast.png     ← 📊 Fan Chart (forecast only)
    ├── fincast_full_..._context.png      ← 📈 Context + Forecast (full view)
    ├── fincast_full_..._zoomed.png       ← 🔍 Zoomed (last 10 points + forecast)
    └── best_config.json                  ← Optimal L, H from --optimize
```
