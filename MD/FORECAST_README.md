# FinOracle - AI-Powered Forecasting Pipeline

A comprehensive **multi-model forecasting engine** that combines classical time-series methods with cutting-edge deep learning to deliver robust, ensemble-based predictions.

## 🎯 Overview

FinOracle integrates **7 state-of-the-art forecasting models** into a unified pipeline:

1. **ARIMAX** - Classical statistical model with exogenous variables
2. **LSTM** - Long Short-Term Memory (RNN-based)
3. **GRU** - Gated Recurrent Unit (faster RNN variant)
4. **XGBoost** - Gradient boosting for time-series
5. **Random Forest** - Ensemble tree-based regression
6. **Transformer** - Multi-head attention mechanism
7. **FTS (FinCast)** - Foundation model with 4GB pretrained weights

All models can be run individually or combined via **ensemble averaging** for maximum robustness.

---

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default models (ARIMAX, LSTM, FTS)
python run_analysis.py --finoracle

# Run from screener
python run_from_screener.py --finoracle --top 5
```

### Custom Model Selection

```bash
# Run specific models
python run_analysis.py --finoracle --fo-models arimax lstm gru fts

# Run all 7 models + ensemble
python run_analysis.py --finoracle --fo-models arimax lstm gru xgboost random_forest transformer fts --fo-ensemble
```

### Custom Horizon

```bash
# 30-day forecast (applies to ALL models)
python run_analysis.py --finoracle --fo-horizon 30

# 90-day forecast with FTS optimization
python run_analysis.py --finoracle --fo-horizon 90 --fo-optimize
```

---

## 📊 Model Comparison

| Model              | Type                | Training Time | Inference Speed | Best For                          |
| ------------------ | ------------------- | ------------- | --------------- | --------------------------------- |
| **ARIMAX**         | Statistical         | ⚡ Fast        | ⚡ Instant       | Stable trends, seasonality        |
| **LSTM**           | Deep Learning (RNN) | 🐢 Slow        | ⚡ Fast          | Long-term dependencies            |
| **GRU**            | Deep Learning (RNN) | 🚀 Medium      | ⚡ Fast          | Shorter sequences, faster than LSTM |
| **XGBoost**        | Gradient Boosting   | 🚀 Medium      | ⚡ Fast          | Non-linear patterns               |
| **Random Forest**  | Ensemble (Trees)    | 🚀 Medium      | ⚡ Fast          | Noisy data, outliers              |
| **Transformer**    | Attention-based     | 🐢 Slow        | 🚀 Medium        | Complex patterns, multi-scale     |
| **FTS (FinCast)**  | Foundation Model    | N/A (Pretrained) | 🐢 Slow (GPU)  | Probabilistic, financial data     |
| **Ensemble**       | Meta-model          | N/A            | ⚡ Instant       | Robust consensus                  |

---

## 🛠️ CLI Parameters

### Data Fetching (Refinitiv)

| Flag               | Default | Description                                      |
| ------------------ | ------- | ------------------------------------------------ |
| `--fo-freq`        | `d`     | Data frequency (tick, 1min, 5min, 1h, d, w, m)  |
| `--fo-days`        | `0`     | Fetch last N days (0 = use years instead)       |
| `--fo-years`       | `5`     | Years of history to fetch                        |
| `--fo-start`       | `None`  | Start date (YYYY-MM-DD) - overrides years       |
| `--fo-end`         | `None`  | End date (YYYY-MM-DD) - default: today           |
| `--fo-skip-fetch`  | `False` | Skip data download, reuse cached CSV            |

### Model Selection

| Flag               | Default                          | Description                                      |
| ------------------ | -------------------------------- | ------------------------------------------------ |
| `--fo-models`      | `['arimax', 'lstm', 'fts']`      | Models to run (space-separated list)             |
| `--fo-ensemble`    | `True`                           | Enable ensemble averaging across models          |

**Available models**: `arimax`, `lstm`, `gru`, `xgboost`, `random_forest`, `transformer`, `fts`

### Forecast Settings

| Flag               | Default | Description                                      |
| ------------------ | ------- | ------------------------------------------------ |
| `--fo-horizon`     | `16`    | Forecast horizon (applies to ALL models)         |

### FTS-Specific (Foundation Model)

| Flag                  | Default | Description                                      |
| --------------------- | ------- | ------------------------------------------------ |
| `--fo-context`        | `128`   | Context length (sliding window size)             |
| `--fo-optimize`       | `False` | Enable Optuna hyperparameter optimization        |
| `--fo-trials`         | `20`    | Number of Optuna trials (if optimize enabled)    |
| `--fo-folds`          | `3`     | Cross-validation folds (if optimize enabled)     |
| `--fo-cpu`            | `False` | Force CPU execution (disables CUDA)              |
| `--fo-skip-inference` | `False` | Skip model run, reuse last results               |

---

## 📂 Output Structure

For each ticker, FinOracle generates:

```text
analysis_outputs/run_{TIMESTAMP}/{TICKER}/forecast/
├── data.csv                           # Raw fetched data
├── {TICKER}_O_d_{TIMESTAMP}.csv       # Timestamped data backup
├── combined_forecasts.csv             # All model predictions
│
├── {TICKER}_arimax_context.png        # Per-model context plots
├── {TICKER}_arimax_zoomed.png         # Per-model zoomed plots
├── {TICKER}_lstm_context.png
├── {TICKER}_lstm_zoomed.png
├── {TICKER}_gru_context.png
├── {TICKER}_gru_zoomed.png
├── {TICKER}_xgboost_context.png
├── {TICKER}_xgboost_zoomed.png
├── {TICKER}_randomforest_context.png
├── {TICKER}_randomforest_zoomed.png
├── {TICKER}_transformer_context.png
├── {TICKER}_transformer_zoomed.png
├── {TICKER}_fts_context.png
├── {TICKER}_fts_zoomed.png
├── {TICKER}_ensemble_context.png      # Ensemble average
├── {TICKER}_ensemble_zoomed.png
│
└── finoracle/                         # FTS-specific outputs
    ├── fincast_full_{TICKER}.csv      # FTS probabilistic forecast
    ├── finoracle_{TICKER}_forecast.png
    ├── finoracle_{TICKER}_context.png
    ├── finoracle_{TICKER}_zoomed.png
    └── run_config.txt                 # FTS configuration log
```

### Plot Styles

All plots use a **dark-themed FTS style**:
- **Context Plot**: 30 days of history + full forecast horizon
- **Zoomed Plot**: Last 10 days + forecast with trend annotation (UP/DOWN %)
- **Color Scheme**: Dark background (#1a1a2e), cyan history (#4cc9f0), red forecast (#e63946)

### Combined Forecasts CSV

The `combined_forecasts.csv` contains date-indexed predictions:

| Date       | ARIMAX  | LSTM    | GRU     | XGBoost | RandomForest | Transformer | FTS     | Ensemble |
| ---------- | ------- | ------- | ------- | ------- | ------------ | ----------- | ------- | -------- |
| 2026-02-11 | 639.72  | 570.78  | 632.05  | 639.72  | 637.74       | 691.54      | 677.96  | 652.69   |
| 2026-02-12 | 640.15  | 571.23  | 632.48  | 640.15  | 638.12       | 692.01      | 678.42  | 653.08   |
| ...        | ...     | ...     | ...     | ...     | ...          | ...         | ...     | ...      |

*Note: Shorter forecasts are NaN-padded to match the longest horizon.*

---

## 🧠 Model Details

### 1. ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables)

**Implementation**: `statsmodels.tsa.arima.model.ARIMA`

**Features**:
- Captures trends and seasonality
- Uses technical indicators (SMA, RSI, MACD) as exogenous variables
- Fast training and inference

**Best For**: Stable, trending markets with clear patterns

### 2. LSTM (Long Short-Term Memory)

**Implementation**: TensorFlow/Keras

**Architecture**:
- 60-day lookback window
- 2 LSTM layers (50 units each)
- Dropout regularization (0.2)
- Adam optimizer

**Best For**: Long-term dependencies, sequential patterns

### 3. GRU (Gated Recurrent Unit)

**Implementation**: TensorFlow/Keras

**Architecture**:
- Similar to LSTM but faster
- 60-day lookback window
- 2 GRU layers (50 units each)

**Best For**: Shorter sequences where speed matters

### 4. XGBoost (Extreme Gradient Boosting)

**Implementation**: `xgboost.XGBRegressor`

**Features**:
- Recursive multi-step forecasting
- Uses lagged features (lag_1, lag_2, lag_3)
- Technical indicators as features

**Best For**: Non-linear patterns, feature interactions

### 5. Random Forest

**Implementation**: `sklearn.ensemble.RandomForestRegressor`

**Features**:
- 100 trees
- Recursive forecasting
- Robust to outliers

**Best For**: Noisy data, handling extreme values

### 6. Transformer

**Implementation**: Keras MultiHeadAttention

**Architecture**:
- 2 attention heads
- Key dimension: 16
- Dense projection layers
- Global average pooling

**Best For**: Complex multi-scale patterns

### 7. FTS (FinCast Foundation Model)

**Implementation**: PyTorch-based custom Transformer

**Features**:
- **4GB pretrained weights** (trained on financial data)
- Probabilistic forecasting (mean + quantiles Q1-Q9)
- Context length: 32-1024 (configurable)
- Horizon: 1-256 days (configurable)
- **AutoML**: Optuna hyperparameter optimization

**Best For**: Financial time-series, probabilistic predictions

**Unique Outputs**:
- `fincast_full_{TICKER}.csv`: Mean + 9 quantiles (Q1-Q9) for uncertainty estimation
- Confidence bands in plots

---

## 🎯 Ensemble Logic

The ensemble forecast is computed as:

```python
# Truncate all forecasts to shortest length
min_len = min(len(forecast) for forecast in all_forecasts)
trimmed = [forecast[:min_len] for forecast in all_forecasts]

# Average across models
ensemble = mean(trimmed, axis=0)
```

**Why Ensemble?**
- Reduces model-specific biases
- More robust to outliers
- Typically outperforms individual models

---

## 🔧 Technical Indicators

All models (except FTS) use the following technical indicators as features:

| Indicator       | Library | Description                          |
| --------------- | ------- | ------------------------------------ |
| **SMA_20**      | `ta`    | 20-day Simple Moving Average         |
| **EMA_20**      | `ta`    | 20-day Exponential Moving Average    |
| **MACD**        | `ta`    | Moving Average Convergence Divergence|
| **RSI**         | `ta`    | Relative Strength Index (14-day)     |
| **Stoch_k**     | `ta`    | Stochastic Oscillator                |
| **BB_High/Low** | `ta`    | Bollinger Bands (upper/lower)        |

---

## 🚀 GPU Acceleration

- **Automatic CUDA detection** for TensorFlow and PyTorch models
- **FTS**: Requires CUDA-capable GPU for optimal performance
  - Falls back to CPU if unavailable (slower)
  - **Memory**: ~6GB VRAM recommended for context=128
- **LSTM/GRU/Transformer**: Benefit from GPU but work fine on CPU

---

## 📈 Integration with Valuation

Forecast results are automatically integrated into the master summary:

```text
[FINORACLE] IDXX Forecast (Ensemble, H=24):
   Expected Price: $652.69 (+2.1%)
   Models: ARIMAX, LSTM, GRU, XGBoost, RF, Transformer, FTS
```

---

## 🧪 Example Workflows

### 1. Quick Forecast (Default Settings)

```bash
python run_analysis.py --finoracle
```

**What it does**:
- Fetches 5 years of daily data
- Runs ARIMAX, LSTM, FTS
- Generates 16-day forecast
- Creates ensemble average
- Outputs plots and CSV

### 2. Comprehensive Analysis (All Models)

```bash
python run_analysis.py \
  --finoracle \
  --fo-models arimax lstm gru xgboost random_forest transformer fts \
  --fo-ensemble \
  --fo-horizon 30
```

**What it does**:
- Runs all 7 models
- 30-day forecast
- Ensemble averaging
- 16 plots (2 per model + 2 for ensemble)

### 3. FTS Optimization (Power User)

```bash
python run_analysis.py \
  --finoracle \
  --fo-models fts \
  --fo-optimize \
  --fo-trials 50 \
  --fo-folds 5 \
  --fo-horizon 60
```

**What it does**:
- Runs FTS only
- Optuna hyperparameter search (50 trials, 5-fold CV)
- Finds optimal context length
- 60-day forecast

### 4. Intraday Forecasting

```bash
python run_analysis.py \
  --finoracle \
  --fo-freq 1h \
  --fo-days 90 \
  --fo-horizon 24 \
  --fo-models lstm gru fts
```

**What it does**:
- Fetches 90 days of hourly data
- Forecasts next 24 hours
- Uses LSTM, GRU, FTS

---

## ⚠️ Known Limitations

1. **FTS Model Size**: 4GB download required (use `download_model.py`)
2. **GPU Memory**: FTS with context=128 needs ~6GB VRAM
3. **Horizon Mismatch**: Different models may produce different-length forecasts (handled via NaN-padding)
4. **Training Time**: Transformer and LSTM can be slow on CPU
5. **Data Quality**: All models require clean, continuous data (no large gaps)

---

## 🧠 Hybrid Sentiment Analysis

### Overview

FinOracle is now augmented by a **Hybrid Sentiment Engine** that combines local quantitative scoring with cloud-based qualitative analysis.

### Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                    Sentiment Pipeline                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐      ┌──────────────────────────────────┐   │
│  │  News Fetching  │      │  Layer 1: FinBERT-tone (Local)   │   │
│  │ Refinitiv → yf  │──────│  yiyanghkust/finbert-tone        │   │
│  │ headlines + body│      │  Score: positive/negative/neutral│   │
│  └─────────────────┘      └──────────────────────────────────┘   │
│                                     │                            │
│                                     ▼                            │
│                           ┌──────────────────────────────────┐   │
│                           │  Layer 2: OpenRouter LLM (API)   │   │
│                           │  Free: GPT-oss-20b, Nemotron...  │   │
│                           │  Full article → deep analysis    │   │
│                           │  (1000 req/day free tier)        │   │
│                           └──────────────────────────────────┘   │
│                                     │                            │
│                                     ▼                            │
│                           ┌──────────────────────────────────┐   │
│                           │  Output: CSV + Plot + Summary    │   │
│                           └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### Components

#### Layer 1: FinBERT (Quantitative)
*   **Model**: `ProsusAI/finbert` (Safetensors)
*   **Speed**: ~50ms per headline on CPU
*   **Output**: High-frequency sentiment signal (-1 to +1)
*   **Use Case**: Real-time trend detection, sudden news spikes

#### Layer 2: OpenRouter LLM (Qualitative)
*   **Models**: 
    *   `openai/gpt-oss-20b` (Good generalist)
    *   `nvidia/nemotron-3` (Strong reasoning)
    *   `meta-llama/llama-3` (State-of-the-art open source)
*   **Output**: Structured JSON containing:
    *   **Summary**: 1-sentence executive summary
    *   **Risks**: Specific downsides (e.g., "Regulatory probe", "Margin compression")
    *   **Opportunities**: Specific catalysts (e.g., "New product launch", "Competitor weakness")
*   **Robustness**: Includes Regex fallback parsing for malformed JSON

### Integration with Forecasting

Sentiment scores are designed to be used as **exogenous features** for the forecasting models (ARIMAX, LSTM, XGBoost), allowing the price predictions to react to breaking news and market mood.

---

## 🔮 Future Enhancements

- **Sentiment Integration**: Combine FinBERT sentiment scores with price forecasts
- **Multi-Asset**: Forecast correlations between assets
- **Volatility Forecasting**: Predict future volatility (GARCH integration)
- **Confidence Intervals**: Extend probabilistic forecasting to all models (not just FTS)

---

**⚠️ Disclaimer**: Forecasts are for **educational and research purposes only**. They do not constitute financial advice. Past performance is not indicative of future results. Always consult with a certified financial advisor before making investment decisions.

---

Built with ❤️ for ECE Business Intelligence
