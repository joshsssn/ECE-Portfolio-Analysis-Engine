# ECE Portfolio Analysis Engine - Complete Documentation

> **Author:** Josh E. SOUSSAN
> **V2:** 02/02/2026
> **Project:** ECE Business Intelligence

---

## 📋 Executive Summary

A comprehensive **Python-based portfolio analysis platform** that combines:

1. **Portfolio Reconstruction** - Rebuilds full portfolio from partial holdings + ETF proxies
2. **Optimal Allocation Finder** - Finds best allocation via Mean-Variance Utility Maximization
3. **Candidate Backtesting** - Simulates adding new stocks to measure risk/return impact
4. **Valuation Engine** - DCF + Monte Carlo + Relative Valuation with regression analysis

All orchestrated via a single script that outputs organized results to timestamped folders.

---

## 🏗️ Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      run_analysis.py                                │
│                   (Master Orchestrator)                             │
└───────────┬────────────┬────────────┬────────────┬──────────────────┘
            │            │            │            │
    ┌───────▼──────┐ ┌───▼────┐ ┌─────▼─────┐ ┌────▼──────┐
    │ portfolio_   │ │optimal_│ │ backtest_ │ │ valuation_│
    │ reconstruct  │ │allocat │ │ candidate │ │ engine    │
    │ .py          │ │ ion.py │ │ .py       │ │ .py       │
    └──────────────┘ └────────┘ └───────────┘ └───────────┘
            │            │            │            │
            ▼            ▼            ▼            ▼
    ┌─────────────────────────────────────────────────────┐
    │                  yfinance API                       │
    │       (Market Data, Financials, Beta)               │
    └─────────────────────────────────────────────────────┘
```

---

## 📁 File Structure

| File                                                                                          | Lines | Purpose                                  |
| :-------------------------------------------------------------------------------------------- | :---- | :--------------------------------------- |
| [run_analysis.py](file:///c:/Users/Joshs/Desktop/BI/ECE/run_analysis.py)                         | ~500  | Master orchestrator - runs all 4 modules |
| [run_from_screener.py](file:///c:/Users/Joshs/Desktop/BI/ECE/run_from_screener.py)               | ~350  | Analyze screener CSV                     |
| [run_multi_allocation.py](file:///c:/Users/Joshs/Desktop/BI/ECE/run_multi_allocation.py)         | ~150  | Multi-allocation metrics generator       |
| [portfolio_reconstruction.py](file:///c:/Users/Joshs/Desktop/BI/ECE/portfolio_reconstruction.py) | ~760  | Portfolio weights + risk metrics         |
| [optimal_allocation.py](file:///c:/Users/Joshs/Desktop/BI/ECE/optimal_allocation.py)             | ~700  | Mean-Variance Utility optimization       |
| [backtest_candidate.py](file:///c:/Users/Joshs/Desktop/BI/ECE/backtest_candidate.py)             | ~700  | Pro-forma portfolio impact (incl. VaR)   |
| [valuation_engine.py](file:///c:/Users/Joshs/Desktop/BI/ECE/valuation_engine.py)                 | ~1100 | DCF + Monte Carlo + Relative Valuation   |
| [streamlit_app.py](file:///c:/Users/Joshs/Desktop/BI/ECE/streamlit_app.py)                       | ~330  | Streamlit app / GUI                      |
| [portfolio_loader.py](file:///c:/Users/Joshs/Desktop/BI/ECE/portfolio_loader.py)                 | ~110  | Handles loading portfolio data           |
| [config.py](file:///c:/Users/Joshs/Desktop/BI/ECE/config.py)                                     | ~40   | Default settings                         |

---

## 📥 Running from Screener CSV

### Screener Format

The CSV must have `symbol` and `companyName` columns:

```csv
symbol,companyName,industry,marketCap,...
ELV,Elevance Health Inc.,Medical - Healthcare Plans,76836829920,...
LLY,Eli Lilly and Company,Drug Manufacturers - General,929760149377,...
```

### Usage

```bash
# Analyze top 5 stocks from screener
python run_from_screener.py --top 5

# Custom CSV path
python run_from_screener.py --csv my_screener.csv --top 10

# Only run valuation (skip portfolio/backtest)
python run_from_screener.py --only-valuation --top 3

# Skip specific steps
python run_from_screener.py --skip-optimal --skip-backtest
```

### CLI Options

| Flag                        | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| `--csv`, `-c`           | Path to screener CSV (default:`screener-results.csv`)   |
| `--top`, `-n`           | Limit to top N stocks (default: 10)                       |
| `--skip-portfolio`        | Skip portfolio reconstruction                             |
| `--skip-optimal`          | Skip optimal allocation finder                            |
| `--skip-backtest`         | Skip backtesting                                          |
| `--skip-valuation`        | Skip valuation engine                                     |
| `--only-valuation`        | Only run valuation (quick mode)                           |
| `--multi-alloc`, `-m`   | Run multi-allocation analysis (optional: step % e.g. 0.5) |
| `--all`                   | Analyze all stocks in screener (default: 10)              |
| `--risk-aversion`         | Risk aversion coefficient (λ) (Default: 2.0)             |
| `--concentration-penalty` | Concentration penalty (γ) (Default: 0.5)                 |
| `--min-recommended`       | Min recommended allocation (e.g. 0.03 for 3%)             |
| `--max-allocation`        | Max allocation cap (e.g. 0.25 for 25%)                    |
| `--risk-free-rate`        | Risk free rate (e.g. 0.045 for 4.5%)                      |
| `--correlation-ticker`    | Ticker for correlation check (Default: IXN)               |
| `--holdings-csv`          | Path to custom portfolio holdings CSV                     |
| `--sectors-csv`           | Path to custom sector targets CSV                         |

### ⚠️ Screener Filtering Recommendations

For **meaningful analysis results**, filter your screener to exclude low-quality stocks:

| Filter                         | Why                                                   |
| ------------------------------ | ----------------------------------------------------- |
| **Market Cap > $1B**     | Micro-caps have unreliable financials and illiquidity |
| **Positive FCF**         | Negative FCF produces meaningless/negative DCF values |
| **P/E Ratio > 5**        | Avoids extreme value traps and distressed companies   |
| **Revenue > $100M**      | Ensures stable business with trackable growth         |
| **Beta between 0.5-2.5** | Avoids illiquid or highly speculative stocks          |

**Example screener query (FinancialModelingPrep):**

```
marketCap > 1000000000 AND 
freeCashFlow > 0 AND 
peRatio > 5 AND 
revenue > 100000000
```

> **Note**: Penny stocks and micro-caps will produce extreme values like:
>
> - DCF Fair Value: $0.00 (no FCF)
> - P/E Discount: +35,000% (nonsensical)
> - Optimal Allocation: 0% (too volatile)

---

## 🐳 Docker Usage [Depreciated for now]

To run the application in a Docker container:

1. **Build the image**:

   ```bash
   docker build -t ece-analysis-app .
   ```
2. **Run the container**:

   ```bash
   docker run -p 8501:8501 ece-analysis-app
   ```
3. **Access the App**:
   Open your browser and navigate to `http://localhost:8501`.

---

## ⚙️ Configuration & Custom Data

### Configuration Parameters Reference

These parameters can be tweaked via the CLI or the Streamlit GUI to customize the analysis logic.

| Parameter                            | Type      | Default   | Description                                                                                                                                                                                                               |
| :----------------------------------- | :-------- | :-------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Risk Aversion (λ)**         | `float` | `2.0`   | Controls the penalty for volatility in the utility function. Higher = more conservative.`<br>`• `0.5`: Aggressive (close to Sharpe max)`<br>`• `2.0`: Moderate (balanced)`<br>`• `5.0+`: Very Conservative |
| **Concentration Penalty (γ)** | `float` | `0.5`   | Penalizes large allocations to a single stock.`<br>`Calculated as $- \gamma \times w^2$. Higher = forces more diversification.                                                                                        |
| **Min Recommended (%)**        | `float` | `3.0%`  | Floor for positive recommendations. If the optimized allocation is below this (e.g., 0.5%), it is rounded down to 0% or up to this floor to avoid negligible positions.                                                   |
| **Max Allocation (%)**         | `float` | `25.0%` | Hard cap on the allocation for any single candidate stock.                                                                                                                                                                |
| **Risk-Free Rate**             | `float` | `4.0%`  | Used for Sharpe Ratio and DCF WACC calculations. Should reflect current 10Y Treasury yield.                                                                                                                               |
| **Lookback Years**             | `int`   | `5`     | Historical data period for calculating volatility and correlation.                                                                                                                                                        |

### Custom Data Formats

You can override the default portfolio assumptions by providing your own CSV files.

#### 1. Holdings CSV (`--holdings-csv`)

Defines your current portfolio or the "base" portfolio for reconstruction.

**Required Columns:** `Ticker`, `Weight`, `Sector`
**Optional Columns:** `Name`, `Country`

**Example:**

```csv
Ticker,Weight,Sector,Name,Country
AAPL,7.0,Information Technology,Apple Inc,USA
MSFT,6.0,Information Technology,Microsoft Corp,USA
JPM,3.0,Financials,JPMorgan,USA
```

#### 2. Sector Targets CSV (`--sectors-csv`)

Defines the target sector allocation distribution that the reconstruction engine will aim for (filling gaps with ETFs).

**Required Columns:** `Sector`, `Weight`

**Example:**

```csv
Sector,Weight
Information Technology,26.5
Financials,12.5
Health Care,9.5
Industrials,8.0
Energy,3.9
```

---

## 🖥️ Streamlit GUI

The application includes a fully interactive web interface (`streamlit_app.py`) for easy configuration and execution.

### Advanced Configuration Panel

Customize the analysis without touching the code:

- **Portfolio Parameters**: Adjust Risk Aversion (λ), Concentration Penalty (γ), and Min Recommended Allocation.
- **Allocation Constraints**: Set Maximum Allocation caps (e.g. 25%).
- **Market Parameters**: Configure Risk-Free Rate, Benchmark (ACWI), and Correlation Ticker (IXN).
- **Simulation Settings**: Control lookback period and number of Monte Carlo simulations (1k-50k).

### Custom Data Upload

Override default portfolio assumptions by uploading your own CSVs:

- **Holdings CSV**: Your current portfolio positions.
- **Sector Targets CSV**: Your target sector weightings.

Run the app locally:

```bash
python -m streamlit run streamlit_app.py
```

---

## 📊 Multi-Allocation Analysis

### Purpose

Generate **comprehensive metrics at 0.5% allocation granularity** from 0.5% to the optimal allocation for any stock.

### Standalone Usage

```bash
# Run for a single ticker (default 0.5% step)
python run_multi_allocation.py COR "Cencora Inc."

# Run with custom step (e.g., 1.0%)
python run_multi_allocation.py COR "Cencora Inc." 1.0
```

### Batch Usage (via Screener)

```bash
# Run multi-allocation (default 0.5% step)
python run_from_screener.py --multi-alloc --top 5

# Run with custom granularity (e.g., 0.25% step)
# Note: Smaller steps = longer runtime!
python run_from_screener.py --multi-alloc 0.25 --top 5
```

### Output

For each stock, generates `{TICKER}_multi_allocation.csv` with:

| Column                    | Description                         |
| ------------------------- | ----------------------------------- |
| `Allocation (%)`        | 0.5, 1.0, 1.5, ... up to optimal    |
| `Is Optimal`            | "Yes" or "No"                       |
| `Annualized Return (%)` | Portfolio return at this allocation |
| `Sharpe Ratio`          | Risk-adjusted return                |
| `VaR (95%, annualized)` | Value at Risk (annualized)          |
| `Max Drawdown (%)`      | Maximum drawdown                    |
| `Return Change (%)`     | Change vs original portfolio        |
| `Sharpe Change`         | Change in Sharpe ratio              |

> **Note**: Multi-allocation analysis is computationally expensive. For a stock with 16% optimal allocation, it runs 33 backtests (0.5% to 16.5%).

---

## 🔧 Module 1: Portfolio Reconstruction

### Purpose

Reconstruct a complete portfolio from **Top 10 holdings** + **sector ETF proxies** to match target sector allocations.

### Key Components

#### 1. Top 10 Holdings (Default)

```python
TOP_10_HOLDINGS = {
    'AAPL': {'weight': 7.0, 'sector': 'Information Technology'},
    'MSFT': {'weight': 6.0, 'sector': 'Information Technology'},
    'NVDA': {'weight': 5.0, 'sector': 'Information Technology'},
    'ASML': {'weight': 4.0, 'sector': 'Information Technology'},
    'SAP':  {'weight': 2.5, 'sector': 'Information Technology'},
    'REY.MI': {'weight': 2.0, 'sector': 'Information Technology'},
    'IDR.MC': {'weight': 2.0, 'sector': 'Industrials'},
    'JPM':  {'weight': 3.0, 'sector': 'Financials'},
    'GS':   {'weight': 2.5, 'sector': 'Financials'},
    'HSBC': {'weight': 2.0, 'sector': 'Financials'},
}
```

#### 2. Target Sector Weights (Default)

| Sector                 | Weight |
| :--------------------- | :----- |
| Information Technology | 26.5%  |
| Financials             | 12.5%  |
| Commodities            | 12.1%  |
| Health Care            | 9.5%   |
| Real Estate            | 8.7%   |
| Industrials            | 8.0%   |
| Communication Services | 6.5%   |
| Consumer Staples       | 5.0%   |
| Consumer Discretionary | 5.0%   |
| Energy                 | 3.9%   |
| Utilities              | 2.3%   |

#### 3. Sector ETF Proxies

Any gap between target weight and Top 10 is filled with iShares Global ETFs:

- `IXN` (Tech), `IXG` (Financials), `IXJ` (Healthcare), `EXI` (Industrials)
- `IXC` (Energy), `MXI` (Commodities), `KXI` (Staples), `RXI` (Discretionary)
- `JXI` (Utilities), `IXP` (Communications), `REET` (Real Estate)

#### 4. Risk Metrics Calculated

| Metric                | Formula                                    |
| :-------------------- | :----------------------------------------- |
| Annualized Return     | $(1 + Total Return)^{1/years} - 1$       |
| Annualized Volatility | $\sigma_{weekly} \times \sqrt{52}$       |
| Sharpe Ratio          | $(R_p - R_f) / \sigma_p$                 |
| Beta                  | $Cov(R_p, R_m) / Var(R_m)$               |
| Alpha (Jensen's)      | $R_p - [R_f + \beta(R_m - R_f)]$         |
| Information Ratio     | $(R_p - R_b) / TE$                       |
| Maximum Drawdown      | $\min[(Cum - Peak) / Peak]$              |
| VaR (95%)             | $Percentile_5(Returns) \times \sqrt{52}$ |

---

## 🆕 Module 2: Optimal Allocation Finder

### Purpose

**Automatically find the best allocation** for a candidate stock using **Mean-Variance Utility Maximization**.

### Primary Method: Mean-Variance Utility (with Concentration Penalty)

The allocation is chosen by maximizing the utility function:

$$
U = E[R] - \frac{\lambda}{2} \times \sigma^2 - \gamma \times w^2
$$

Where:

- **E[R]** = Expected annualized return
- **σ** = Annualized volatility
- **w** = Allocation weight (0 to 1)
- **λ** = Risk aversion coefficient (Default: 2.0)
- **γ** = Concentration penalty (Default: 0.5)

The term $-\gamma \times w^2$ acts as a **soft cap**, mathematically discouraging extreme allocations (like 25%) in favor of more diversified positions (typically 5-15%).

```python
# Parameters in optimal_allocation.py
RISK_AVERSION = 2.0          # Moderate-Aggressive
CONCENTRATION_PENALTY = 0.5  # Institutional-grade diversification
MIN_RECOMMENDED_ALLOCATION = 0.03 # Floor
```

### Reference Methods (for comparison)

| Method                        | Description                   | When λ →        |
| ----------------------------- | ----------------------------- | ----------------- |
| **Sharpe Optimization** | Maximize Sharpe Ratio         | 0 (aggressive)    |
| **Min Volatility**      | Minimize portfolio volatility | ∞ (conservative) |

### Visualization (4-panel)

1. **Sharpe Curve** - Find maximum (reference)
2. **Volatility Curve** - Find minimum (reference)
3. **Utility Curve** - Find maximum (**primary method**)
4. **Efficient Frontier** - Risk vs Return with color = Sharpe

### Sample Output

```text
   📈 SHARPE OPTIMIZATION (reference):   25.0%
   📊 MIN VOLATILITY (reference):        12.3%
   🎯 MEAN-VARIANCE UTILITY (λ=3.0):     18.4%

   ✅ RECOMMENDED: 18.4% via Mean-Variance Utility (λ=3.0)
   
   Utility: 12.71% → 13.45% (+0.74%)
```

---

## 🛡️ Module 2b: Advanced Risk & Execution (Sprint 1)

**New in v2.1**: A suite of professional-grade risk management tools inspired by hedge fund best practices.

### 1. Advanced Risk Metrics

Beyond standard volatility, we now calculate:

* **CVaR (95%)**: Conditional Value at Risk (Expected Shortfall) - usually 2x worse than VaR.
* **Sortino Ratio**: Like Sharpe, but only penalizes *downside* volatility.
* **Omega Ratio**: Probability weighted ratio of gains vs losses.

### 2. Drawdown Protection (Bridgewater-style)

* **Logic**: Automatically reduces allocation if the asset is currently in a drawdown.
* **Example**: If an asset has an optimal weight of 10% but is down -15% from ATH, the system cuts allocation to 5% to preserve capital ("Cut losers fast").

### 3. Stress Testing

Simulates portfolio performance during historical crises:

* **2008 Financial Crisis**: Modeled as -37% drop (SPY proxy).
* **COVID-19 Crash**: Modeled as -34% drop.
* **Output**: Shows estimated $ loss and % drawdown for your specific portfolio.

### 4. Ledoit-Wolf Shrinkage

* **Problem**: Standard covariance matrices are noisy and prone to estimation error with limited data.
* **Solution**: We use Ledoit-Wolf shrinkage to "pull" the correlation matrix towards a constant correlation target.
* **Benefit**: Mathematically stable borders and less extreme allocations.

### 5. Rebalancing Execution

Generates a precise **Trade List** to go from Current Portfolio → Target Portfolio.

* **Features**:
  * Round to 100-share lots (Institutional mode).
  * Minimum trade size filter (avoid $5 fees for $10 trades).
  * Turnover analysis.

---

## 🔧 Module 3: Candidate Backtesting

### Purpose

Simulate adding a **candidate stock** (using the optimal allocation found in Module 2) to the existing portfolio and measure impact.

### Pro-Forma Construction

```python
New_Portfolio = (1 - allocation) × Old_Portfolio + allocation × Candidate
```

### Key Function: `run_backtest()`

```python
from backtest_candidate import run_backtest

result = run_backtest(
    ticker='UNH',
    name='UnitedHealth Group',
    allocation=0.05,  # 5%
    output_dir='./outputs',
    show_plot=False
)
```

### Output Metrics

| Metric                    | What It Measures               |
| :------------------------ | :----------------------------- |
| Return Change             | Impact on annualized return    |
| Volatility Change         | Impact on portfolio volatility |
| Sharpe Change             | Impact on risk-adjusted return |
| Beta Change               | More/less market exposure      |
| Correlation vs Portfolio  | Diversification potential      |
| Correlation vs Tech (IXN) | Sector overlap                 |

### Visualization (4-panel)

1. **Cumulative Returns** - Before vs After comparison
2. **Rolling 52-Week Beta** - Market sensitivity over time
3. **Correlation Heatmap** - Candidate vs all components
4. **Drawdown Comparison** - Worst-case scenarios

---

## 🔧 Module 4: Valuation Engine

### Purpose

Compute **intrinsic value** using state-of-the-art probabilistic methods.

### Architecture

```text
┌────────────────────────────────────────────────────┐
│               ValuationEngine (Class)              │
├────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ DataFetcher  │  │DCFValuation  │  │MonteCarlo │ │
│  │              │  │              │  │Valuation  │ │
│  │ - Revenue    │  │ - FCF Proj   │  │           │ │
│  │ - FCF        │  │ - WACC       │  │ - 10,000  │ │
│  │ - Debt/Cash  │  │ - Terminal   │  │   trials  │ │
│  │ - Beta       │  │   Value      │  │           │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                    │
│  ┌──────────────┐  ┌──────────────────────────────┐│
│  │ Sensitivity  │  │   RelativeValuation          ││
│  │ Analysis     │  │   (Regression Comps)         ││
│  │              │  │                              ││
│  │ WACC ± 1%    │  │ P/E ~ f(Growth)              ││
│  │ Growth ± 1%  │  │ EV/EBITDA ~ f(Growth)        ││
│  └──────────────┘  └──────────────────────────────┘│
└────────────────────────────────────────────────────┘
```

### Components

1. **DCF Valuation** - 10-year FCF projection + terminal value
2. **Monte Carlo** - 10,000 simulations with randomized inputs
3. **Relative Valuation** - Regression-based trading comps (P/E ~ f(Growth))
4. **Sensitivity Analysis** - WACC × Terminal Growth matrix

### Key Formulas

```python
WACC = (E/(E+D)) × Cost_Equity + (D/(E+D)) × Cost_Debt × (1 - Tax)
Terminal_Value = FCF_Year10 × (1 + g) / (WACC - g)
Win_Probability = % of MC simulations where Fair_Value > Current_Price
```

---

---

---

## 🧠 Module 5: Hybrid Sentiment Analysis

### Purpose

Combine **local speed** (FinBERT) with **cloud intelligence** (OpenRouter LLMs) to provide a 360° view of market sentiment. It analyzes both headlines (quant) and full articles (qualitative) to generate actionable risks and opportunities.

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

### Key Components

1. **News Fetching**:

   * **Refinitiv (Eikon)**: Primary source for institutional-grade news (Reuters).
   * **yfinance**: Fallback source if Eikon is unavailable.
2. **Layer 1: FinBERT (Quantitative)**:

   * **Model**: `ProsusAI/finbert` (Safetensors version).
   * **Function**: Instantly scores hundreds of headlines.
   * **Output**: Sentiment score (-1 to +1) and label (Positive/Negative/Neutral).
3. **Layer 2: AI Deep Analysis (Qualitative)**:

   * **Model**: OpenRouter (e.g., `openai/gpt-oss-20b`, `nvidia/nemotron-3`).
   * **Function**: Reads full article text to extract nuance.
   * **Outputs**:
     * **Summary**: 1-sentence executive summary.
     * **Key Risks**: Bullet points of specific downsides.
     * **Key Opportunities**: Bullet points of potential upside catalysts.
     * **AI Sentiment**: A second opinion score based on deep reading.

### Usage

Enable sentiment analysis in the GUI (sidebar) or via CLI:

```bash
# Basic usage (FinBERT only)
python run_analysis.py --sentiment

# Enable OpenRouter Deep Analysis
python run_analysis.py --sentiment --openrouter --or-model "openrouter/free"
```

### Outputs

* **Interactive UI**: "Sentiment Analysis Insights" section in Streamlit with tabs for FinBERT vs AI plots and a detailed insights table.
* **Files**:
  * `{TICKER}_sentiment.csv`: Detailed logs including raw text and scores.
  * `{TICKER}_sentiment.png`: FinBERT score timeline.
  * `{TICKER}_sentiment_llm.png`: AI Deep Analysis timeline.

---

## 🔮 Module 6: FinOracle - AI-Powered Forecasting

### Purpose

Leverage **7 state-of-the-art forecasting models** to predict future price movements, providing an AI-driven edge to valuation and risk analysis. FinOracle combines classical time-series methods with cutting-edge deep learning to deliver robust, ensemble-based predictions.

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    FinOracle Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   ARIMAX     │  │     LSTM     │  │     GRU      │       │
│  │ (Classical)  │  │  (RNN-based) │  │  (RNN-based) │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   XGBoost    │  │Random Forest │  │ Transformer  │       │
│  │ (Gradient)   │  │  (Ensemble)  │  │ (Attention)  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FTS (Foundation Model)                  │   │
│  │    FinCast Transformer - 4GB Pretrained Weights      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│                          ↓                                  │
│                   ┌──────────────┐                          │
│                   │   Ensemble   │                          │
│                   │   Average    │                          │
│                   └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### Forecasting Models

| Model                   | Type                | Strengths                                    | Horizon        |
| ----------------------- | ------------------- | -------------------------------------------- | -------------- |
| **ARIMAX**        | Statistical         | Captures trends + seasonality + exogenous    | Configurable   |
| **LSTM**          | Deep Learning (RNN) | Long-term dependencies, sequential patterns  | Configurable   |
| **GRU**           | Deep Learning (RNN) | Faster than LSTM, good for shorter sequences | Configurable   |
| **XGBoost**       | Gradient Boosting   | Non-linear patterns, feature interactions    | Configurable   |
| **Random Forest** | Ensemble (Trees)    | Robust to noise, handles outliers well       | Configurable   |
| **Transformer**   | Attention-based     | Multi-head attention, parallel processing    | Configurable   |
| **FTS (FinCast)** | Foundation Model    | Pretrained on financial data, probabilistic  | Up to 256 days |
| **Ensemble**      | Meta-model          | Averages all models for robust prediction    | Auto-aligned   |

### Usage

Enable FinOracle forecasting with the `--finoracle` flag:

```bash
# Basic usage with default settings
python run_analysis.py --finoracle

# Custom horizon and models
python run_analysis.py --finoracle --fo-horizon 30 --fo-models arimax lstm fts

# Enable ensemble averaging
python run_analysis.py --finoracle --fo-ensemble

# Run from screener with forecasting
python run_from_screener.py --finoracle --top 5
```

### CLI Parameters

#### Data Fetching (Refinitiv)

| Flag                | Default   | Description                                    |
| ------------------- | --------- | ---------------------------------------------- |
| `--fo-freq`       | `d`     | Data frequency (tick, 1min, 5min, 1h, d, w, m) |
| `--fo-days`       | `0`     | Fetch last N days (0 = use years instead)      |
| `--fo-years`      | `5`     | Years of history to fetch                      |
| `--fo-start`      | `None`  | Start date (YYYY-MM-DD) - overrides years      |
| `--fo-end`        | `None`  | End date (YYYY-MM-DD) - default: today         |
| `--fo-skip-fetch` | `False` | Skip data download, reuse cached CSV           |

#### Model Selection

| Flag              | Default                       | Description                             |
| ----------------- | ----------------------------- | --------------------------------------- |
| `--fo-models`   | `['arimax', 'lstm', 'fts']` | Models to run (space-separated list)    |
| `--fo-ensemble` | `True`                      | Enable ensemble averaging across models |

**Available models**: `arimax`, `lstm`, `gru`, `xgboost`, `random_forest`, `transformer`, `fts`

#### Forecast Settings

| Flag             | Default | Description                              |
| ---------------- | ------- | ---------------------------------------- |
| `--fo-horizon` | `16`  | Forecast horizon (applies to ALL models) |

#### FTS-Specific (Foundation Model)

| Flag                    | Default   | Description                                   |
| ----------------------- | --------- | --------------------------------------------- |
| `--fo-context`        | `128`   | Context length (sliding window size)          |
| `--fo-optimize`       | `False` | Enable Optuna hyperparameter optimization     |
| `--fo-trials`         | `20`    | Number of Optuna trials (if optimize enabled) |
| `--fo-folds`          | `3`     | Cross-validation folds (if optimize enabled)  |
| `--fo-cpu`            | `False` | Force CPU execution (disables CUDA)           |
| `--fo-skip-inference` | `False` | Skip model run, reuse last results            |

### Output Structure

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

All plots use a **dark-themed FTS style** with:

- **Context Plot**: 30 days of history + full forecast horizon
- **Zoomed Plot**: Last 10 days + forecast with trend annotation
- Color scheme: Dark background (#1a1a2e), cyan history (#4cc9f0), red forecast (#e63946)

### Combined Forecasts CSV

The `combined_forecasts.csv` contains date-indexed predictions:

| Date       | ARIMAX | LSTM   | GRU    | XGBoost | RandomForest | Transformer | FTS    | Ensemble |
| ---------- | ------ | ------ | ------ | ------- | ------------ | ----------- | ------ | -------- |
| 2026-02-11 | 639.72 | 570.78 | 632.05 | 639.72  | 637.74       | 691.54      | 677.96 | 652.69   |
| 2026-02-12 | 640.15 | 571.23 | 632.48 | 640.15  | 638.12       | 692.01      | 678.42 | 653.08   |
| ...        | ...    | ...    | ...    | ...     | ...          | ...         | ...    | ...      |

*Note: Shorter forecasts are NaN-padded to match the longest horizon.*

### Integration with Valuation

Forecast results are automatically integrated into the master summary:

```text
[FINORACLE] IDXX Forecast (Ensemble, H=24):
   Expected Price: $652.69 (+2.1%)
   Models: ARIMAX, LSTM, GRU, XGBoost, RF, Transformer, FTS
```

### Technical Details

#### Model Implementations

- **ARIMAX**: `statsmodels.tsa.arima.model.ARIMA` with technical indicators as exogenous variables
- **LSTM/GRU**: TensorFlow/Keras with 60-day lookback, dropout regularization
- **XGBoost/RF**: Scikit-learn with recursive multi-step forecasting
- **Transformer**: Keras MultiHeadAttention with positional encoding
- **FTS**: PyTorch-based foundation model (4GB weights) with probabilistic output

#### Ensemble Logic

The ensemble forecast is computed as:

```python
ensemble = mean([model1_forecast[:min_len], model2_forecast[:min_len], ...])
```

Where `min_len` is the shortest forecast length among all models to handle horizon mismatches.

### GPU Acceleration

- **Automatic CUDA detection** for TensorFlow and PyTorch models
- **FTS**: Requires CUDA-capable GPU for optimal performance (falls back to CPU if unavailable)
- **Memory**: ~6GB VRAM recommended for FTS with context=128

### Example Workflow

```bash
# 1. Fetch 3 years of daily data for AAPL
# 2. Run ARIMAX, LSTM, and FTS models
# 3. Generate 30-day forecast
# 4. Create ensemble average
# 5. Output plots and CSV

python run_analysis.py \
  --finoracle \
  --fo-years 3 \
  --fo-horizon 30 \
  --fo-models arimax lstm fts \
  --fo-ensemble
```

---

## 🚀 Master Orchestrator: run_analysis.py

### Configuration

```python
CANDIDATE_STOCKS = [
    {'ticker': 'UNH', 'name': 'UnitedHealth Group', 'allocation': 0.05},
    {'ticker': 'V', 'name': 'Visa Inc.', 'allocation': 0.04},
]

RUN_PORTFOLIO_RECONSTRUCTION = True
RUN_OPTIMAL_ALLOCATION = True  # NEW!
RUN_BACKTESTS = True
RUN_VALUATION = True
```

### Usage

```bash
python run_analysis.py
```

### Output Folder Structure

```text
analysis_outputs/run_{TIMESTAMP}/
├── 0_portfolio/
│   ├── portfolio_risk_metrics.csv
│   ├── portfolio_weights.csv
│   └── portfolio_analysis_chart.png
│
├── {TICKER}/                       # Per-stock folder
│   ├── optimal.png                 # 4-panel optimization chart
│   ├── optimal_summary.csv         # Individual optimal allocation
│   ├── backtest.png                # Backtest visualization
│   ├── backtest.csv                # Backtest metrics
│   ├── {TICKER}_multi_allocation.csv  # Multi-allocation metrics (if --multi-alloc)
│   ├── valuation_dcf.png           # DCF chart
│   └── valuation_relative.png      # Relative valuation chart
│
├── summary/
│   ├── master_summary.csv          # Cross-stock comparison
│   ├── optimal_summary.csv         # All optimal allocations
│   ├── valuation_summary.csv       # All valuations
│   └── analysis_report.txt         # Human-readable report
│
└── input_screener.csv              # Copy of input data
```

## 📊 Sample Results

### Master Summary (3 Candidates)

| Ticker | Price         | DCF Value          | Margin of Safety | Win Prob | P/E Discount |
| ------ | ------------- | ------------------ | ---------------- | -------- | ------------ |
| UNH    | $287 | $1,033 | **+260%** 🟢 | 100%             | +81%     |              |
| V      | $322 | $345   | +7% 🟡             | 64%              | -7%      |              |
| TMO    | $579 | $288   | **-50%** 🔴  | 0%               | -23%     |              |

### Interpretation

- **UNH**: Massively undervalued by DCF. High FCF yield healthcare stock.
- **V**: Fairly valued by DCF, but overvalued vs payment peers (high EV/EBITDA).
- **TMO**: Overvalued by both methods. 0% win probability.

---

## 📦 Installation :

### (assuming you run Windows)

1) Open a CMD/PowerShell in the root folder and run `uv sync`
2) Activate the venv by running `.\.venv\Scripts\activate.ps1`
3) Run `uv run python Forecast\FinCast-fts\Inference\download_model.py` (optional, only if you want to download the Frontier model)

Few notes :

- If you don't have Python installed, you can download it from [here](https://www.python.org/downloads/) (make sure to install Python 3.10 or higher).
- If you don't have uv installed, you can install it from [here](https://astral.sh/uv/install.sh)
- You'll also need CUDA installed for the GPU acceleration (if you don't have a CUDA compatible GPU or no GPU at all, you can skip this step). You can download it from [here](https://developer.nvidia.com/cuda-downloads)
- Depending on your CUDA version, you may need to install a specific version of PyTorch. You can check the compatibility of PyTorch with CUDA versions [here](https://pytorch.org/get-started/locally/) and modify the pyproject.toml file accordingly.

---

## 🔮 Future Enhancements

Lookup implementation_plan.md if you're curious / want to help out a bro :)

---

**⚠️ Disclaimer**: This software is for **educational and research purposes only**. It does not constitute financial advice, investment recommendations, or an offer to buy/sell any securities. Financial markets involve significant risk. Past performance is not indicative of future results. Always consult with a certified financial advisor before making any investment decisions.

---

Built with ❤️ for ECE Business Intelligence (and a bit for my resume too lol)
