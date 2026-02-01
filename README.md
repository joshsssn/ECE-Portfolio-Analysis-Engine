# ECE Portfolio Analysis Engine - Complete Documentation

> **Author:** Josh E. SOUSSAN
> **V1:** 01/02/2026
> **Project:** ECE Business Intelligence

---

## 📋 Executive Summary

A comprehensive **Python-based portfolio analysis platform** that combines:

1. **Portfolio Reconstruction** - Rebuilds full portfolio from partial holdings + ETF proxies
2. **Optimal Allocation Finder** - Automatically finds best allocation via Sharpe Optimization + Risk Budgeting (MCTR)
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
| [run_from_screener.py](file:///c:/Users/Joshs/Desktop/BI/ECE/run_from_screener.py)               | ~200  | Analyze screener CSV                     |
| [portfolio_reconstruction.py](file:///c:/Users/Joshs/Desktop/BI/ECE/portfolio_reconstruction.py) | ~760  | Portfolio weights + risk metrics         |
| [optimal_allocation.py](file:///c:/Users/Joshs/Desktop/BI/ECE/optimal_allocation.py)             | ~550  | Sharpe + MCTR optimization               |
| [backtest_candidate.py](file:///c:/Users/Joshs/Desktop/BI/ECE/backtest_candidate.py)             | ~680  | Pro-forma portfolio impact analysis      |
| [valuation_engine.py](file:///c:/Users/Joshs/Desktop/BI/ECE/valuation_engine.py)                 | ~1100 | DCF + Monte Carlo + Relative Valuation   |

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

| Flag                 | Description                                             |
| -------------------- | ------------------------------------------------------- |
| `--csv`, `-c`    | Path to screener CSV (default:`screener-results.csv`) |
| `--top`, `-n`    | Limit to top N stocks (default: 10)                     |
| `--skip-portfolio` | Skip portfolio reconstruction                           |
| `--skip-optimal`   | Skip optimal allocation finder                          |
| `--skip-backtest`  | Skip backtesting                                        |
| `--skip-valuation` | Skip valuation engine                                   |
| `--only-valuation` | Only run valuation (quick mode)                         |
| `--all`            | Analyze all stocks in screener (default: 10)            |

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

## 🔧 Module 1: Portfolio Reconstruction

### Purpose

Reconstruct a complete portfolio from **Top 10 holdings** + **sector ETF proxies** to match target sector allocations.

### Key Components

#### 1. Top 10 Holdings (Hard-coded)

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

#### 2. Target Sector Weights

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

**Automatically find the best allocation** for a candidate stock using two scientific approaches.

### Method 1: Sharpe Ratio Optimization

Scan allocations from 0% to 25% and find the point that **maximizes Sharpe Ratio**.

```python
# Scans 50 allocation levels
for allocation in np.linspace(0, 0.25, 50):
    blended = (1 - allocation) * portfolio + allocation * candidate
    sharpe = calculate_sharpe(blended)
# Find argmax(sharpe)
```

### Method 2: Risk Budgeting (MCTR)

Uses **Marginal Contribution to Risk** to find where adding more starts increasing volatility:

```python
MCTR = d(Portfolio_Vol) / d(Allocation)

# If MCTR < 0: Adding more REDUCES risk (diversification benefit)
# If MCTR > 0: Adding more INCREASES risk (concentration effect)
# Optimal = Zero crossing point
```

### Visualization (4-panel)

1. **Sharpe Curve** - Find maximum
2. **Volatility Curve** - Find minimum
3. **MCTR Chart** - Green = risk-reducing, Red = risk-increasing
4. **Efficient Frontier** - Risk vs Return with color = Sharpe

### Sample Output

```text
   📈 SHARPE OPTIMIZATION:       0.0% (no improvement)
   📊 RISK BUDGETING:           14.8% (min volatility)
   ⚖️  MCTR ANALYSIS:           14.6% (risk-neutral point)

   ✅ RECOMMENDED: 14.6% via Risk Budgeting
   
   Volatility: 15.82% → 14.98% (-0.84%)
```

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

## 📦 Dependencies

```
pandas>=2.0
numpy>=1.24
yfinance>=0.2.28
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
```

Install: `pip install pandas numpy yfinance matplotlib seaborn scipy`

---

## 🔮 Future Enhancements

1. **Database caching** for API calls (avoid rate limits)
2. **Factor model integration** (Fama-French 5-factor)
3. **Options-implied volatility** for Monte Carlo inputs
4. **PDF report generation** with charts embedded
5. **Web dashboard** (Streamlit/Dash)
6. **Add perplexity's natural language screener API** (So we won't even need to manually screen stocks)
7. **Find a way to add sentiment analysis** (To get a better idea of the market's sentiment)

---

Built with ❤️ for ECE Business Intelligence
