# ECE Portfolio Analysis Engine - Web UI Guide

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

All orchestrated via an **interactive web interface** that outputs organized results to timestamped folders.

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

### Main Workflow

```
┌─────────────────────────┐
│  Choose Screener Input  │
└────────────┬────────────┘
             │
      ┌──────▼──────┐
      │  Use Sample │──→ Load default healthcare stocks
      │   OR        │
      │  Upload CSV │──→ Load your own screener results
      └──────┬──────┘
             │
      ┌──────▼──────────────┐
      │  Configure Analysis │
      │ (Sidebar + Adv Opts)│
      └──────┬──────────────┘
             │
      ┌──────▼──────────┐
      │ Run Analysis    │
      │ Click Button    │
      └──────┬──────────┘
             │
      ┌──────▼──────────────────┐
      │ View Results & Download │
      │ (ZIP archive)           │
      └─────────────────────────┘
```

---

## 📥 Screener Input

### Option 1: Use Default Screener (No Upload Required)

Perfect for **testing and evaluation**:

1. Check the **"📋 Use Default Screener (for testing)"** checkbox
2. View a preview of 20 healthcare stocks
3. Click **"🚀 Run Analysis"** to proceed

The default screener includes stocks like: ELV, LLY, IDXX, MCK, ZTS, RMD, MRK, HCA, GILD, UTHR, VRTX, ISRG, JNJ, BMY, COR, A, EW, DXCM, AMGN

### Option 2: Upload Your Own Screener CSV

Your CSV must contain at minimum: `symbol` and `companyName` columns. Example:

```csv
symbol,companyName,industry,marketCap,...
ELV,Elevance Health Inc.,Medical - Healthcare Plans,76836829920,...
LLY,Eli Lilly and Company,Drug Manufacturers - General,929760149377,...
UNH,UnitedHealth Group Inc.,Medical - Healthcare Plans,515000000000,...
```

---

## ⚙️ Configuration Panel (Sidebar)

The left sidebar contains all configuration options organized in sections:

### 📊 Stock Selection

| Option                       | Default   | Purpose                                                  |
| ---------------------------- | --------- | -------------------------------------------------------- |
| **Analyze ALL stocks** | Unchecked | When checked, runs analysis on every stock in screener   |
| **Top N Stocks**       | 5         | When above is unchecked, limits analysis to top N stocks |

### 🔄 Pipeline Steps

Toggle which analysis modules to run:

| Option                                  | Default   | Purpose                                        |
| --------------------------------------- | --------- | ---------------------------------------------- |
| **Skip Portfolio Reconstruction** | Unchecked | Rebuilds portfolio from holdings + ETF proxies |
| **Skip Optimal Allocation**       | Unchecked | Finds best allocation for each candidate       |
| **Skip Backtests**                | Unchecked | Simulates adding stock to portfolio            |
| **Skip Valuation**                | Unchecked | Runs DCF, Monte Carlo, and relative valuation  |
| **Only Run Valuation**            | Unchecked | Quick mode: skips portfolio, optimal, backtest |

### 📈 Multi-Allocation Analysis

| Option                                     | Default   | Purpose                                                 |
| ------------------------------------------ | --------- | ------------------------------------------------------- |
| **Enable Multi-Allocation Analysis** | Unchecked | Generate metrics at 0.5% granularity from 0% to optimal |
| **Step Granularity (%)**             | 0.5%      | Controls granularity (smaller = longer runtime)         |

⚠️ **Note**: Multi-allocation is computationally expensive. Enabled only when checkbox is ticked.

### 🛡️ Risk Management (NEW)

Enable institutional-grade risk controls:

| Option                       | Default | Function                                                   |
| ---------------------------- | ------- | ---------------------------------------------------------- |
| **Drawdown Protection**      | ☑️      | Reduce weight if asset is >10% down from peak.             |
| **Drawdown Threshold**       | 10%     | Level where protection kicks in.                           |
| **Run Stress Tests**         | ☑️      | Simulate 2008 & COVID crashes on portfolio.                |
| **Use Ledoit-Wolf**          | ☑️      | Stabilize covariance matrix (Recommended).                 |

### ⚖️ Rebalancing (NEW)

Generate executable orders to reach target weights:

| Option                       | Default | Function                                                   |
| ---------------------------- | ------- | ---------------------------------------------------------- |
| **Enable Rebalancing**       | ☐       | Turn on/off order generation.                              |
| **Portfolio Value ($)**      | $100k   | Total cash + equity value for sizing.                      |
| **Min Trade Size**           | $100    | Filter out tiny trades to save fees.                       |
| **Round to 100 Lots**        | ☐       | Institutional execution mode.                              |

### 📁 Portfolio Data

Upload custom data to override defaults:

| Upload                       | Purpose                                                   |
| ---------------------------- | --------------------------------------------------------- |
| **Holdings CSV**       | Your current portfolio positions (Ticker, Weight, Sector) |
| **Sector Targets CSV** | Your target sector allocation (Sector, Weight)            |

---

## 🔧 Advanced Configuration (Expandable Tab)

Click **"Advanced Configuration"** in the sidebar to access detailed parameters:

### Portfolio Parameters

| Parameter                            | Range      | Default | What It Controls                                    |
| ------------------------------------ | ---------- | ------- | --------------------------------------------------- |
| **Risk Aversion (λ)**         | 0.5 - 10.0 | 2.0     | How much to penalize volatility in optimization     |
| **Concentration Penalty (γ)** | 0.0 - 50.0 | 0.5     | How much to penalize large single-stock allocations |

**Guidance:**

- **Lower Risk Aversion (0.5)**: More aggressive, higher returns but more volatile
- **Higher Risk Aversion (5.0+)**: More conservative, lower volatility
- **Higher Concentration Penalty**: Forces more diversification, prevents 20%+ allocations

### Allocation Constraints

| Parameter                    | Range        | Default | What It Controls                                        |
| ---------------------------- | ------------ | ------- | ------------------------------------------------------- |
| **Min Allocation (%)** | 0.0 - 50.0%  | 3.0%    | Minimum recommended allocation (rounds 0-3% down to 0%) |
| **Max Allocation (%)** | 0.0 - 100.0% | 25.0%   | Hard cap on any single stock allocation                 |

### Market Parameters

| Parameter                    | Default | What It Controls                                     |
| ---------------------------- | ------- | ---------------------------------------------------- |
| **Risk-Free Rate (%)** | 4.0%    | Current Treasury yield (used for Sharpe ratio, WACC) |
| **Benchmark Ticker**   | ACWX    | Benchmark for alpha calculation                      |
| **Correlation Ticker** | IXN     | Tech sector proxy for correlation checks             |

### Simulation Settings

| Parameter                    | Default    | What It Controls                                      |
| ---------------------------- | ---------- | ----------------------------------------------------- |
| **Lookback Years**     | 5          | Historical data period for volatility/correlation     |
| **Resample Frequency** | W (Weekly) | Data granularity: D (Daily), W (Weekly), M (Monthly)  |
| **Monte Carlo Sims**   | 10,000     | Number of simulations for valuation (higher = slower) |

**Tip**: Start with defaults, then fine-tune based on your risk tolerance and market expectations.

---

## 🚀 Running Analysis

### Step 1: Configure Options

- Set **Top N Stocks** or **Analyze ALL stocks**
- Toggle pipeline steps (e.g., skip valuation for speed)
- Optionally enable **Multi-Allocation** for detailed granularity
- Customize **Advanced Configuration** parameters if needed

### Step 2: View Screener Preview

- Once you select a file (upload or use default), a preview appears
- Shows first 5 rows and total row count

### Step 3: Click "🚀 Run Analysis"

- Real-time execution log displays in the interface
- Shows which modules are running and their progress
- Monitor for any errors in real-time

### Step 4: Download Results

- Once complete, a **"📥 Download Results (ZIP)"** button appears
- Contains organized analysis outputs for all stocks

---

## 📊 Understanding Results (Output Files)

After analysis completes, your ZIP archive contains:

```
analysis_results/
├── 0_portfolio/
│   ├── portfolio_risk_metrics.csv      # Overall portfolio stats
│   └── portfolio_weights.csv           # Sector allocation breakdown
│
├── {TICKER}/                           # Per-stock folder
│   ├── optimal_summary.csv             # Optimal allocation details
│   ├── backtest.csv                    # Impact on portfolio
│   ├── {TICKER}_multi_allocation.csv   # Allocation granularity (if enabled)
│   └── DCF_Valuation_Audit.md          # Detailed valuation breakdown
│
├── summary/
│   ├── master_summary.csv              # All stocks comparison
│   ├── optimal_summary.csv             # All optimal allocations
│   ├── valuation_summary.csv           # All valuations (DCF, MC, P/E discount)
│   └── analysis_report.txt             # Human-readable summary
│
└── input_screener.csv                  # Your input file (for reference)
```

### Key Output Metrics

| Metric                           | Found In                  | Interpretation                                     |
| -------------------------------- | ------------------------- | -------------------------------------------------- |
| **Optimal Allocation (%)** | `optimal_summary.csv`   | Recommended % to allocate to this stock            |
| **Utility (%)**            | `optimal_summary.csv`   | Improvement in risk-adjusted return vs no addition |
| **Sharpe Change (%)**      | `backtest.csv`          | Change to portfolio Sharpe ratio                   |
| **Return Change (%)**      | `backtest.csv`          | Change to portfolio annualized return              |
| **DCF Fair Value**         | `valuation_summary.csv` | Intrinsic value from cash flow projection          |
| **Margin of Safety (%)**   | `valuation_summary.csv` | How much upside (DCF - Price) / Price              |
| **Win Probability (%)**    | `valuation_summary.csv` | % of Monte Carlo simulations where upside exists   |

---

## ⚠️ Screener Quality Tips

For **meaningful analysis results**, your screener should exclude low-quality stocks:

| Filter                         | Why                                                   |
| ------------------------------ | ----------------------------------------------------- |
| **Market Cap > $1B**     | Micro-caps have unreliable financials and illiquidity |
| **Positive FCF**         | Negative FCF produces meaningless DCF values          |
| **P/E Ratio > 5**        | Avoids extreme value traps and distressed companies   |
| **Revenue > $100M**      | Ensures stable business with trackable growth         |
| **Beta between 0.5-2.5** | Avoids illiquid or highly speculative stocks          |

**Example screener query** (FinancialModelingPrep or similar):

```
marketCap > 1000000000 AND 
freeCashFlow > 0 AND 
peRatio > 5 AND 
revenue > 100000000
```

> **⚠️ Important**: Penny stocks and micro-caps will produce extreme/invalid values:
>
> - DCF Fair Value: $0.00 (no FCF)
> - P/E Discount: +35,000% (nonsensical)
> - Optimal Allocation: 0% (too volatile)

---

## 🔧 Module 1: Portfolio Reconstruction

### Purpose

Reconstruct a complete portfolio from **Top 10 holdings** + **sector ETF proxies** to match target sector allocations.

### Key Components

#### Default Top 10 Holdings

```python
AAPL (Tech, 7%), MSFT (Tech, 6%), NVDA (Tech, 5%), ASML (Tech, 4%),
SAP (Tech, 2.5%), REY.MI (Tech, 2%), IDR.MC (Industrial, 2%),
JPM (Financials, 3%), GS (Financials, 2.5%), HSBC (Financials, 2%)
```

#### Default Target Sector Weights

| Sector                 | Weight |
| ---------------------- | ------ |
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

#### Risk Metrics Calculated

| Metric                | Formula                                            |
| --------------------- | -------------------------------------------------- |
| Annualized Return     | $(1 + \text{Total Return})^{1/\text{years}} - 1$ |
| Annualized Volatility | $\sigma_{\text{weekly}} \times \sqrt{52}$        |
| Sharpe Ratio          | $(R_p - R_f) / \sigma_p$                         |
| Beta                  | $\text{Cov}(R_p, R_m) / \text{Var}(R_m)$         |
| Maximum Drawdown      | $\min[(\text{Cum} - \text{Peak}) / \text{Peak}]$ |

---

## 🆕 Module 2: Optimal Allocation Finder

### Purpose

**Automatically find the best allocation** for each candidate stock using **Mean-Variance Utility Maximization**.

### The Math

The allocation is chosen by maximizing:

$$
U = E[R] - \frac{\lambda}{2} \times \sigma^2 - \gamma \times w^2
$$

Where:

- **E[R]** = Expected annualized return
- **σ** = Annualized volatility
- **w** = Allocation weight (0 to 1)
- **λ** = Risk Aversion (from Advanced Config)
- **γ** = Concentration Penalty (from Advanced Config)

The term $-\gamma \times w^2$ acts as a **soft cap**, discouraging extreme allocations (like 25%) in favor of diversification (typically 5-15%).

### Output

Four visualization panels:

1. **Sharpe Curve** - Maximum Sharpe ratio (reference)
2. **Volatility Curve** - Minimum volatility (reference)
3. **Utility Curve** - Maximum utility (✅ **our method**)
4. **Efficient Frontier** - Risk vs Return scatter

Sample output:

```
   📈 Sharpe Optimization:  25.0% (reference)
   📊 Min Volatility:       12.3% (reference)
   🎯 Mean-Variance Utility: 18.4% (⭐ RECOMMENDED)
   
   Improvement: +0.74% utility vs no addition
```

---

## 🔧 Module 3: Candidate Backtesting

### Purpose

Simulate adding a **candidate stock** to your portfolio at the optimal allocation and measure the impact.

### What Gets Calculated

| Metric                             | What It Shows                                 |
| ---------------------------------- | --------------------------------------------- |
| **Return Change (%)**        | How much portfolio return increases/decreases |
| **Volatility Change (%)**    | How much portfolio risk increases/decreases   |
| **Sharpe Change**            | Impact on risk-adjusted returns               |
| **Beta Change**              | How much market sensitivity changes           |
| **Correlation vs Portfolio** | Diversification benefit (lower = better)      |
| **Max Drawdown Change (%)**  | Worst-case loss impact                        |
| **VaR 95% Impact**           | Change in 95% downside risk                   |

### Output Visualization (4 panels)

1. **Cumulative Returns** - Portfolio with vs without stock
2. **Rolling 52-Week Beta** - Market sensitivity over time
3. **Correlation Heatmap** - Stock vs all portfolio components
4. **Drawdown Comparison** - Worst periods side-by-side

---

## 🔧 Module 4: Valuation Engine

### Purpose

Compute **intrinsic value** using state-of-the-art probabilistic methods.

### Architecture

```
┌────────────────────────────────────────────────────┐
│         Multi-Method Valuation Engine              │
├────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ DataFetcher  │  │DCFValuation  │  │MonteCarlo │ │
│  │              │  │              │  │Valuation  │ │
│  │ - Revenue    │  │ - 10yr FCF   │  │           │ │
│  │ - FCF        │  │ - WACC       │  │ - 10,000  │ │
│  │ - Debt/Cash  │  │ - Terminal   │  │   trials  │ │
│  │ - Beta       │  │   Value      │  │ - PDF     │ │
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

### Methods

1. **DCF Valuation** - 10-year FCF projection + terminal value
2. **Monte Carlo** - 10,000 simulations (configurable) with randomized inputs
3. **Relative Valuation** - Regression-based P/E discount vs growth peers
4. **Sensitivity Analysis** - WACC × Terminal Growth impact matrix

### Key Outputs

| Output                         | Meaning                                         |
| ------------------------------ | ----------------------------------------------- |
| **DCF Fair Value**       | Expected intrinsic value                        |
| **Margin of Safety (%)** | How much upside: (DCF - Price) / Price          |
| **Win Probability (%)**  | % of Monte Carlo trials with positive upside    |
| **P/E Discount (%)**     | Relative valuation: Compare P/E to growth peers |

---

## 📦 Custom Data Formats

### Holdings CSV

Override default portfolio with your own holdings:

**Required columns**: `Ticker`, `Weight`, `Sector`

```csv
Ticker,Weight,Sector,Name
AAPL,7.0,Information Technology,Apple Inc
MSFT,6.0,Information Technology,Microsoft Corp
JPM,3.0,Financials,JPMorgan Chase
```

Upload via **"Upload Holdings CSV"** in sidebar.

### Sector Targets CSV

Define your target sector allocation:

**Required columns**: `Sector`, `Weight`

```csv
Sector,Weight
Information Technology,26.5
Financials,12.5
Health Care,9.5
Industrials,8.0
Energy,3.9
```

Upload via **"Upload Sector Targets CSV"** in sidebar.

---

## 📊 Sample Output: Master Summary

After analysis, review `master_summary.csv`:

| Ticker | Price         | DCF Value          | Margin of Safety | Win Prob | P/E Discount |
| ------ | ------------- | ------------------ | ---------------- | -------- | ------------ |
| UNH    | $287 | $1,033 | **+260%** 🟢 | 100%             | +81%     |              |
| V      | $322 | $345   | +7% 🟡             | 64%              | -7%      |              |
| TMO    | $579 | $288   | **-50%** 🔴  | 0%               | -23%     |              |

**How to interpret:**

- **UNH**: Massively undervalued. Strong FCF, high margin of safety.
- **V**: Fairly valued by DCF, but overvalued vs payment peers.
- **TMO**: Overvalued by both methods. Avoid.

---

## 🔮 Tips & Tricks

### For Speed

- Uncheck **"Analyze ALL stocks"** and set **Top N = 5**
- Uncheck **"Skip Valuation"** if you don't need DCF analysis
- Disable **Multi-Allocation** (expensive)
- Set **Monte Carlo Sims** to 5,000 instead of 10,000

### For Precision

- Enable **Multi-Allocation** to see allocation granularity
- Increase **Monte Carlo Sims** to 50,000
- Increase **Lookback Years** to 10 for stability
- Upload your own **Holdings CSV** for realistic backtest

### For Conservative Analysis

- Increase **Risk Aversion (λ)** to 5.0+
- Increase **Concentration Penalty (γ)** to 2.0+
- Increase **Min Allocation (%)** to 5.0%
- Decrease **Max Allocation (%)** to 15.0%

---

## 🔮 Future Enhancements

1. **Database caching** for API calls (avoid rate limits)
2. **Factor model integration** (Fama-French 5-factor)
3. **Options-implied volatility** for Monte Carlo inputs
4. **PDF report generation** with charts embedded
5. **User authentication** for multi-user access
6. **Saved configurations** for quick re-runs
7. **Sentiment analysis integration** (market mood)

---

Built with ❤️ for ECE Business Intelligence
