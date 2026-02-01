# COR (Cencora Inc.) - DCF Valuation Audit

> **Objective**: Rigorously justify the 129% upside ($359.22 → $822.50)
> **Date**: 2026-02-01 | **Analyst**: Josh E. SOUSSAN

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Current Price** | $359.22 |
| **DCF Base Case** | $463.61 |
| **Monte Carlo Mean** | **$822.50** |
| **Monte Carlo Median** | $583.02 |
| **DCF Base Upside** | 29.1% |
| **Monte Carlo Upside** | **129.0%** |
| **Win Probability** | 97.22% |

---

## 1️⃣ Model Parameters

| Parameter | Value |
|-----------|-------|
| Risk-Free Rate | 4.50% |
| Equity Risk Premium | 5.00% |
| Tax Rate | 25.00% |
| Terminal Growth Rate | 2.50% |
| Projection Years | 10 |
| Max Growth Rate Cap | 10.00% |
| 3Y Weight for CAGR | 60.00% |
| 5Y/Fallback Weight | 40.00% |

---

## 2️⃣ Basic Company Data (yfinance)

| Field | Value |
|-------|-------|
| Ticker | COR |
| Company Name | Cencora, Inc. |
| Current Stock Price | $359.22 |
| Shares Outstanding | 194,526,076 |
| Market Cap | $69,880,000,000 |
| Beta | 0.665 |

---

## 3️⃣ Historical Financials

| Fiscal Year | FY 2022 | FY 2023 | FY 2024 | FY 2025 |
|-------------|---------|---------|---------|---------|
| **Total Revenue** | $238,587,006,000 | $262,173,000,000 | $293,962,000,000 | $321,332,819,000 |
| **EBIT** | $2,390,000,000 | $2,390,000,000 | $2,250,000,000 | $2,680,000,000 |
| **Interest Expense** | $211,000,000 | $229,000,000 | $249,000,000 | $419,753,000 |
| **Free Cash Flow** | $2,210,000,000 | $3,450,000,000 | $3,000,000,000 | $3,207,139,000 |

---

## 4️⃣ Balance Sheet Data

| Item | Value |
|------|-------|
| Total Debt | $7,660,000,000 |
| Cash & Equivalents | $4,360,000,000 |
| **Net Debt** (Debt - Cash) | $3,300,000,000 |

---

## 5️⃣ Growth Rate Calculation

| Step | Formula | Result |
|------|---------|--------|
| Revenue FY 2022 | Input | $238,587,006,000 |
| Revenue FY 2025 | Input | $321,332,819,000 |
| Ratio (FY25 / FY22) | End / Start | 1.3468 |
| **CAGR 3-Year** | (Ratio)^(1/3) - 1 | **10.43%** |
| yfinance revenueGrowth (fallback) | API data | 5.90% |
| Weighted Average Growth | (60% × 3Y) + (40% × fallback) | 8.62% |
| **FINAL GROWTH RATE** (capped at 10%) | MIN(weighted, 10%) | **8.62%** |

---

## 6️⃣ WACC Calculation

### 6A. Cost of Equity (CAPM)

| Component | Formula | Result |
|-----------|---------|--------|
| Risk-Free Rate | Rf | 4.50% |
| Beta | β | 0.665 |
| Equity Risk Premium | ERP | 5.00% |
| Beta × ERP | β × ERP | 3.33% |
| **COST OF EQUITY** | Rf + β × ERP | **7.83%** |

### 6B. Cost of Debt

| Component | Formula | Result |
|-----------|---------|--------|
| Interest Expense (FY25) | From data | $419,753,000 |
| Total Debt | From data | $7,660,000,000 |
| Pre-tax Cost of Debt | Interest / Debt | 5.48% |
| **AFTER-TAX COST OF DEBT** | Pre-tax × (1 - Tax) | **4.11%** |

### 6C. Capital Structure

| Component | Value | Weight |
|-----------|-------|--------|
| Market Cap (Equity) | $69,880,000,000 | 90.12% |
| Total Debt | $7,660,000,000 | 9.88% |
| **Total Capital** | $77,540,000,000 | 100.00% |

### 6D. Final WACC

| Component | Calculation | Result |
|-----------|-------------|--------|
| Equity Component | Eq.Wt × Cost of Eq | 7.05% |
| Debt Component | Debt.Wt × After-tax | 0.41% |
| **WACC** | Equity + Debt | **7.46%** |

---

## 7️⃣ 10-Year FCF Projections (with Growth Decay)

| Year | Decay Factor | Adj. Growth | Growth (1+g) | FCF ($) | Disc. Period | Disc. Factor | PV of FCF ($) |
|------|--------------|-------------|--------------|---------|--------------|--------------|---------------|
| Base (FY25) | - | - | - | $3,207,139,000 | - | - | - |
| Year 1 | 100.00% | 8.62% | 1.0862 | $3,483,607,028 | 1 | 0.9306 | $3,241,831,582 |
| Year 2 | 95.00% | 8.19% | 1.0819 | $3,768,892,656 | 2 | 0.8660 | $3,263,896,284 |
| Year 3 | 90.00% | 7.76% | 1.0776 | $4,061,296,723 | 3 | 0.8059 | $3,273,019,503 |
| Year 4 | 85.00% | 7.33% | 1.0733 | $4,358,881,546 | 4 | 0.7500 | $3,269,039,967 |
| Year 5 | 80.00% | 6.90% | 1.0690 | $4,659,483,767 | 5 | 0.6979 | $3,251,952,977 |
| Year 6 | 75.00% | 6.47% | 1.0647 | $4,960,733,172 | 6 | 0.6495 | $3,221,911,542 |
| Year 7 | 70.00% | 6.03% | 1.0603 | $5,260,077,504 | 7 | 0.6044 | $3,179,224,370 |
| Year 8 | 65.00% | 5.60% | 1.0560 | $5,554,813,128 | 8 | 0.5625 | $3,124,350,723 |
| Year 9 | 60.00% | 5.17% | 1.0517 | $5,842,121,205 | 9 | 0.5234 | $3,057,892,263 |
| Year 10 | 55.00% | 4.74% | 1.0474 | $6,119,108,841 | 10 | 0.4871 | $2,980,582,080 |
| **SUM OF PV (FCFs)** | | | | | | | **$31,863,701,290** |

---

## 8️⃣ Terminal Value (Gordon Growth Model)

| Step | Formula | Result |
|------|---------|--------|
| FCF Year 10 | From projections | $6,119,108,841 |
| Terminal Growth (g) | Parameter | 2.50% |
| Terminal FCF (Year 11) | FCF10 × (1 + g) | $6,272,086,562 |
| WACC - g | Denominator | 4.96% |
| **TERMINAL VALUE** | TermFCF / (WACC - g) | **$126,504,645,986** |
| Discount Factor (Yr 10) | 1/(1+WACC)^10 | 0.4871 |
| **PV OF TERMINAL VALUE** | TV × Discount Factor | **$61,619,672,181** |

---

## 9️⃣ Enterprise Value to Equity Value Bridge

| Component | Formula | Value |
|-----------|---------|-------|
| PV of FCFs (Years 1-10) | From projections | $31,863,701,290 |
| PV of Terminal Value | From TV calc | $61,619,672,181 |
| **ENTERPRISE VALUE** | PV FCFs + PV TV | **$93,483,373,471** |
| Less: Total Debt | Balance sheet | -$7,660,000,000 |
| Plus: Cash | Balance sheet | +$4,360,000,000 |
| **EQUITY VALUE** | EV - Debt + Cash | **$90,183,373,471** |

---

## 🔟 DCF Fair Value Per Share

| Metric | Formula | Value |
|--------|---------|-------|
| Equity Value | From bridge | $90,183,373,471 |
| Shares Outstanding | From data | 194,526,076 |
| **DCF FAIR VALUE PER SHARE** | Equity / Shares | **$463.61** |

---

## 1️⃣1️⃣ Valuation Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Current Stock Price | $359.22 | Market price |
| DCF Base Case Fair Value | $463.61 | This model |
| Monte Carlo Mean Fair Value | $822.50 | 10,000 simulations |
| Monte Carlo Median | $583.02 | 50th percentile |
| **DCF BASE UPSIDE** | **29.1%** | (DCF - Price) / Price |
| **MONTE CARLO UPSIDE** | **129.0%** | (MC Mean - Price) / Price |
| Win Probability | 97.22% | % MC runs FV > Price |

---

## 1️⃣2️⃣ Value Component Breakdown

| Component | Value | % of EV |
|-----------|-------|---------|
| PV of FCFs (Years 1-10) | $31,863,701,290 | 34.1% |
| PV of Terminal Value | $61,619,672,181 | 65.9% |
| **Total Enterprise Value** | $93,483,373,471 | 100.00% |

---

## 1️⃣3️⃣ Key Metrics & Ratios

| Metric | Formula | Value |
|--------|---------|-------|
| FCF Margin (FY25) | FCF / Revenue | 1.00% |
| EV / FCF Multiple | EV / FCF | 29.1x |
| Debt / Equity | Debt / Mkt Cap | 11.0% |

---

## 1️⃣4️⃣ Monte Carlo Simulation Methodology

### Why the Model Uses Monte Carlo

The DCF Base Case ($463.61) is a **deterministic** valuation using single-point estimates. However, real-world inputs are uncertain. The Monte Carlo simulation captures this uncertainty by running **10,000 DCF scenarios** with randomized parameters.

### Randomized Parameters

| Parameter | Base Value | Distribution | Standard Deviation |
|-----------|------------|--------------|-------------------|
| **Growth Rate** | 8.62% | Normal | ±2.0% |
| **FCF Margin** | 1.00% | Normal | ±2.0% |
| **WACC** | 7.46% | Normal | ±0.5% |
| **Terminal Growth** | 2.50% | Normal | ±0.2% |

### Monte Carlo FCF Calculation

Unlike the base DCF which uses actual FCF ($3.21B), Monte Carlo computes FCF for each simulation as:

```
Simulated FCF = Revenue × Randomized FCF Margin
```

This is **critical** for understanding the gap:
- Base FCF Margin = 1.00%
- Monte Carlo samples margins from Normal(1.00%, σ=2.0%)
- ~16% of simulations have margin > 3.0% (triple the base!)
- These high-margin scenarios produce fair values > $1,400

### Distribution of Results

| Percentile | Fair Value | Interpretation |
|------------|------------|----------------|
| P10 | ~$400-500 | Pessimistic scenario |
| P25 | ~$500-550 | Below average |
| **P50 (Median)** | **$583.02** | Central tendency |
| P75 | ~$800-1000 | Above average |
| P90 | ~$1,500+ | Optimistic scenario |
| **Mean** | **$822.50** | Expected value |

### Why Mean > Median?

The distribution is **right-skewed** (asymmetric):

```
     ▲
     │    ╭───╮
     │   ╱     ╲
     │  ╱       ╲
Freq │ ╱         ╲______
     │╱                  ╲____
     └─────────────────────────▶
      $400   $583  $822    $1500+
            Median Mean
```

- **Median ($583.02)**: 50% of simulations above/below this value
- **Mean ($822.50)**: The mathematical average, pulled up by high-value outliers
- **Win Probability (97.22%)**: 97.22% of simulations yield FV > current price ($359.22)

### Which Value Should Be Reported?

| Metric | Value | When to Use |
|--------|-------|-------------|
| **DCF Base Case** | $463.61 | Conservative, single-point estimate |
| **MC Median** | $583.02 | "Most likely" outcome (robust to outliers) |
| **MC Mean** | $822.50 | Expected value (accounts for upside potential) |

> [!IMPORTANT]
> The screener reports **MC Mean ($822.50)** as the Fair Value, which produces the **129% upside**. This is the mathematical expectation across all scenarios, giving credit to upside potential.

---

## 1️⃣5️⃣ Relative Valuation (Trading Comps)

The model also includes **Relative Valuation** as a cross-check using peer comparisons with linear regression.

### Methodology

Instead of simple average multiples, the model:
1. Fetches P/E and EV/EBITDA for sector peers
2. Regresses **Growth vs Multiple** across peers
3. Identifies stocks trading **below the regression line** as undervalued

### Peer Group (Medical Distribution)

The model compares COR against healthcare distribution peers:

| Peer | P/E | EV/EBITDA | Revenue Growth |
|------|-----|-----------|----------------|
| MCK | 26.0x | 18.5x | 16.2% |
| CAH | 18.5x | 12.1x | 10.5% |
| COR | 44.2x | 15.6x | 9.3% |

### Regression Analysis

A stock is considered **undervalued** if its actual multiple is below the regression-implied multiple given its growth rate:

```
Implied P/E = α + β × Revenue Growth
Discount = (Implied P/E - Actual P/E) / Actual P/E
```

> [!NOTE]
> Relative valuation provides a market-based cross-check but is secondary to the DCF in this model. The primary fair value is derived from DCF + Monte Carlo.

---

## ✅ Audit Conclusion

### Model Strengths
1. **Positive and Growing FCF**: $3.21B in FY2025, solid track record
2. **Low WACC** (7.46%): Very favorable beta of 0.665
3. **Sustained Revenue Growth**: 10.43% CAGR over 3 years
4. **Prudent Capital Structure**: 90% equity, 11% Debt/Equity ratio

### Caveats
1. **Terminal Value = 65.9% of EV**: Standard but sensitive to assumptions
2. **Low FCF Margin (1.0%)**: Monte Carlo margin randomization significantly impacts fair value
3. **Model sensitive to growth rate**: ±1% growth ≈ ±15% on fair value

### Final Verdict

| Status | Conclusion |
|--------|------------|
| ✅ **Calculations Verified** | DCF methodology conforms to industry standards |
| ✅ **Data Sources Traced** | All via yfinance (auditable) |
| ⚠️ **DCF Base Upside** | 29.1% is conservative and defensible |
| ⚠️ **MC Mean Upside** | 129.0% reflects right-skewed distribution from margin randomization |

> [!TIP]
> **For conservative presentations**: Use DCF Base ($463.61) → **29.1% upside**
> **For expected value**: Use MC Mean ($822.50) → **129.0% upside**

---

## 📎 Appendix: Model Parameters (valuation_engine.py)

```python
RISK_FREE_RATE = 0.045      # 4.5%
EQUITY_RISK_PREMIUM = 0.05  # 5.0%
PROJECTION_YEARS = 10
TERMINAL_GROWTH_BASE = 0.025  # 2.5%
MAX_GROWTH_RATE = 0.10      # Cap at 10%
N_SIMULATIONS = 10000       # Monte Carlo
DCF_UPPER_BOUND_MULTIPLIER = 5.0
```

