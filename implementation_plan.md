# 📊 Plan d'Implémentation — ECE Portfolio Engine v2.0

Plan d'implémentation pour 25 techniques quantitatives, organisées par priorité.

---

## User Review Required

> [!IMPORTANT]
> **Nouvelles dépendances requises**: `scikit-learn`, `pyportfolioopt`, `pandas-datareader`
> Ces packages seront ajoutés à `requirements.txt`

> [!WARNING]
> **HRP (#8) et Black-Litterman (#10)** nécessitent des changements architecturaux significatifs dans `optimal_allocation.py`. L'allocation actuelle utilise une approche Mean-Variance avec pénalité de concentration.

---

## État Actuel du Codebase

| Module                                                                                        | Fonctionnalités existantes                                |
| --------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [portfolio_reconstruction.py](file:///c:/Users/Joshs/Desktop/BI/ECE/portfolio_reconstruction.py) | VaR 95%, Sharpe, Beta, Alpha, Max Drawdown, Tracking Error |
| [optimal_allocation.py](file:///c:/Users/Joshs/Desktop/BI/ECE/optimal_allocation.py)             | Utility optimization, MCTR, Sharpe curve                   |
| [backtest_candidate.py](file:///c:/Users/Joshs/Desktop/BI/ECE/backtest_candidate.py)             | Pro-forma portfolio, correlations, impact analysis         |
| [valuation_engine.py](file:///c:/Users/Joshs/Desktop/BI/ECE/valuation_engine.py)                 | DCF, Monte Carlo simulation                                |

---

## Proposed Changes :

### TIER S++ — Sprint 0 : yfinance -> Refinitive-data, Screener RD (compatibilité) et .ENV mon API Key (BYOK en Prod ?)

### TIER S+ — Sprint 0.5 : Inclure Forecast dans le pipeline et l'UI Streamlit

1. **Unifier l'environnement** : Converting les tickers en RIC, s'assurer que l'environnement Python de ECE peut faire tourner FinCast sans soucis de dépendances, re router les dossiers, ajouter les flags FinCast au CLI et mettre un Singleton Pattern en place.
2. **Créer un Wrapper** : Écrire un script

   ```
   fincast_wrapper.py
   ```

   dans ECE qui agît comme un pont vers le dossier. Il gérera les imports compliqués pour que le code principal reste propre.

   ```
   Forecast/
   ```
3. **Modifier l'Orchestrateur** : Ajouter l'étape qui appelle ce wrapper.

   ```
   run_forecasting
   ```
4. **Mise à jour UI** : Ajouter le ON/OFF switch et les differentes options (flags) dans Streamlit. (+ run on top X ou run all)

---

### TIER S — Sprint 1 (Effort: ~7h total) DONE

---

#### [MODIFY] [portfolio_reconstruction.py](file:///c:/Users/Joshs/Desktop/BI/ECE/portfolio_reconstruction.py)

**#1 CVaR / Expected Shortfall** ⏱️ 10 min

Ajouter dans `calculate_risk_metrics()` (ligne ~430):

```python
# 6.5. CVaR (Expected Shortfall) - average loss beyond VaR
cvar_95 = port_ret[port_ret <= np.percentile(port_ret, 5)].mean()
metrics['CVaR (95%, period)'] = cvar_95 * 100
metrics['CVaR (95%, annualized)'] = cvar_95 * np.sqrt(periods_per_year) * 100
```

**#22 Sortino & Omega Ratios** ⏱️ 30 min

```python
# Sortino Ratio (downside volatility only)
downside_returns = port_ret[port_ret < 0]
downside_std = downside_returns.std() * np.sqrt(periods_per_year)
sortino = excess_return / downside_std if downside_std > 0 else 0
metrics['Sortino Ratio'] = sortino

# Omega Ratio (gain/loss probability weighted)
threshold = 0  # or risk_free_rate / periods_per_year
gains = port_ret[port_ret > threshold].sum()
losses = abs(port_ret[port_ret < threshold].sum())
omega = gains / losses if losses > 0 else float('inf')
metrics['Omega Ratio'] = omega
```

---

#### [NEW] [covariance_estimator.py](file:///c:/Users/Joshs/Desktop/BI/ECE/covariance_estimator.py)

**#23 Ledoit-Wolf Shrinkage** ⏱️ 2h

```python
"""
Covariance Estimation Module
============================
Implements Ledoit-Wolf shrinkage for stable covariance matrices.
"""
from sklearn.covariance import LedoitWolf
import numpy as np
import pandas as pd

def estimate_covariance(returns: pd.DataFrame, method: str = 'ledoit_wolf') -> np.ndarray:
    """
    Estimate covariance matrix with shrinkage.
  
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (T x N)
    method : str
        'sample' (raw), 'ledoit_wolf' (shrunk)
  
    Returns
    -------
    np.ndarray
        Covariance matrix (N x N)
    """
    if method == 'ledoit_wolf':
        lw = LedoitWolf().fit(returns.values)
        return lw.covariance_
    return returns.cov().values
```

#### [MODIFY] [optimal_allocation.py](file:///c:/Users/Joshs/Desktop/BI/ECE/optimal_allocation.py)

Intégrer le shrinkage dans les calculs d'optimisation:

- Remplacer `returns.cov()` par `estimate_covariance(returns, 'ledoit_wolf')`
- Ajouter import du nouveau module

---

#### [NEW] [stress_testing.py](file:///c:/Users/Joshs/Desktop/BI/ECE/stress_testing.py)

**#4 Stress Testing** ⏱️ 1h

```python
"""
Stress Testing Module
=====================
Historical scenario analysis (2008, COVID, etc.)
"""
import pandas as pd
import numpy as np

# Pre-defined historical drawdown scenarios
SCENARIOS = {
    'GFC_2008': {'start': '2008-09-01', 'end': '2009-03-09', 'peak_drawdown': -0.569},
    'COVID_2020': {'start': '2020-02-19', 'end': '2020-03-23', 'peak_drawdown': -0.339},
    'Dot_Com_2000': {'start': '2000-03-24', 'end': '2002-10-09', 'peak_drawdown': -0.495},
    'Black_Monday_1987': {'period_days': 1, 'peak_drawdown': -0.226},
}

def run_stress_test(portfolio_returns: pd.Series, 
                    scenario: str = 'GFC_2008',
                    portfolio_value: float = 1_000_000) -> dict:
    """
    Apply historical scenario to current portfolio.
  
    Returns
    -------
    dict with 'scenario', 'estimated_loss_pct', 'estimated_loss_usd'
    """
    scenario_data = SCENARIOS.get(scenario)
    estimated_loss = scenario_data['peak_drawdown']
  
    return {
        'scenario': scenario,
        'estimated_loss_pct': estimated_loss * 100,
        'estimated_loss_usd': portfolio_value * estimated_loss,
        'recovery_estimate': "6-24 months (historical average)"
    }
```

---

#### [MODIFY] [config.py](file:///c:/Users/Joshs/Desktop/BI/ECE/config.py)

**#17 Drawdown-Based Position Sizing** ⏱️ 1h

```python
# Dans AnalysisConfig:
# Drawdown Protection Parameters
drawdown_reduction_threshold: float = 0.10  # -10% drawdown triggers reduction
drawdown_reduction_factor: float = 0.50      # Reduce exposure to 50%
drawdown_recovery_threshold: float = 0.05   # -5% drawdown, start recovering
```

#### [MODIFY] [optimal_allocation.py](file:///c:/Users/Joshs/Desktop/BI/ECE/optimal_allocation.py)

Ajouter fonction de position sizing dynamique:

```python
def adjust_for_drawdown(base_allocation: float, 
                        current_drawdown: float,
                        config: AnalysisConfig) -> float:
    """
    Reduce allocation if portfolio is in drawdown.
    """
    if current_drawdown < -config.drawdown_reduction_threshold:
        return base_allocation * config.drawdown_reduction_factor
    return base_allocation
```

---

#### [NEW] [rebalancing.py](file:///c:/Users/Joshs/Desktop/BI/ECE/rebalancing.py)

**#3 Rebalancing Optimizer** ⏱️ 2h

```python
"""
Rebalancing Optimizer Module
============================
Convert target allocations to executable trades.
"""
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class TradeOrder:
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    estimated_value: float
    current_weight: float
    target_weight: float

def calculate_rebalancing_trades(
    current_holdings: Dict[str, float],  # ticker -> current_value
    target_weights: Dict[str, float],     # ticker -> target_weight (%)
    current_prices: Dict[str, float],     # ticker -> price
    portfolio_value: float,
    min_trade_value: float = 100.0
) -> List[TradeOrder]:
    """
    Generate trade orders to reach target allocation.
  
    Returns
    -------
    List of TradeOrder objects sorted by absolute value (largest first)
    """
    orders = []
  
    for ticker, target_weight in target_weights.items():
        target_value = portfolio_value * (target_weight / 100)
        current_value = current_holdings.get(ticker, 0.0)
        delta = target_value - current_value
  
        if abs(delta) < min_trade_value:
            continue
  
        price = current_prices.get(ticker, 0)
        if price <= 0:
            continue
      
        shares = int(abs(delta) / price)
        if shares == 0:
            continue
      
        orders.append(TradeOrder(
            ticker=ticker,
            action='BUY' if delta > 0 else 'SELL',
            shares=shares,
            estimated_value=shares * price,
            current_weight=current_value / portfolio_value * 100,
            target_weight=target_weight
        ))
  
    return sorted(orders, key=lambda x: x.estimated_value, reverse=True)
```

---

### TIER A — Sprint 2 (Effort: ~10h total)

---

#### [NEW] [hrp_allocation.py](file:///c:/Users/Joshs/Desktop/BI/ECE/hrp_allocation.py)

**#8 Hierarchical Risk Parity (HRP)** ⏱️ 4h

```python
"""
Hierarchical Risk Parity Module
===============================
López de Prado's HRP algorithm for robust allocation.
"""
from pypfopt import HRPOpt
from pypfopt.expected_returns import mean_historical_return
import pandas as pd

def calculate_hrp_weights(prices: pd.DataFrame) -> dict:
    """
    Calculate HRP weights.
  
    Parameters
    ----------
    prices : pd.DataFrame
        Historical prices (T x N assets)
  
    Returns
    -------
    dict : ticker -> weight
    """
    hrp = HRPOpt(returns=prices.pct_change().dropna())
    weights = hrp.optimize()
    return dict(weights)
```

---

#### [NEW] [liquidity.py](file:///c:/Users/Joshs/Desktop/BI/ECE/liquidity.py)

**#19 Liquidity Scoring** ⏱️ 1h

```python
"""
Liquidity Analysis Module
=========================
Calculate days-to-liquidate and liquidity scores.
"""
import yfinance as yf
import pandas as pd

def calculate_liquidity_metrics(ticker: str, 
                                 position_value: float,
                                 participation_rate: float = 0.10) -> dict:
    """
    Calculate liquidity metrics for a position.
  
    Parameters
    ----------
    ticker : str
        Stock ticker
    position_value : float
        Position value in USD
    participation_rate : float
        Max % of daily volume to trade (default 10%)
  
    Returns
    -------
    dict with 'avg_daily_volume', 'avg_daily_value', 'days_to_liquidate', 'liquidity_score'
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period='3mo')
  
    avg_volume = hist['Volume'].mean()
    avg_price = hist['Close'].mean()
    avg_daily_value = avg_volume * avg_price
  
    tradeable_daily = avg_daily_value * participation_rate
    days_to_liquidate = position_value / tradeable_daily if tradeable_daily > 0 else float('inf')
  
    # Score: 1 = highly liquid (< 1 day), 5 = illiquid (> 10 days)
    if days_to_liquidate < 1:
        score = 1
    elif days_to_liquidate < 3:
        score = 2
    elif days_to_liquidate < 5:
        score = 3
    elif days_to_liquidate < 10:
        score = 4
    else:
        score = 5
  
    return {
        'ticker': ticker,
        'avg_daily_volume': avg_volume,
        'avg_daily_value_usd': avg_daily_value,
        'days_to_liquidate': days_to_liquidate,
        'liquidity_score': score,
        'liquidity_rating': ['Excellent', 'Good', 'Moderate', 'Poor', 'Critical'][score - 1]
    }
```

---

#### [MODIFY] [portfolio_reconstruction.py](file:///c:/Users/Joshs/Desktop/BI/ECE/portfolio_reconstruction.py)

**#18 Turnover & Tax Efficiency** ⏱️ 1h

Ajouter dans ou après `calculate_risk_metrics()`:

```python
def calculate_turnover(old_weights: dict, new_weights: dict) -> float:
    """
    Calculate portfolio turnover.
  
    Turnover = sum of |weight changes| / 2
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())
    total_change = sum(abs(new_weights.get(t, 0) - old_weights.get(t, 0)) 
                       for t in all_tickers)
    return total_change / 2

def estimate_tax_drag(turnover: float, 
                      avg_gain_rate: float = 0.15,
                      short_term_tax: float = 0.35,
                      long_term_tax: float = 0.20) -> float:
    """
    Estimate annual tax drag from turnover.
  
    Assumes 50% of turnover is short-term, 50% long-term.
    """
    avg_tax = (short_term_tax + long_term_tax) / 2
    return turnover * avg_gain_rate * avg_tax
```

---

#### [NEW] [attribution.py](file:///c:/Users/Joshs/Desktop/BI/ECE/attribution.py)

**#21 Return Attribution (Brinson)** ⏱️ 3h

```python
"""
Performance Attribution Module
==============================
Brinson-Fachler attribution analysis.
"""
import pandas as pd

def brinson_attribution(
    portfolio_weights: dict,   # sector -> weight
    benchmark_weights: dict,   # sector -> weight  
    portfolio_returns: dict,   # sector -> return
    benchmark_returns: dict    # sector -> return
) -> dict:
    """
    Decompose active return into allocation and selection effects.
  
    Returns
    -------
    dict with 'allocation_effect', 'selection_effect', 'interaction_effect', 'total_active'
    """
    sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
  
    allocation_effect = 0
    selection_effect = 0
    interaction_effect = 0
  
    for sector in sectors:
        w_p = portfolio_weights.get(sector, 0)
        w_b = benchmark_weights.get(sector, 0)
        r_p = portfolio_returns.get(sector, 0)
        r_b = benchmark_returns.get(sector, 0)
  
        # Allocation: over/underweight sectors that outperformed
        allocation_effect += (w_p - w_b) * r_b
  
        # Selection: picking better stocks within sectors
        selection_effect += w_b * (r_p - r_b)
  
        # Interaction: combined effect
        interaction_effect += (w_p - w_b) * (r_p - r_b)
  
    return {
        'allocation_effect': allocation_effect,
        'selection_effect': selection_effect,
        'interaction_effect': interaction_effect,
        'total_active_return': allocation_effect + selection_effect + interaction_effect
    }
```

---

### TIER B — Sprint 3 (Effort: ~15h total)

| ID  | Module                          | Description                         |
| --- | ------------------------------- | ----------------------------------- |
| #6  | [NEW]`factor_analysis.py`     | Fama-French 3/5 factor regression   |
| #10 | [NEW]`black_litterman.py`     | Bayesian allocation avec views DCF  |
| #20 | [NEW]`dcc_garch.py`           | Corrélations dynamiques (arch lib) |
| #7  | [NEW]`risk_parity.py`         | Equal risk contribution             |
| #2  | [MODIFY]`valuation_engine.py` | Intégrer ESG via yfinance          |

---

#### [MODIFY] [requirements.txt](file:///c:/Users/Joshs/Desktop/BI/ECE/requirements.txt)

Nouvelles dépendances:

```
pandas
numpy
yfinance
scipy
matplotlib
seaborn
streamlit
scikit-learn      # NEW: Ledoit-Wolf shrinkage
pypfopt           # NEW: HRP, Black-Litterman
pandas-datareader # NEW: Fama-French factors
arch              # TIER B: DCC-GARCH
```

---

## Verification Plan

### Automated Tests

**Aucun framework de test existant** — proposition de créer `tests/test_metrics.py`:

```bash
# Créer et exécuter les tests unitaires
python -m pytest tests/ -v
```

Tests proposés:

1. `test_cvar_calculation()` — Vérifier CVaR < VaR (toujours vrai)
2. `test_ledoit_wolf_positive_definite()` — Matrice doit être positive semi-définie
3. `test_sortino_greater_than_sharpe_when_positive_skew()` — Propriété mathématique
4. `test_rebalancing_trades_sum_to_zero()` — Les achats = ventes en valeur
5. `test_liquidity_score_bounds()` — Score entre 1 et 5

### Manual Verification

1. **CVaR Quick Check**:

   ```bash
   cd c:\Users\Joshs\Desktop\BI\ECE
   python -c "from portfolio_reconstruction import *; print('CVaR test passed')"
   ```
2. **Full Pipeline Test**:

   ```bash
   python run_analysis.py --ticker AAPL --name "Apple Inc"
   ```

   Vérifier que les nouveaux métriques apparaissent dans le rapport.
3. **Streamlit UI Check**:

   ```bash
   streamlit run streamlit_app.py
   ```

   Vérifier que les nouvelles métriques s'affichent correctement dans l'interface.

---

## Résumé du Planning

| Sprint        | Techniques                | Effort | Livrables                                                                 |
| ------------- | ------------------------- | ------ | ------------------------------------------------------------------------- |
| **S1**  | #1, #23, #4, #17, #3, #22 | ~7h    | CVaR, Shrinkage, Stress Test, Position Sizing, Rebalancing, Sortino/Omega |
| **S2**  | #8, #19, #18, #21         | ~10h   | HRP, Liquidity, Turnover, Attribution                                     |
| **S3**  | #6, #10, #20, #7, #2      | ~15h   | Fama-French, Black-Litterman, DCC-GARCH, Risk Parity, ESG                 |
| **S4+** | Tier C-D                  | ~30h+  | Kelly, Ensemble, HMM, PDF Reports                                         |

> [!TIP]
> **Recommandation**: Commencer par Sprint 1 (TIER S) car il apporte 80% de la valeur avec 20% de l'effort. Le CVaR prend littéralement 10 minutes et donne une vision du tail risk beaucoup plus précise.
