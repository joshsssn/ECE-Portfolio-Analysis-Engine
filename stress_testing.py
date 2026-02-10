"""
ECE Portfolio Analysis Engine
=============================
Stress Testing Module
=====================
Historical scenario analysis (GFC 2008, COVID 2020, etc.)
Answers: "If 2008 happens again, how much do we lose?"

Author: Josh E. SOUSSAN
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


# =============================================================================
# HISTORICAL SCENARIOS
# =============================================================================

HISTORICAL_SCENARIOS = {
    'GFC_2008': {
        'name': 'Global Financial Crisis (2008-2009)',
        'start': '2008-09-15',  # Lehman Brothers bankruptcy
        'end': '2009-03-09',    # Market bottom
        'peak_drawdown': -0.569,
        'duration_days': 175,
        'recovery_months': 49,
        'description': 'Subprime mortgage crisis, Lehman collapse'
    },
    'COVID_2020': {
        'name': 'COVID-19 Crash (2020)',
        'start': '2020-02-19',
        'end': '2020-03-23',
        'peak_drawdown': -0.339,
        'duration_days': 33,
        'recovery_months': 5,
        'description': 'Pandemic-induced market crash'
    },
    'DOT_COM_2000': {
        'name': 'Dot-Com Bubble (2000-2002)',
        'start': '2000-03-24',
        'end': '2002-10-09',
        'peak_drawdown': -0.495,
        'duration_days': 929,
        'recovery_months': 56,
        'description': 'Tech bubble burst'
    },
    'BLACK_MONDAY_1987': {
        'name': 'Black Monday (1987)',
        'start': '1987-10-19',
        'end': '1987-10-19',
        'peak_drawdown': -0.226,
        'duration_days': 1,
        'recovery_months': 21,
        'description': 'Single-day market crash'
    },
    'EURO_CRISIS_2011': {
        'name': 'European Debt Crisis (2011)',
        'start': '2011-07-22',
        'end': '2011-10-03',
        'peak_drawdown': -0.215,
        'duration_days': 73,
        'recovery_months': 6,
        'description': 'Greek debt crisis, eurozone contagion'
    },
    'INFLATION_2022': {
        'name': 'Inflation/Rate Hike Crash (2022)',
        'start': '2022-01-03',
        'end': '2022-10-12',
        'peak_drawdown': -0.274,
        'duration_days': 282,
        'recovery_months': 14,
        'description': 'Fed rate hikes, inflation shock'
    }
}


@dataclass
class StressTestResult:
    """Container for stress test results."""
    scenario_id: str
    scenario_name: str
    portfolio_beta: float
    market_drawdown: float
    estimated_drawdown: float
    portfolio_value: float
    estimated_loss_usd: float
    estimated_loss_pct: float
    recovery_estimate_months: int
    description: str


# =============================================================================
# STRESS TESTING FUNCTIONS
# =============================================================================

def calculate_portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate portfolio beta vs benchmark.
    
    Beta > 1 = amplifies market moves
    Beta < 1 = dampens market moves
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    port = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
    
    covariance = port.cov(bench)
    variance = bench.var()
    
    return covariance / variance if variance != 0 else 1.0


def run_stress_test(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    scenario: str = 'GFC_2008',
    portfolio_value: float = 1_000_000,
    custom_beta: Optional[float] = None
) -> StressTestResult:
    """
    Apply historical scenario to estimate portfolio impact.
    
    Uses portfolio beta to scale market drawdown to portfolio-specific impact.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Historical portfolio returns
    benchmark_returns : pd.Series
        Historical benchmark returns
    scenario : str
        Scenario key from HISTORICAL_SCENARIOS
    portfolio_value : float
        Current portfolio value in USD
    custom_beta : float, optional
        Override calculated beta (useful for hypothetical analysis)
    
    Returns
    -------
    StressTestResult
        Estimated portfolio impact under the scenario
    """
    if scenario not in HISTORICAL_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(HISTORICAL_SCENARIOS.keys())}")
    
    scenario_data = HISTORICAL_SCENARIOS[scenario]
    
    # Calculate or use provided beta
    if custom_beta is not None:
        beta = custom_beta
    else:
        beta = calculate_portfolio_beta(portfolio_returns, benchmark_returns)
    
    # Scale market drawdown by beta
    market_drawdown = scenario_data['peak_drawdown']
    estimated_drawdown = market_drawdown * beta
    
    # Cap at -100% (can't lose more than everything)
    estimated_drawdown = max(estimated_drawdown, -1.0)
    
    estimated_loss_usd = portfolio_value * estimated_drawdown
    
    return StressTestResult(
        scenario_id=scenario,
        scenario_name=scenario_data['name'],
        portfolio_beta=beta,
        market_drawdown=market_drawdown,
        estimated_drawdown=estimated_drawdown,
        portfolio_value=portfolio_value,
        estimated_loss_usd=estimated_loss_usd,
        estimated_loss_pct=estimated_drawdown * 100,
        recovery_estimate_months=scenario_data['recovery_months'],
        description=scenario_data['description']
    )


def run_all_stress_tests(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_value: float = 1_000_000
) -> Dict[str, StressTestResult]:
    """
    Run all historical stress tests on the portfolio.
    
    Returns
    -------
    dict
        scenario_id -> StressTestResult
    """
    results = {}
    
    for scenario_id in HISTORICAL_SCENARIOS:
        results[scenario_id] = run_stress_test(
            portfolio_returns, 
            benchmark_returns, 
            scenario_id, 
            portfolio_value
        )
    
    return results


def display_stress_test_results(results: Dict[str, StressTestResult]):
    """
    Display stress test results in a formatted table.
    """
    print("\n" + "="*80)
    print("STRESS TEST RESULTS")
    print("="*80 + "\n")
    
    for scenario_id, result in results.items():
        print(f">> {result.scenario_name}")
        print(f"   Market Drawdown:    {result.market_drawdown*100:>7.1f}%")
        print(f"   Portfolio Beta:     {result.portfolio_beta:>7.2f}")
        print(f"   Estimated Loss:     {result.estimated_loss_pct:>7.1f}%  (${abs(result.estimated_loss_usd):,.0f})")
        print(f"   Recovery Estimate:  {result.recovery_estimate_months:>4d} months")
        print()


def stress_test_summary_df(results: Dict[str, StressTestResult]) -> pd.DataFrame:
    """
    Convert stress test results to DataFrame for export.
    """
    data = []
    for r in results.values():
        data.append({
            'Scenario': r.scenario_name,
            'Market Drawdown (%)': r.market_drawdown * 100,
            'Portfolio Beta': r.portfolio_beta,
            'Estimated Loss (%)': r.estimated_loss_pct,
            'Estimated Loss ($)': r.estimated_loss_usd,
            'Recovery (months)': r.recovery_estimate_months
        })
    
    return pd.DataFrame(data).set_index('Scenario')


if __name__ == "__main__":
    # Quick demo with synthetic data
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("="*60)
    print("STRESS TESTING MODULE - DEMO")
    print("="*60)
    
    # Download sample data
    end = datetime.now()
    start = end - timedelta(days=365 * 5)
    
    spy = yf.download('SPY', start=start, end=end, progress=False)['Close'].pct_change().dropna()
    spy.name = 'Portfolio'
    
    acwi = yf.download('ACWI', start=start, end=end, progress=False)['Close'].pct_change().dropna()
    acwi.name = 'Benchmark'
    
    # Run stress tests
    results = run_all_stress_tests(spy, acwi, portfolio_value=1_000_000)
    display_stress_test_results(results)
    
    # Show summary DataFrame
    df = stress_test_summary_df(results)
    print("\n--- Summary Table ---")
    print(df.to_string())
