"""
Optimal Allocation Finder
=========================
Automatically finds the optimal allocation for a candidate stock using:
1. Sharpe Ratio Optimization - Find allocation that maximizes risk-adjusted return
2. Risk Budgeting (MCTR) - Find allocation where marginal risk contribution is optimal

Author: Josh E. SOUSSAN
Usage: python optimal_allocation.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar, minimize
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Candidate stock to optimize
CANDIDATE_TICKER = 'UNH'
CANDIDATE_NAME = 'UnitedHealth Group'

# Allocation search range
MIN_ALLOCATION = 0.00  # 0%
MAX_ALLOCATION = 0.25  # 25%
ALLOCATION_STEPS = 50  # Granularity

# Risk-free rate
RISK_FREE_RATE = 0.04

# Data parameters
LOOKBACK_YEARS = 5
RESAMPLE_FREQ = 'W'

# Benchmark
BENCHMARK_TICKER = 'ACWI'


# =============================================================================
# PORTFOLIO CONFIGURATION (from reconstruction)
# =============================================================================

TOP_10_HOLDINGS = {
    'AAPL': {'weight': 7.0, 'sector': 'Information Technology'},
    'MSFT': {'weight': 6.0, 'sector': 'Information Technology'},
    'NVDA': {'weight': 5.0, 'sector': 'Information Technology'},
    'ASML': {'weight': 4.0, 'sector': 'Information Technology'},
    'SAP': {'weight': 2.5, 'sector': 'Information Technology'},
    'REY.MI': {'weight': 2.0, 'sector': 'Information Technology'},
    'IDR.MC': {'weight': 2.0, 'sector': 'Industrials'},
    'JPM': {'weight': 3.0, 'sector': 'Financials'},
    'GS': {'weight': 2.5, 'sector': 'Financials'},
    'HSBC': {'weight': 2.0, 'sector': 'Financials'},
}

TARGET_SECTOR_WEIGHTS = {
    'Information Technology': 26.5,
    'Financials': 12.5,
    'Industrials': 8.0,
    'Health Care': 9.5,
    'Consumer Discretionary': 5.0,
    'Communication Services': 6.5,
    'Real Estate': 8.7,
    'Consumer Staples': 5.0,
    'Utilities': 2.3,
    'Energy': 3.9,
    'Commodities': 12.1,
}

SECTOR_ETF_PROXIES = {
    'Information Technology': 'IXN',
    'Financials': 'IXG',
    'Health Care': 'IXJ',
    'Industrials': 'EXI',
    'Energy': 'IXC',
    'Commodities': 'MXI',
    'Consumer Staples': 'KXI',
    'Consumer Discretionary': 'RXI',
    'Utilities': 'JXI',
    'Communication Services': 'IXP',
    'Real Estate': 'REET',
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OptimizationResult:
    """Results from allocation optimization."""
    ticker: str
    name: str
    
    # Optimal allocations
    sharpe_optimal_allocation: float
    sharpe_optimal_value: float
    
    risk_budget_optimal_allocation: float
    min_volatility: float
    
    # Combined recommendation
    recommended_allocation: float
    recommendation_method: str
    
    # Original portfolio metrics
    original_sharpe: float
    original_volatility: float
    original_return: float
    
    # At recommended allocation
    new_sharpe: float
    new_volatility: float
    new_return: float
    
    # Improvement
    sharpe_improvement: float
    vol_reduction: float
    
    # Scan data for plotting
    allocation_range: np.ndarray = None
    sharpe_curve: np.ndarray = None
    volatility_curve: np.ndarray = None
    return_curve: np.ndarray = None
    mctr_curve: np.ndarray = None


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def download_data(tickers: list, period_years: int = 5) -> pd.DataFrame:
    """Download adjusted close prices."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    
    print(f"Downloading data for {len(tickers)} tickers...")
    
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers
    
    return prices.ffill().dropna()


def build_portfolio_weights() -> dict:
    """Build complete portfolio weights from Top 10 + ETF proxies."""
    weights = {}
    
    for ticker, info in TOP_10_HOLDINGS.items():
        weights[ticker] = info['weight']
    
    sector_used = {}
    for ticker, info in TOP_10_HOLDINGS.items():
        sector = info['sector']
        sector_used[sector] = sector_used.get(sector, 0) + info['weight']
    
    for sector, target in TARGET_SECTOR_WEIGHTS.items():
        used = sector_used.get(sector, 0)
        remaining = max(0, target - used)
        if remaining > 0:
            etf = SECTOR_ETF_PROXIES.get(sector)
            if etf:
                weights[etf] = weights.get(etf, 0) + remaining
    
    total = sum(weights.values())
    if abs(total - 100) > 0.01:
        weights = {k: v / total * 100 for k, v in weights.items()}
    
    return weights


def compute_portfolio_returns(prices: pd.DataFrame, weights: dict, 
                              resample_freq: str = 'W') -> pd.Series:
    """Compute weighted portfolio returns."""
    if resample_freq == 'W':
        prices_resampled = prices.resample('W-FRI').last()
    else:
        prices_resampled = prices
    
    returns = prices_resampled.pct_change().dropna()
    
    available = [t for t in weights.keys() if t in returns.columns]
    available_weights = {t: weights[t] for t in available}
    total = sum(available_weights.values())
    normalized = {t: w / total * 100 for t, w in available_weights.items()}
    
    portfolio_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in normalized.items():
        portfolio_returns += returns[ticker] * (weight / 100)
    
    return portfolio_returns


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def calculate_metrics(returns: pd.Series, risk_free: float = 0.04, 
                      periods_per_year: int = 52) -> Tuple[float, float, float]:
    """Calculate annualized return, volatility, and Sharpe ratio."""
    n_periods = len(returns)
    years = n_periods / periods_per_year
    
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0
    
    return ann_return, ann_vol, sharpe


def construct_blended_portfolio(original_returns: pd.Series, 
                                 candidate_returns: pd.Series,
                                 allocation: float) -> pd.Series:
    """Blend original portfolio with candidate at given allocation."""
    aligned = pd.concat([original_returns, candidate_returns], axis=1).dropna()
    blended = (1 - allocation) * aligned.iloc[:, 0] + allocation * aligned.iloc[:, 1]
    return blended


def calculate_mctr(original_returns: pd.Series, 
                   candidate_returns: pd.Series,
                   allocation: float,
                   periods_per_year: int = 52) -> float:
    """
    Calculate Marginal Contribution to Risk (MCTR).
    
    MCTR = d(Portfolio_Vol) / d(Candidate_Weight)
    
    If MCTR < 0, adding more of the candidate reduces total risk.
    If MCTR > 0, adding more increases risk (concentration effect).
    """
    delta = 0.001  # 0.1% perturbation
    
    # Volatility at current allocation
    blended_current = construct_blended_portfolio(original_returns, candidate_returns, allocation)
    vol_current = blended_current.std() * np.sqrt(periods_per_year)
    
    # Volatility at slightly higher allocation
    blended_higher = construct_blended_portfolio(original_returns, candidate_returns, allocation + delta)
    vol_higher = blended_higher.std() * np.sqrt(periods_per_year)
    
    # Marginal change
    mctr = (vol_higher - vol_current) / delta
    
    return mctr


def scan_allocations(original_returns: pd.Series,
                     candidate_returns: pd.Series,
                     min_alloc: float = 0.0,
                     max_alloc: float = 0.25,
                     n_steps: int = 50) -> Dict:
    """
    Scan across allocation range and calculate metrics at each point.
    """
    allocations = np.linspace(min_alloc, max_alloc, n_steps)
    
    sharpe_values = []
    vol_values = []
    return_values = []
    mctr_values = []
    
    for alloc in allocations:
        blended = construct_blended_portfolio(original_returns, candidate_returns, alloc)
        ann_ret, ann_vol, sharpe = calculate_metrics(blended)
        mctr = calculate_mctr(original_returns, candidate_returns, alloc)
        
        sharpe_values.append(sharpe)
        vol_values.append(ann_vol * 100)  # Convert to percentage
        return_values.append(ann_ret * 100)
        mctr_values.append(mctr * 100)  # Convert to percentage points
    
    return {
        'allocations': allocations,
        'sharpe': np.array(sharpe_values),
        'volatility': np.array(vol_values),
        'returns': np.array(return_values),
        'mctr': np.array(mctr_values),
    }


def find_sharpe_optimal(scan_data: Dict) -> Tuple[float, float]:
    """Find allocation that maximizes Sharpe ratio."""
    idx = np.argmax(scan_data['sharpe'])
    return scan_data['allocations'][idx], scan_data['sharpe'][idx]


def find_min_volatility(scan_data: Dict) -> Tuple[float, float]:
    """Find allocation that minimizes volatility (Risk Budgeting approach)."""
    idx = np.argmin(scan_data['volatility'])
    return scan_data['allocations'][idx], scan_data['volatility'][idx]


def find_mctr_zero_crossing(scan_data: Dict) -> float:
    """
    Find allocation where MCTR crosses from negative to positive.
    This is the point where adding more starts increasing risk.
    """
    mctr = scan_data['mctr']
    allocations = scan_data['allocations']
    
    # Find where MCTR changes sign from negative to positive
    for i in range(len(mctr) - 1):
        if mctr[i] < 0 and mctr[i + 1] >= 0:
            # Linear interpolation to find exact crossing
            t = -mctr[i] / (mctr[i + 1] - mctr[i])
            return allocations[i] + t * (allocations[i + 1] - allocations[i])
    
    # If always negative (rare), return max allocation
    if all(m < 0 for m in mctr):
        return allocations[-1]
    
    # If always positive, return 0
    return 0.0


# =============================================================================
# MAIN OPTIMIZATION FUNCTION
# =============================================================================

def optimize_allocation(ticker: str, name: str) -> OptimizationResult:
    """
    Find optimal allocation using both Sharpe Maximization and Risk Budgeting.
    
    Returns combined recommendation.
    """
    print("\n" + "="*70)
    print(f"OPTIMAL ALLOCATION FINDER: {name} ({ticker})")
    print("="*70)
    
    # Build original portfolio
    print("\n1. Building original portfolio...")
    weights = build_portfolio_weights()
    
    # Download data
    all_tickers = list(weights.keys()) + [BENCHMARK_TICKER, ticker]
    all_tickers = list(set(all_tickers))
    prices = download_data(all_tickers, LOOKBACK_YEARS)
    
    # Compute original portfolio returns
    original_returns = compute_portfolio_returns(prices, weights, RESAMPLE_FREQ)
    
    # Get candidate returns
    if RESAMPLE_FREQ == 'W':
        prices_resampled = prices.resample('W-FRI').last()
    else:
        prices_resampled = prices
    
    candidate_returns = prices_resampled[ticker].pct_change().dropna()
    
    # Original metrics
    orig_ret, orig_vol, orig_sharpe = calculate_metrics(original_returns)
    print(f"   Original Portfolio: Return={orig_ret*100:.2f}%, Vol={orig_vol*100:.2f}%, Sharpe={orig_sharpe:.3f}")
    
    # Candidate standalone metrics
    cand_ret, cand_vol, cand_sharpe = calculate_metrics(candidate_returns)
    print(f"   {ticker} Standalone:  Return={cand_ret*100:.2f}%, Vol={cand_vol*100:.2f}%, Sharpe={cand_sharpe:.3f}")
    
    # Correlation
    aligned = pd.concat([original_returns, candidate_returns], axis=1).dropna()
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    print(f"   Correlation with Portfolio: {correlation:.3f}")
    
    # Scan allocations
    print(f"\n2. Scanning allocations from {MIN_ALLOCATION*100:.0f}% to {MAX_ALLOCATION*100:.0f}%...")
    scan_data = scan_allocations(
        original_returns, 
        candidate_returns,
        MIN_ALLOCATION,
        MAX_ALLOCATION,
        ALLOCATION_STEPS
    )
    
    # Find optima
    print("\n3. Finding optimal allocations...")
    
    # Method 1: Sharpe Optimization
    sharpe_alloc, sharpe_value = find_sharpe_optimal(scan_data)
    print(f"   📈 SHARPE OPTIMIZATION:")
    print(f"      Best Allocation: {sharpe_alloc*100:.1f}%")
    print(f"      Sharpe Ratio: {sharpe_value:.3f} (vs {orig_sharpe:.3f} original)")
    
    # Method 2: Minimum Volatility (Risk Budgeting)
    minvol_alloc, min_vol = find_min_volatility(scan_data)
    print(f"\n   📊 RISK BUDGETING (Min Volatility):")
    print(f"      Best Allocation: {minvol_alloc*100:.1f}%")
    print(f"      Portfolio Vol: {min_vol:.2f}% (vs {orig_vol*100:.2f}% original)")
    
    # Method 3: MCTR Zero Crossing
    mctr_alloc = find_mctr_zero_crossing(scan_data)
    print(f"\n   ⚖️  MCTR ANALYSIS (Marginal Risk Contribution):")
    print(f"      Risk-Neutral Allocation: {mctr_alloc*100:.1f}%")
    print(f"      (Beyond this, adding more increases risk)")
    
    # Combined recommendation
    print("\n4. Generating recommendation...")
    
    # Priority: If Sharpe improves and allocation is reasonable, use Sharpe
    # Otherwise, use MCTR as the risk-conscious choice
    if sharpe_value > orig_sharpe * 1.01:  # At least 1% Sharpe improvement
        recommended = sharpe_alloc
        method = "Sharpe Optimization"
    else:
        # Use the more conservative of minvol and mctr
        recommended = min(minvol_alloc, mctr_alloc) if mctr_alloc > 0 else minvol_alloc
        method = "Risk Budgeting"
    
    # Calculate metrics at recommended allocation
    blended = construct_blended_portfolio(original_returns, candidate_returns, recommended)
    new_ret, new_vol, new_sharpe = calculate_metrics(blended)
    
    # Create result
    result = OptimizationResult(
        ticker=ticker,
        name=name,
        sharpe_optimal_allocation=sharpe_alloc,
        sharpe_optimal_value=sharpe_value,
        risk_budget_optimal_allocation=minvol_alloc,
        min_volatility=min_vol,
        recommended_allocation=recommended,
        recommendation_method=method,
        original_sharpe=orig_sharpe,
        original_volatility=orig_vol * 100,
        original_return=orig_ret * 100,
        new_sharpe=new_sharpe,
        new_volatility=new_vol * 100,
        new_return=new_ret * 100,
        sharpe_improvement=(new_sharpe - orig_sharpe) / orig_sharpe * 100,
        vol_reduction=(orig_vol - new_vol) / orig_vol * 100 * 100,
        allocation_range=scan_data['allocations'],
        sharpe_curve=scan_data['sharpe'],
        volatility_curve=scan_data['volatility'],
        return_curve=scan_data['returns'],
        mctr_curve=scan_data['mctr'],
    )
    
    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_optimization(result: OptimizationResult, save_path: str = None):
    """Create 4-panel optimization visualization."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    alloc_pct = result.allocation_range * 100
    
    # 1. Sharpe Ratio Curve
    ax1 = axes[0, 0]
    ax1.plot(alloc_pct, result.sharpe_curve, 'b-', linewidth=2, label='Sharpe Ratio')
    ax1.axhline(y=result.original_sharpe, color='gray', linestyle='--', alpha=0.7, label='Original')
    ax1.axvline(x=result.sharpe_optimal_allocation * 100, color='green', linestyle=':', 
                linewidth=2, label=f'Optimal: {result.sharpe_optimal_allocation*100:.1f}%')
    ax1.scatter([result.sharpe_optimal_allocation * 100], [result.sharpe_optimal_value], 
                color='green', s=100, zorder=5)
    ax1.set_xlabel('Allocation (%)')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('📈 Sharpe Ratio Optimization', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Volatility Curve
    ax2 = axes[0, 1]
    ax2.plot(alloc_pct, result.volatility_curve, 'r-', linewidth=2, label='Portfolio Volatility')
    ax2.axhline(y=result.original_volatility, color='gray', linestyle='--', alpha=0.7, label='Original')
    ax2.axvline(x=result.risk_budget_optimal_allocation * 100, color='orange', linestyle=':', 
                linewidth=2, label=f'Min Vol: {result.risk_budget_optimal_allocation*100:.1f}%')
    ax2.scatter([result.risk_budget_optimal_allocation * 100], [result.min_volatility], 
                color='orange', s=100, zorder=5)
    ax2.set_xlabel('Allocation (%)')
    ax2.set_ylabel('Annualized Volatility (%)')
    ax2.set_title('📊 Risk Budgeting (Min Volatility)', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. MCTR Curve
    ax3 = axes[1, 0]
    ax3.plot(alloc_pct, result.mctr_curve, 'purple', linewidth=2, label='MCTR')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.fill_between(alloc_pct, result.mctr_curve, 0, 
                     where=result.mctr_curve < 0, alpha=0.3, color='green', label='Risk-Reducing')
    ax3.fill_between(alloc_pct, result.mctr_curve, 0, 
                     where=result.mctr_curve >= 0, alpha=0.3, color='red', label='Risk-Increasing')
    ax3.set_xlabel('Allocation (%)')
    ax3.set_ylabel('MCTR (% pts per 1% allocation)')
    ax3.set_title('⚖️ Marginal Contribution to Risk (MCTR)', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficient Frontier (Risk vs Return)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(result.volatility_curve, result.return_curve, 
                          c=result.sharpe_curve, cmap='RdYlGn', s=50, alpha=0.8)
    ax4.scatter([result.original_volatility], [result.original_return], 
                color='blue', s=200, marker='*', label='Original Portfolio', zorder=5)
    
    # Mark recommended
    rec_idx = np.argmin(np.abs(result.allocation_range - result.recommended_allocation))
    ax4.scatter([result.volatility_curve[rec_idx]], [result.return_curve[rec_idx]], 
                color='green', s=200, marker='D', label=f'Recommended ({result.recommended_allocation*100:.1f}%)', zorder=5)
    
    ax4.set_xlabel('Annualized Volatility (%)')
    ax4.set_ylabel('Annualized Return (%)')
    ax4.set_title('🎯 Efficient Frontier (Color = Sharpe)', fontweight='bold')
    ax4.legend(loc='best')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Sharpe Ratio')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Optimal Allocation Analysis: {result.name} ({result.ticker})\n'
                 f'Recommended: {result.recommended_allocation*100:.1f}% via {result.recommendation_method}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {save_path}")
    
    plt.show()


def print_recommendation(result: OptimizationResult):
    """Print formatted recommendation."""
    
    print("\n" + "="*70)
    print("🎯 RECOMMENDATION")
    print("="*70)
    
    print(f"\n   Stock: {result.name} ({result.ticker})")
    print(f"\n   ✅ OPTIMAL ALLOCATION: {result.recommended_allocation*100:.1f}%")
    print(f"   Method: {result.recommendation_method}")
    
    print("\n   Impact at Recommended Allocation:")
    print(f"   ┌─────────────────┬─────────────┬─────────────┬──────────────┐")
    print(f"   │ Metric          │ Original    │ New         │ Change       │")
    print(f"   ├─────────────────┼─────────────┼─────────────┼──────────────┤")
    print(f"   │ Return          │ {result.original_return:>10.2f}% │ {result.new_return:>10.2f}% │ {result.new_return - result.original_return:>+10.2f}% │")
    print(f"   │ Volatility      │ {result.original_volatility:>10.2f}% │ {result.new_volatility:>10.2f}% │ {result.new_volatility - result.original_volatility:>+10.2f}% │")
    print(f"   │ Sharpe Ratio    │ {result.original_sharpe:>10.3f}  │ {result.new_sharpe:>10.3f}  │ {result.new_sharpe - result.original_sharpe:>+10.3f}  │")
    print(f"   └─────────────────┴─────────────┴─────────────┴──────────────┘")
    
    print("\n   Summary:")
    if result.sharpe_improvement > 0:
        print(f"   • Sharpe Ratio improved by {result.sharpe_improvement:.1f}%")
    if result.vol_reduction > 0:
        print(f"   • Volatility reduced by {result.vol_reduction:.2f} percentage points")
    
    if result.sharpe_improvement > 5:
        print(f"\n   💡 STRONG ADD: Clear risk-adjusted improvement")
    elif result.sharpe_improvement > 0:
        print(f"\n   💡 GOOD ADD: Moderate improvement in portfolio efficiency")
    elif result.vol_reduction > 0:
        print(f"\n   💡 DIVERSIFICATION PLAY: Lower risk, similar return")
    else:
        print(f"\n   ⚠️  WEAK CASE: Consider alternative candidates")


# =============================================================================
# BATCH OPTIMIZATION
# =============================================================================

def optimize_multiple(candidates: List[Dict]) -> pd.DataFrame:
    """
    Optimize allocation for multiple candidates and return comparison.
    
    candidates = [
        {'ticker': 'UNH', 'name': 'UnitedHealth Group'},
        {'ticker': 'V', 'name': 'Visa Inc.'},
        ...
    ]
    """
    results = []
    
    for candidate in candidates:
        ticker = candidate['ticker']
        name = candidate['name']
        
        try:
            result = optimize_allocation(ticker, name)
            results.append({
                'Ticker': ticker,
                'Name': name,
                'Optimal Allocation': f"{result.recommended_allocation*100:.1f}%",
                'Method': result.recommendation_method,
                'Original Sharpe': result.original_sharpe,
                'New Sharpe': result.new_sharpe,
                'Sharpe Improvement': f"{result.sharpe_improvement:+.1f}%",
                'Vol Change': f"{result.new_volatility - result.original_volatility:+.2f}%",
            })
            
            # Save individual chart
            plot_optimization(result, save_path=f'optimal_allocation_{ticker}.png')
            
        except Exception as e:
            print(f"❌ Failed to optimize {ticker}: {e}")
            results.append({
                'Ticker': ticker,
                'Name': name,
                'Optimal Allocation': 'ERROR',
                'Method': str(e),
            })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point."""
    
    # Single stock optimization
    result = optimize_allocation(CANDIDATE_TICKER, CANDIDATE_NAME)
    
    # Print recommendation
    print_recommendation(result)
    
    # Generate visualization
    plot_optimization(result, save_path=f'optimal_allocation_{CANDIDATE_TICKER}.png')
    
    # Save results
    summary = pd.DataFrame([{
        'Ticker': result.ticker,
        'Name': result.name,
        'Sharpe Optimal (%)': result.sharpe_optimal_allocation * 100,
        'Min Vol Optimal (%)': result.risk_budget_optimal_allocation * 100,
        'Recommended (%)': result.recommended_allocation * 100,
        'Method': result.recommendation_method,
        'Original Sharpe': result.original_sharpe,
        'New Sharpe': result.new_sharpe,
        'Original Vol (%)': result.original_volatility,
        'New Vol (%)': result.new_volatility,
    }])
    
    summary.to_csv(f'optimal_allocation_{CANDIDATE_TICKER}_summary.csv', index=False)
    print(f"\nSummary saved to: optimal_allocation_{CANDIDATE_TICKER}_summary.csv")
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = main()
