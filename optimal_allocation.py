"""
ECE Portfolio Analysis Engine
=============================
Optimal Allocation Finder
=========================
Automatically finds the optimal allocation for a candidate stock using:
1. Mean-Variance Utility Maximization (with Concentration Penalty) - Primary method
2. Sharpe Ratio Optimization - Reference (λ→0 asymptotic case)
3. Minimum Volatility - Reference (λ→∞ asymptotic case)

The utility function is: U = E[R] - (λ/2) × σ² - γ × w²
Where:
  - λ (RISK_AVERSION = 2.0): Penalizes volatility
  - γ (CONCENTRATION_PENALTY = 0.5): Penalizes high allocations (quadratic cost)

This ensures allocations remain realistic (typically 5-15%) rather than hitting 25% caps.

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
from config import AnalysisConfig
from portfolio_reconstruction import build_portfolio_weights, compute_portfolio_returns

# =============================================================================
# DATA CLASSES
# =============================================================================


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
    
    # Utility optimization
    utility_optimal_allocation: float
    utility_optimal_value: float
    risk_aversion: float
    
    # Combined recommendation
    recommended_allocation: float
    recommendation_method: str
    
    # Original portfolio metrics
    original_sharpe: float
    original_volatility: float
    original_return: float
    original_utility: float
    
    # At recommended allocation
    new_sharpe: float
    new_volatility: float
    new_return: float
    new_utility: float
    
    # Improvement
    sharpe_improvement: float
    vol_reduction: float
    utility_improvement: float
    
    # Scan data for plotting
    allocation_range: np.ndarray = None
    sharpe_curve: np.ndarray = None
    volatility_curve: np.ndarray = None
    return_curve: np.ndarray = None
    utility_curve: np.ndarray = None
    mctr_curve: np.ndarray = None


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def download_data(tickers: list, config: AnalysisConfig) -> pd.DataFrame:
    """Download adjusted close prices."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.lookback_years * 365)
    
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


def calculate_utility(returns: pd.Series, allocation: float, config: AnalysisConfig) -> float:
    """
    Calculate Mean-Variance Utility with Concentration Penalty.
    
    U = E[R] - (λ/2) × σ² - γ × w²
    """
    ann_return, ann_vol, _ = calculate_metrics(returns, config.risk_free_rate, 52 if config.resample_freq == 'W' else 252)
    
    # Base utility: return minus risk penalty
    base_utility = ann_return - (config.risk_aversion / 2) * (ann_vol ** 2)
    
    # Concentration penalty: quadratic penalty on allocation weight
    concentration_cost = config.concentration_penalty * (allocation ** 2)
    
    utility = base_utility - concentration_cost
    return utility


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
                     config: AnalysisConfig,
                     n_steps: int = 50) -> Dict:
    """
    Scan across allocation range and calculate metrics at each point.
    """
    allocations = np.linspace(config.min_allocation, config.max_allocation, n_steps)
    
    sharpe_values = []
    vol_values = []
    return_values = []
    utility_values = []
    mctr_values = []
    
    for alloc in allocations:
        blended = construct_blended_portfolio(original_returns, candidate_returns, alloc)
        ann_ret, ann_vol, sharpe = calculate_metrics(blended, config.risk_free_rate, 52 if config.resample_freq == 'W' else 252)
        utility = calculate_utility(blended, alloc, config)
        mctr = calculate_mctr(original_returns, candidate_returns, alloc, 52 if config.resample_freq == 'W' else 252)
        
        sharpe_values.append(sharpe)
        vol_values.append(ann_vol * 100)  # Convert to percentage
        return_values.append(ann_ret * 100)
        utility_values.append(utility * 100)  # Convert to percentage for display
        mctr_values.append(mctr * 100)  # Convert to percentage points
    
    return {
        'allocations': allocations,
        'sharpe': np.array(sharpe_values),
        'volatility': np.array(vol_values),
        'returns': np.array(return_values),
        'utility': np.array(utility_values),
        'mctr': np.array(mctr_values),
    }


def find_sharpe_optimal(scan_data: Dict) -> Tuple[float, float, bool]:
    """
    Find allocation that maximizes Sharpe ratio.
    
    Returns:
        - optimal allocation
        - Sharpe ratio at that allocation
        - is_at_boundary: True if optimal is at max of scan range (suggests true optimal may be higher)
    """
    idx = np.argmax(scan_data['sharpe'])
    is_at_boundary = (idx == len(scan_data['allocations']) - 1)
    return scan_data['allocations'][idx], scan_data['sharpe'][idx], is_at_boundary


def find_min_volatility(scan_data: Dict) -> Tuple[float, float]:
    """Find allocation that minimizes volatility (Risk Budgeting approach)."""
    idx = np.argmin(scan_data['volatility'])
    return scan_data['allocations'][idx], scan_data['volatility'][idx]


def find_utility_optimal(scan_data: Dict) -> Tuple[float, float]:
    """Find allocation that maximizes Mean-Variance Utility."""
    idx = np.argmax(scan_data['utility'])
    return scan_data['allocations'][idx], scan_data['utility'][idx]


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

def optimize_allocation(ticker: str, name: str, config: AnalysisConfig, 
                        top_holdings: Dict, sector_targets: Dict) -> OptimizationResult:
    """
    Find optimal allocation using both Sharpe Maximization and Risk Budgeting.
    
    Returns combined recommendation.
    """
    print("\n" + "="*70)
    print(f"OPTIMAL ALLOCATION FINDER: {name} ({ticker})")
    print("="*70)
    
    # Build original portfolio
    print("\n1. Building original portfolio...")
    weights = build_portfolio_weights(top_holdings, sector_targets)
    
    # Download data
    all_tickers = list(weights.keys()) + [config.benchmark_ticker, ticker]
    all_tickers = list(set(all_tickers))
    prices = download_data(all_tickers, config)
    
    # Compute original portfolio returns
    original_returns, _ = compute_portfolio_returns(prices, weights, config.resample_freq)
    
    # Get candidate returns
    if config.resample_freq == 'W':
        prices_resampled = prices.resample('W-FRI').last()
    else:
        prices_resampled = prices
    
    candidate_returns = prices_resampled[ticker].pct_change().dropna()
    
    # Original metrics
    periods = 52 if config.resample_freq == 'W' else 252
    orig_ret, orig_vol, orig_sharpe = calculate_metrics(original_returns, config.risk_free_rate, periods)
    print(f"   Original Portfolio: Return={orig_ret*100:.2f}%, Vol={orig_vol*100:.2f}%, Sharpe={orig_sharpe:.3f}")
    
    # Candidate standalone metrics
    cand_ret, cand_vol, cand_sharpe = calculate_metrics(candidate_returns, config.risk_free_rate, periods)
    print(f"   {ticker} Standalone:  Return={cand_ret*100:.2f}%, Vol={cand_vol*100:.2f}%, Sharpe={cand_sharpe:.3f}")
    
    # Correlation
    aligned = pd.concat([original_returns, candidate_returns], axis=1).dropna()
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    print(f"   Correlation with Portfolio: {correlation:.3f}")
    
    # Scan allocations
    print(f"\n2. Scanning allocations from {config.min_allocation*100:.0f}% to {config.max_allocation*100:.0f}%...")
    scan_data = scan_allocations(
        original_returns, 
        candidate_returns,
        config,
        n_steps=50
    )
    
    # Find optima
    print("\n3. Finding optimal allocations...")
    
    # Method 1: Sharpe Optimization (for reference)
    sharpe_alloc, sharpe_value, sharpe_at_boundary = find_sharpe_optimal(scan_data)
    print(f"   📈 SHARPE OPTIMIZATION (reference):")
    print(f"      Best Allocation: {sharpe_alloc*100:.1f}%")
    print(f"      Sharpe Ratio: {sharpe_value:.3f} (vs {orig_sharpe:.3f} original)")
    if sharpe_at_boundary:
        print(f"      ⚠️  WARNING: Optimal is at {config.max_allocation*100:.0f}% cap - true optimal may be higher")
    
    # Method 2: Minimum Volatility (for reference)
    minvol_alloc, min_vol = find_min_volatility(scan_data)
    print(f"\n   📊 MIN VOLATILITY (reference):")
    print(f"      Best Allocation: {minvol_alloc*100:.1f}%")
    print(f"      Portfolio Vol: {min_vol:.2f}% (vs {orig_vol*100:.2f}% original)")
    
    # Method 3: Mean-Variance Utility (PRIMARY METHOD)
    utility_alloc, utility_value = find_utility_optimal(scan_data)
    orig_utility = calculate_utility(original_returns, 0.0, config)
    print(f"\n   🎯 MEAN-VARIANCE UTILITY (λ={config.risk_aversion}, γ={config.concentration_penalty}):")
    print(f"      Best Allocation: {utility_alloc*100:.1f}%")
    print(f"      Utility: {utility_value:.2f}% (vs {orig_utility*100:.2f}% original)")
    print(f"      (γ penalizes high allocations → targets ~10%)")
    
    # Combined recommendation: Use Utility Maximization as primary method
    print("\n4. Generating recommendation...")
    
    # Apply minimum floor for positions (institutional constraint)
    # Apply floor to ALL allocations, even if optimal is 0%
    if utility_alloc < config.min_recommended_allocation:
        recommended = config.min_recommended_allocation
        method = f"Mean-Variance Utility (λ={config.risk_aversion}, γ={config.concentration_penalty}) [floor applied]"
    else:
        recommended = utility_alloc
        method = f"Mean-Variance Utility (λ={config.risk_aversion}, γ={config.concentration_penalty})"
    
    # Calculate metrics at recommended allocation
    blended = construct_blended_portfolio(original_returns, candidate_returns, recommended)
    new_ret, new_vol, new_sharpe = calculate_metrics(blended, config.risk_free_rate, periods)
    
    # Calculate new utility at recommended allocation
    new_utility = calculate_utility(blended, recommended, config)
    
    # Create result
    result = OptimizationResult(
        ticker=ticker,
        name=name,
        sharpe_optimal_allocation=sharpe_alloc,
        sharpe_optimal_value=sharpe_value,
        risk_budget_optimal_allocation=minvol_alloc,
        min_volatility=min_vol,
        utility_optimal_allocation=utility_alloc,
        utility_optimal_value=utility_value,
        risk_aversion=config.risk_aversion,
        recommended_allocation=recommended,
        recommendation_method=method,
        original_sharpe=orig_sharpe,
        original_volatility=orig_vol * 100,
        original_return=orig_ret * 100,
        original_utility=orig_utility * 100,
        new_sharpe=new_sharpe,
        new_volatility=new_vol * 100,
        new_return=new_ret * 100,
        new_utility=new_utility * 100,
        sharpe_improvement=(new_sharpe - orig_sharpe) / orig_sharpe * 100,
        vol_reduction=(orig_vol - new_vol) / orig_vol * 100 * 100,
        utility_improvement=(new_utility - orig_utility) / abs(orig_utility) * 100 if orig_utility != 0 else 0,
        allocation_range=scan_data['allocations'],
        sharpe_curve=scan_data['sharpe'],
        volatility_curve=scan_data['volatility'],
        return_curve=scan_data['returns'],
        utility_curve=scan_data['utility'],
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
    
    # 3. Utility Curve (Mean-Variance)
    ax3 = axes[1, 0]
    ax3.plot(alloc_pct, result.utility_curve, 'purple', linewidth=2, label=f'Utility (λ={result.risk_aversion})')
    ax3.axhline(y=result.original_utility, color='gray', linestyle='--', alpha=0.7, label='Original')
    ax3.axvline(x=result.utility_optimal_allocation * 100, color='green', linestyle=':', 
                linewidth=2, label=f'Optimal: {result.utility_optimal_allocation*100:.1f}%')
    ax3.scatter([result.utility_optimal_allocation * 100], [result.utility_optimal_value], 
                color='green', s=100, zorder=5)
    ax3.set_xlabel('Allocation (%)')
    ax3.set_ylabel('Utility (%)')
    ax3.set_title(f'🎯 Mean-Variance Utility (λ={result.risk_aversion})', fontweight='bold')
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
    from portfolio_loader import DEFAULT_TOP_HOLDINGS, DEFAULT_SECTOR_TARGETS
    
    config = AnalysisConfig()
    
    # Single stock optimization (default test case)
    # Using 'UNH' as default if running standalone
    ticker = 'UNH'
    name = 'UnitedHealth Group'
    
    result = optimize_allocation(ticker, name, config, DEFAULT_TOP_HOLDINGS, DEFAULT_SECTOR_TARGETS)
    
    # Print recommendation
    print_recommendation(result)
    
    # Generate visualization
    plot_optimization(result, save_path=f'optimal_allocation_{ticker}.png')
    
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
    
    summary.to_csv(f'optimal_allocation_{ticker}_summary.csv', index=False)
    print(f"\nSummary saved to: optimal_allocation_{ticker}_summary.csv")
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    
    return result


if __name__ == "__main__":
    result = main()
