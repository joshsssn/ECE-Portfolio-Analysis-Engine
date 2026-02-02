"""
ECE Portfolio Analysis Engine
=============================
Backtest Candidate Stock Analysis
==================================
Analyze the impact of adding a candidate stock to the reconstructed portfolio.
Compares risk/return metrics before and after the addition.

Author: Josh E. SOUSSAN
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - MODIFY THESE FOR YOUR ANALYSIS
# =============================================================================
from config import AnalysisConfig
from portfolio_reconstruction import (
    build_portfolio_weights, 
    compute_portfolio_returns, 
    calculate_risk_metrics as calculate_metrics, # Rename to match local usage
    download_data, 
    clean_data
)

# =============================================================================
# DATA FUNCTIONS
# =============================================================================


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

# =============================================================================
# PRO-FORMA PORTFOLIO CONSTRUCTION
# =============================================================================

def construct_new_portfolio(old_portfolio: pd.Series, candidate_returns: pd.Series,
                           allocation_size: float) -> pd.Series:
    """
    Construct new portfolio by adding candidate stock.
    New = (1 - allocation) * Old + allocation * Candidate
    """
    # Align dates
    aligned = pd.concat([old_portfolio, candidate_returns], axis=1).dropna()
    old = aligned.iloc[:, 0]
    candidate = aligned.iloc[:, 1]
    
    # Pro-forma construction
    new_portfolio = (1 - allocation_size) * old + allocation_size * candidate
    new_portfolio.name = 'New_Portfolio'
    
    return new_portfolio


# =============================================================================
# PRO-FORMA PORTFOLIO CONSTRUCTION
# =============================================================================

def construct_new_portfolio(old_portfolio: pd.Series, candidate_returns: pd.Series,
                           allocation_size: float) -> pd.Series:
    """
    Construct new portfolio by adding candidate stock.
    New = (1 - allocation) * Old + allocation * Candidate
    """
    # Align dates
    aligned = pd.concat([old_portfolio, candidate_returns], axis=1).dropna()
    old = aligned.iloc[:, 0]
    candidate = aligned.iloc[:, 1]
    
    # Pro-forma construction
    new_portfolio = (1 - allocation_size) * old + allocation_size * candidate
    new_portfolio.name = 'New_Portfolio'
    
    return new_portfolio


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def calculate_correlations(candidate_returns: pd.Series, old_portfolio: pd.Series,
                          tech_returns: pd.Series, benchmark_returns: pd.Series,
                          tech_label: str = 'Tech Sector') -> dict:
    """Calculate correlation of candidate with key series."""
    # Align all series
    all_series = pd.concat([candidate_returns, old_portfolio, tech_returns, benchmark_returns], axis=1).dropna()
    all_series.columns = ['Candidate', 'Original Portfolio', f'{tech_label}', 'Benchmark (ACWI)']
    
    correlations = {
        'vs Original Portfolio': all_series['Candidate'].corr(all_series['Original Portfolio']),
        f'vs {tech_label}': all_series['Candidate'].corr(all_series[f'{tech_label}']),
        'vs Benchmark (ACWI)': all_series['Candidate'].corr(all_series['Benchmark (ACWI)']),
    }
    
    return correlations, all_series


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_analysis(old_portfolio: pd.Series, new_portfolio: pd.Series,
                  candidate_returns: pd.Series, benchmark_returns: pd.Series,
                  correlation_matrix: pd.DataFrame, candidate_name: str,
                  allocation_pct: float, save_path: str = None,
                  benchmark_name: str = 'ACWI', rolling_window: int = 52):
    """Create comprehensive analysis visualization."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    
    # Align all data
    aligned = pd.concat([old_portfolio, new_portfolio, benchmark_returns, candidate_returns], axis=1).dropna()
    benchmark_col = f'{benchmark_name} Benchmark'
    aligned.columns = ['Original Portfolio', 'New Portfolio', benchmark_col, f'{candidate_name}']
    
    # 1. Cumulative Returns
    ax1 = fig.add_subplot(2, 2, 1)
    cumulative = (1 + aligned[['Original Portfolio', 'New Portfolio', benchmark_col]]).cumprod()
    cumulative.plot(ax=ax1, linewidth=2)
    ax1.set_title(f'Cumulative Returns: Before vs After Adding {candidate_name} ({allocation_pct:.0%})', 
                  fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Annotate final values
    for col in cumulative.columns:
        final = cumulative[col].iloc[-1]
        ax1.annotate(f'{final:.2f}', xy=(cumulative.index[-1], final),
                     xytext=(5, 0), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # 2. Rolling Beta
    ax2 = fig.add_subplot(2, 2, 2)
    
    for name, series in [('Original Portfolio', aligned['Original Portfolio']), 
                         ('New Portfolio', aligned['New Portfolio'])]:
        rolling_beta = []
        dates = []
        for i in range(rolling_window, len(series)):
            port_slice = series.iloc[i-rolling_window:i]
            bench_slice = aligned[benchmark_col].iloc[i-rolling_window:i]
            cov = port_slice.cov(bench_slice)
            var = bench_slice.var()
            beta = cov / var if var != 0 else 1
            rolling_beta.append(beta)
            dates.append(series.index[i])
        ax2.plot(dates, rolling_beta, label=name, linewidth=2)
    
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Beta = 1')
    ax2.set_title(f'Rolling {rolling_window}-Week Beta vs {benchmark_name}', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Beta')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation Heatmap
    ax3 = fig.add_subplot(2, 2, 3)
    corr_matrix = correlation_matrix.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax3, mask=mask, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax3.set_title(f'Correlation Matrix: {candidate_name} vs Portfolio Components', fontweight='bold')
    
    # 4. Drawdown Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    for col in ['Original Portfolio', 'New Portfolio']:
        cum = (1 + aligned[col]).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max * 100
        ax4.fill_between(dd.index, dd, 0, alpha=0.4, label=col)
        ax4.plot(dd.index, dd, linewidth=1)
    ax4.set_title('Drawdown Comparison', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Drawdown (%)')
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Candidate Stock Analysis: {candidate_name} ({allocation_pct:.0%} Allocation)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {save_path}")
    
    plt.show()


def print_impact_summary(metrics_before: dict, metrics_after: dict, 
                         candidate_metrics: dict, correlations: dict,
                         candidate_name: str, allocation_pct: float):
    """Print formatted impact summary with complete portfolio metrics."""
    
    print("\n" + "="*80)
    print(f"IMPACT ANALYSIS: Adding {candidate_name} at {allocation_pct:.0%} Allocation")
    print("="*80)
    
    # Key Changes
    print("\n📊 KEY CHANGES:")
    print("-"*40)
    
    vol_change = metrics_after['Annualized Volatility (%)'] - metrics_before['Annualized Volatility (%)']
    ret_change = metrics_after['Annualized Return (%)'] - metrics_before['Annualized Return (%)']
    sharpe_change = metrics_after['Sharpe Ratio'] - metrics_before['Sharpe Ratio']
    # Use dynamic key for Beta and Correlation
    beta_key = [k for k in metrics_after.keys() if k.startswith('Beta vs')][0]
    beta_change = metrics_after[beta_key] - metrics_before[beta_key]
    dd_change = metrics_after['Max Drawdown (%)'] - metrics_before['Max Drawdown (%)']
    var_period_change = metrics_after['VaR (95%, period)'] - metrics_before['VaR (95%, period)']
    var_annual_change = metrics_after['VaR (95%, annualized)'] - metrics_before['VaR (95%, annualized)']
    
    print(f"  • Return:          {ret_change:+.2f}% ({'↑' if ret_change > 0 else '↓'})")
    print(f"  • Volatility:      {vol_change:+.2f}% ({'↓ Better' if vol_change < 0 else '↑ Higher Risk'})")
    print(f"  • Sharpe:          {sharpe_change:+.3f} ({'↑ Better' if sharpe_change > 0 else '↓ Worse'})")
    print(f"  • {beta_key.replace('Beta', 'Beta Change')}: {beta_change:+.3f} ({'↓ More Defensive' if beta_change < 0 else '↑ More Aggressive'})")
    print(f"  • Max DD:          {dd_change:+.2f}% ({'↑ Worse' if dd_change > 0 else '↓ Better'})")
    print(f"  • VaR (95% ann.):  {var_annual_change:+.2f}% ({'↑ Better' if var_annual_change > 0 else '↓ Worse'})")
    
    # New Portfolio Complete Metrics
    print("\n📈 NEW PORTFOLIO COMPLETE METRICS:")
    print("-"*40)
    print(f"  • Annualized Return:       {metrics_after['Annualized Return (%)']:.2f}%")
    print(f"  • Annualized Volatility:   {metrics_after['Annualized Volatility (%)']:.2f}%")
    print(f"  • Sharpe Ratio:            {metrics_after['Sharpe Ratio']:.3f}")
    print(f"  • {beta_key}:            {metrics_after[beta_key]:.3f}")
    print(f"  • Alpha:                   {metrics_after['Alpha (%)']:.2f}%")
    print(f"  • VaR (95%, weekly):       {metrics_after['VaR (95%, period)']:.2f}%")
    print(f"  • VaR (95%, annualized):   {metrics_after['VaR (95%, annualized)']:.2f}%")
    print(f"  • Max Drawdown:            {metrics_after['Max Drawdown (%)']:.2f}%")
    
    corr_key = [k for k in metrics_after.keys() if k.startswith('Correlation vs')][0]
    print(f"  • {corr_key}:     {metrics_after[corr_key]:.4f}")
    print(f"  • Tracking Error:          {metrics_after['Tracking Error (%)']:.2f}%")
    print(f"  • Information Ratio:       {metrics_after['Information Ratio']:.3f}")
    
    # Correlation Insights
    print("\n🔗 DIVERSIFICATION ANALYSIS:")
    print("-"*40)
    for key, val in correlations.items():
        insight = "High correlation" if abs(val) > 0.7 else "Moderate diversification" if abs(val) > 0.4 else "Good diversifier"
        print(f"  • {key}: {val:.3f} ({insight})")
    
    # Investment Thesis
    print("\n💡 INVESTMENT THESIS:")
    print("-"*40)
    
    if sharpe_change > 0 and vol_change < 0:
        print(f"  ✅ STRONG ADD: {candidate_name} improves risk-adjusted returns")
        print(f"     → Higher Sharpe (+{sharpe_change:.3f}) with lower volatility ({vol_change:.2f}%)")
    elif sharpe_change > 0:
        print(f"  ⚠️  CONDITIONAL ADD: {candidate_name} improves Sharpe but adds volatility")
        print(f"     → Accept {vol_change:+.2f}% vol for +{sharpe_change:.3f} Sharpe improvement")
    elif correlations['vs Original Portfolio'] < 0.5:
        print(f"  ⚠️  DIVERSIFICATION PLAY: Low correlation ({correlations['vs Original Portfolio']:.2f})")
        print(f"     → Adds diversification despite slight Sharpe reduction")
    else:
        print(f"  ❌ WEAK CASE: {candidate_name} offers limited benefit")
        print(f"     → Consider alternative candidates")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_backtest(ticker: str, name: str, config: AnalysisConfig,
                 top_holdings: dict, sector_targets: dict,
                 allocation: float = 0.05, 
                 output_dir: str = '.', show_plot: bool = True) -> dict:
    """
    Run backtest for a candidate stock. Callable from external scripts.
    """
    import os
    
    # Build original portfolio
    weights = build_portfolio_weights(top_holdings, sector_targets)
    
    # Download data
    all_tickers = list(weights.keys()) + [config.benchmark_ticker, ticker, config.tech_etf_ticker]
    all_tickers = list(set(all_tickers))
    prices = download_data(all_tickers, config)
    prices = clean_data(prices) # Ensure clean
    
    # Compute returns
    old_portfolio, _ = compute_portfolio_returns(prices, weights, config.resample_freq)
    old_portfolio.name = 'Original_Portfolio'
    
    if config.resample_freq == 'W':
        prices_resampled = prices.resample('W-FRI').last()
    else:
        prices_resampled = prices
    
    returns = prices_resampled.pct_change().dropna()
    benchmark_returns = returns[config.benchmark_ticker]
    candidate_returns = returns[ticker]
    tech_returns = returns[config.tech_etf_ticker]
    
    # New portfolio
    new_portfolio = construct_new_portfolio(old_portfolio, candidate_returns, allocation)
    
    # Metrics
    periods_per_year = 52 if config.resample_freq == 'W' else 252
    metrics_before = calculate_metrics(
        old_portfolio, benchmark_returns, config.risk_free_rate, periods_per_year,
        benchmark_name=config.benchmark_ticker
    )
    # Ensure self-correlation for Original Portfolio is 1
    metrics_before['Correlation vs Original Portfolio'] = 1.0
    metrics_after = calculate_metrics(
        new_portfolio, benchmark_returns, config.risk_free_rate, periods_per_year,
        benchmark_name=config.benchmark_ticker
    )
    metrics_candidate = calculate_metrics(
        candidate_returns, benchmark_returns, config.risk_free_rate, periods_per_year,
        benchmark_name=config.benchmark_ticker
    )
    
    # Correlations
    correlations, corr_data = calculate_correlations(
        candidate_returns, old_portfolio, tech_returns, benchmark_returns, config.tech_etf_ticker
    )
    
    # Calculate portfolio correlations
    # Original Portfolio vs Tech ETF
    aligned_orig_tech = pd.concat([old_portfolio, tech_returns], axis=1).dropna()
    metrics_before[f'Correlation vs {config.tech_etf_ticker}'] = aligned_orig_tech.iloc[:, 0].corr(aligned_orig_tech.iloc[:, 1])
    
    # New Portfolio vs Original Portfolio
    aligned_new_orig = pd.concat([new_portfolio, old_portfolio], axis=1).dropna()
    metrics_after['Correlation vs Original Portfolio'] = aligned_new_orig.iloc[:, 0].corr(aligned_new_orig.iloc[:, 1])
    
    # New Portfolio vs Tech ETF
    aligned_new_tech = pd.concat([new_portfolio, tech_returns], axis=1).dropna()
    metrics_after[f'Correlation vs {config.tech_etf_ticker}'] = aligned_new_tech.iloc[:, 0].corr(aligned_new_tech.iloc[:, 1])
    
    # Add correlation metrics to candidate standalone (excluding vs Benchmark)
    metrics_candidate_with_corr = metrics_candidate.copy()
    metrics_candidate_with_corr['Correlation vs Original Portfolio'] = correlations['vs Original Portfolio']
    metrics_candidate_with_corr[f'Correlation vs {config.tech_etf_ticker}'] = correlations[f'vs {config.tech_etf_ticker}']
    
    # Save comparison
    comparison_df = pd.DataFrame({
        'Original Portfolio': metrics_before,
        'New Portfolio': metrics_after,
        f'{name} (Standalone)': metrics_candidate_with_corr,
    }).T
    
    csv_path = os.path.join(output_dir, 'backtest.csv')
    comparison_df.to_csv(csv_path)
    
    # Plot
    chart_path = os.path.join(output_dir, 'backtest.png')
    
    # Determine rolling window (same logic as portfolio_reconstruction)
    rolling_window = 52 if config.resample_freq == 'W' else 63
    
    if show_plot:
        plot_analysis(old_portfolio, new_portfolio, candidate_returns, benchmark_returns,
                      corr_data, name, allocation, save_path=chart_path,
                      benchmark_name=config.benchmark_ticker, rolling_window=rolling_window)
    else:
        # Save without showing
        import matplotlib
        matplotlib.use('Agg')
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(16, 12))
        
        # Recreate plots (simplified for headless)
        aligned = pd.concat([old_portfolio, new_portfolio, benchmark_returns, candidate_returns], axis=1).dropna()
        benchmark_col = f'{config.benchmark_ticker} Benchmark'
        aligned.columns = ['Original', 'New', benchmark_col, 'Candidate']
        
        ax1 = fig.add_subplot(2, 2, 1)
        cumulative = (1 + aligned[['Original', 'New', benchmark_col]]).cumprod()
        cumulative.plot(ax=ax1)
        ax1.set_title(f'Cumulative Returns: +{name}')
        
        ax2 = fig.add_subplot(2, 2, 2)
        corr_matrix = aligned.corr()
        sns.heatmap(corr_matrix, annot=True, ax=ax2, cmap='RdYlGn')
        ax2.set_title('Correlation Matrix')
        
        ax3 = fig.add_subplot(2, 2, 3)
        for col in ['Original', 'New']:
            cum = (1 + aligned[col]).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax() * 100
            ax3.fill_between(dd.index, dd, 0, alpha=0.4, label=col)
        ax3.set_title('Drawdowns')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Return results
    # Find dynamic Beta key
    beta_key = [k for k in metrics_after.keys() if k.startswith('Beta vs')][0]
    
    return {
        'ticker': ticker,
        'name': name,
        'allocation': allocation,
        'return_change': f"{metrics_after['Annualized Return (%)'] - metrics_before['Annualized Return (%)']:+.2f}%",
        'vol_change': f"{metrics_after['Annualized Volatility (%)'] - metrics_before['Annualized Volatility (%)']:+.2f}%",
        'sharpe_change': f"{metrics_after['Sharpe Ratio'] - metrics_before['Sharpe Ratio']:+.3f}",
        'beta_change': f"{metrics_after[beta_key] - metrics_before[beta_key]:+.3f}",
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'correlations': correlations,
        'csv_path': csv_path,
        'chart_path': chart_path,
    }


def main():
    """Main execution function."""
    from portfolio_loader import DEFAULT_TOP_HOLDINGS, DEFAULT_SECTOR_TARGETS
    
    config = AnalysisConfig()
    
    # Default candidate for standalone run
    candidate_ticker = 'UNH'
    candidate_name = 'UnitedHealth Group'
    allocation_size = 0.05
    
    print("="*80)
    print(f"CANDIDATE STOCK ANALYSIS: {candidate_name} ({candidate_ticker})")
    print(f"Target Allocation: {allocation_size:.0%}")
    print("="*80)
    
    # Call run_backtest with defaults
    run_backtest(
        ticker=candidate_ticker,
        name=candidate_name,
        config=config,
        top_holdings=DEFAULT_TOP_HOLDINGS,
        sector_targets=DEFAULT_SECTOR_TARGETS,
        allocation=allocation_size,
        output_dir='.',
        show_plot=True
    )

if __name__ == "__main__":
    main()

