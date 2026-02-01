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

# Candidate Stock to Analyze
CANDIDATE_TICKER = 'UNH'  # Change this to test different stocks 
CANDIDATE_NAME = 'UnitedHealth Group'  # Display name for reports

# Allocation Size (as decimal)
ALLOCATION_SIZE = 0.05  # 5% allocation to candidate

# Benchmark
BENCHMARK_TICKER = 'ACWI'

# Tech Sector ETF (for correlation analysis)
TECH_ETF_TICKER = 'IXN'

# Risk-free rate (annual)
RISK_FREE_RATE = 0.04

# Data parameters
LOOKBACK_YEARS = 5
RESAMPLE_FREQ = 'W'  # Weekly


# =============================================================================
# PORTFOLIO RECONSTRUCTION (from previous script)
# =============================================================================

# Top 10 Holdings
TOP_10_HOLDINGS = {
    'AAPL': {'name': 'APPLE INC.', 'weight': 7.0, 'sector': 'Information Technology', 'country': 'United States'},
    'MSFT': {'name': 'MICROSOFT CORP.', 'weight': 6.0, 'sector': 'Information Technology', 'country': 'United States'},
    'NVDA': {'name': 'NVIDIA CORP.', 'weight': 5.0, 'sector': 'Information Technology', 'country': 'United States'},
    'ASML': {'name': 'ASML HOLDING NV', 'weight': 4.0, 'sector': 'Information Technology', 'country': 'Netherlands'},
    'SAP': {'name': 'SAP SE', 'weight': 2.5, 'sector': 'Information Technology', 'country': 'Germany'},
    'REY.MI': {'name': 'REPLY SPA', 'weight': 2.0, 'sector': 'Information Technology', 'country': 'Italy'},
    'IDR.MC': {'name': 'INDRA SISTEMAS SA', 'weight': 2.0, 'sector': 'Industrials', 'country': 'Spain'},
    'JPM': {'name': 'JPMORGAN CHASE & CO.', 'weight': 3.0, 'sector': 'Financials', 'country': 'United States'},
    'GS': {'name': 'GOLDMAN SACHS GROUP INC.', 'weight': 2.5, 'sector': 'Financials', 'country': 'United States'},
    'HSBC': {'name': 'HSBC HOLDINGS PLC', 'weight': 2.0, 'sector': 'Financials', 'country': 'United Kingdom'},
}

# Target Sector Weights
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

# Sector ETF Proxies
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
# DATA FUNCTIONS
# =============================================================================

def download_data(tickers: list, period_years: int = 5) -> pd.DataFrame:
    """Download adjusted close prices for given tickers."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    
    print(f"Downloading data for {len(tickers)} tickers...")
    
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True
    )
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers
    
    # Clean data
    prices = prices.ffill().dropna()
    
    return prices


def build_portfolio_weights() -> dict:
    """Build complete portfolio weights from Top 10 + ETF proxies."""
    weights = {}
    
    # Add Top 10
    for ticker, info in TOP_10_HOLDINGS.items():
        weights[ticker] = info['weight']
    
    # Calculate remaining sector weights
    sector_used = {}
    for ticker, info in TOP_10_HOLDINGS.items():
        sector = info['sector']
        sector_used[sector] = sector_used.get(sector, 0) + info['weight']
    
    # Add ETF proxies for remaining weights
    for sector, target in TARGET_SECTOR_WEIGHTS.items():
        used = sector_used.get(sector, 0)
        remaining = max(0, target - used)
        if remaining > 0:
            etf = SECTOR_ETF_PROXIES.get(sector)
            if etf:
                weights[etf] = weights.get(etf, 0) + remaining
    
    # Normalize
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
    
    # Filter available tickers
    available = [t for t in weights.keys() if t in returns.columns]
    available_weights = {t: weights[t] for t in available}
    total = sum(available_weights.values())
    normalized = {t: w / total * 100 for t, w in available_weights.items()}
    
    # Compute weighted returns
    portfolio_returns = pd.Series(0, index=returns.index)
    for ticker, weight in normalized.items():
        portfolio_returns += returns[ticker] * (weight / 100)
    
    return portfolio_returns, returns


# =============================================================================
# RISK METRICS CALCULATION
# =============================================================================

def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series,
                     risk_free_rate: float = 0.04, periods_per_year: int = 52) -> dict:
    """Calculate comprehensive risk metrics."""
    metrics = {}
    
    # Align series
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    ret = aligned.iloc[:, 0]
    bench = aligned.iloc[:, 1]
    
    n_periods = len(ret)
    years = n_periods / periods_per_year
    
    # Annualized Return
    total_return = (1 + ret).prod() - 1
    ann_return = (1 + total_return) ** (1 / years) - 1
    metrics['Annualized Return (%)'] = ann_return * 100
    
    # Annualized Volatility
    volatility = ret.std() * np.sqrt(periods_per_year)
    metrics['Annualized Volatility (%)'] = volatility * 100
    
    # Sharpe Ratio
    excess = ann_return - risk_free_rate
    metrics['Sharpe Ratio'] = excess / volatility if volatility != 0 else 0
    
    # Beta
    cov = ret.cov(bench)
    bench_var = bench.var()
    beta = cov / bench_var if bench_var != 0 else 1
    metrics['Beta vs ACWI'] = beta
    
    # Maximum Drawdown
    cumulative = (1 + ret).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics['Max Drawdown (%)'] = drawdown.min() * 100
    
    # Alpha (Jensen's)
    bench_total = (1 + bench).prod() - 1
    bench_ann = (1 + bench_total) ** (1 / years) - 1
    alpha = ann_return - (risk_free_rate + beta * (bench_ann - risk_free_rate))
    metrics['Alpha (%)'] = alpha * 100
    
    # Information Ratio
    tracking_diff = ret - bench
    tracking_error = tracking_diff.std() * np.sqrt(periods_per_year)
    excess_return = ann_return - bench_ann
    metrics['Information Ratio'] = excess_return / tracking_error if tracking_error != 0 else 0
    
    # Correlation with benchmark
    metrics['Correlation vs ACWI'] = ret.corr(bench)
    
    return metrics


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

def calculate_correlations(candidate_returns: pd.Series, old_portfolio: pd.Series,
                          tech_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """Calculate correlation of candidate with key series."""
    # Align all series
    all_series = pd.concat([candidate_returns, old_portfolio, tech_returns, benchmark_returns], axis=1).dropna()
    all_series.columns = ['Candidate', 'Original Portfolio', 'Tech (IXN)', 'Benchmark (ACWI)']
    
    correlations = {
        'vs Original Portfolio': all_series['Candidate'].corr(all_series['Original Portfolio']),
        'vs Tech Sector (IXN)': all_series['Candidate'].corr(all_series['Tech (IXN)']),
        'vs Benchmark (ACWI)': all_series['Candidate'].corr(all_series['Benchmark (ACWI)']),
    }
    
    return correlations, all_series


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_analysis(old_portfolio: pd.Series, new_portfolio: pd.Series,
                  candidate_returns: pd.Series, benchmark_returns: pd.Series,
                  correlation_matrix: pd.DataFrame, candidate_name: str,
                  allocation_pct: float, save_path: str = None):
    """Create comprehensive analysis visualization."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 12))
    
    # Align all data
    aligned = pd.concat([old_portfolio, new_portfolio, benchmark_returns, candidate_returns], axis=1).dropna()
    aligned.columns = ['Original Portfolio', 'New Portfolio', 'ACWI Benchmark', f'{candidate_name}']
    
    # 1. Cumulative Returns
    ax1 = fig.add_subplot(2, 2, 1)
    cumulative = (1 + aligned[['Original Portfolio', 'New Portfolio', 'ACWI Benchmark']]).cumprod()
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
    
    # 2. Rolling 52-Week Beta
    ax2 = fig.add_subplot(2, 2, 2)
    window = 52  # 1 year for weekly data
    
    for name, series in [('Original Portfolio', aligned['Original Portfolio']), 
                         ('New Portfolio', aligned['New Portfolio'])]:
        rolling_beta = []
        dates = []
        for i in range(window, len(series)):
            port_slice = series.iloc[i-window:i]
            bench_slice = aligned['ACWI Benchmark'].iloc[i-window:i]
            cov = port_slice.cov(bench_slice)
            var = bench_slice.var()
            beta = cov / var if var != 0 else 1
            rolling_beta.append(beta)
            dates.append(series.index[i])
        ax2.plot(dates, rolling_beta, label=name, linewidth=2)
    
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Beta = 1')
    ax2.set_title('Rolling 52-Week Beta vs ACWI', fontweight='bold')
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
    """Print formatted impact summary."""
    
    print("\n" + "="*80)
    print(f"IMPACT ANALYSIS: Adding {candidate_name} at {allocation_pct:.0%} Allocation")
    print("="*80)
    
    # Key Changes
    print("\n📊 KEY CHANGES:")
    print("-"*40)
    
    vol_change = metrics_after['Annualized Volatility (%)'] - metrics_before['Annualized Volatility (%)']
    ret_change = metrics_after['Annualized Return (%)'] - metrics_before['Annualized Return (%)']
    sharpe_change = metrics_after['Sharpe Ratio'] - metrics_before['Sharpe Ratio']
    beta_change = metrics_after['Beta vs ACWI'] - metrics_before['Beta vs ACWI']
    dd_change = metrics_after['Max Drawdown (%)'] - metrics_before['Max Drawdown (%)']
    
    print(f"  • Return:     {ret_change:+.2f}% ({'↑' if ret_change > 0 else '↓'})")
    print(f"  • Volatility: {vol_change:+.2f}% ({'↓ Better' if vol_change < 0 else '↑ Higher Risk'})")
    print(f"  • Sharpe:     {sharpe_change:+.3f} ({'↑ Better' if sharpe_change > 0 else '↓ Worse'})")
    print(f"  • Beta:       {beta_change:+.3f} ({'↓ More Defensive' if beta_change < 0 else '↑ More Aggressive'})")
    print(f"  • Max DD:     {dd_change:+.2f}% ({'↑ Worse' if dd_change > 0 else '↓ Better'})")
    
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

def run_backtest(ticker: str, name: str, allocation: float = 0.05, 
                 output_dir: str = '.', show_plot: bool = True) -> dict:
    """
    Run backtest for a candidate stock. Callable from external scripts.
    
    Parameters
    ----------
    ticker : str
        Candidate stock ticker
    name : str
        Display name for reports
    allocation : float
        Target allocation (e.g., 0.05 for 5%)
    output_dir : str
        Directory to save outputs
    show_plot : bool
        Whether to display the plot
    
    Returns
    -------
    dict
        Dictionary with backtest results and key metrics
    """
    import os
    
    # Build original portfolio
    weights = build_portfolio_weights()
    
    # Download data
    all_tickers = list(weights.keys()) + [BENCHMARK_TICKER, ticker, TECH_ETF_TICKER]
    all_tickers = list(set(all_tickers))
    prices = download_data(all_tickers, LOOKBACK_YEARS)
    
    # Compute returns
    old_portfolio, _ = compute_portfolio_returns(prices, weights, RESAMPLE_FREQ)
    old_portfolio.name = 'Original_Portfolio'
    
    if RESAMPLE_FREQ == 'W':
        prices_resampled = prices.resample('W-FRI').last()
    else:
        prices_resampled = prices
    
    returns = prices_resampled.pct_change().dropna()
    benchmark_returns = returns[BENCHMARK_TICKER]
    candidate_returns = returns[ticker]
    tech_returns = returns[TECH_ETF_TICKER]
    
    # New portfolio
    new_portfolio = construct_new_portfolio(old_portfolio, candidate_returns, allocation)
    
    # Metrics
    periods_per_year = 52 if RESAMPLE_FREQ == 'W' else 252
    metrics_before = calculate_metrics(old_portfolio, benchmark_returns, RISK_FREE_RATE, periods_per_year)
    metrics_after = calculate_metrics(new_portfolio, benchmark_returns, RISK_FREE_RATE, periods_per_year)
    metrics_candidate = calculate_metrics(candidate_returns, benchmark_returns, RISK_FREE_RATE, periods_per_year)
    
    # Correlations
    correlations, corr_data = calculate_correlations(
        candidate_returns, old_portfolio, tech_returns, benchmark_returns
    )
    
    # Save comparison
    comparison_df = pd.DataFrame({
        'Original Portfolio': metrics_before,
        'New Portfolio': metrics_after,
        f'{name} (Standalone)': metrics_candidate,
    }).T
    
    csv_path = os.path.join(output_dir, 'backtest.csv')
    comparison_df.to_csv(csv_path)
    
    # Plot
    chart_path = os.path.join(output_dir, 'backtest.png')
    if show_plot:
        plot_analysis(old_portfolio, new_portfolio, candidate_returns, benchmark_returns,
                     corr_data, name, allocation, save_path=chart_path)
    else:
        # Save without showing
        import matplotlib
        matplotlib.use('Agg')
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(16, 12))
        
        # Recreate plots (simplified for headless)
        aligned = pd.concat([old_portfolio, new_portfolio, benchmark_returns, candidate_returns], axis=1).dropna()
        aligned.columns = ['Original', 'New', 'Benchmark', 'Candidate']
        
        ax1 = fig.add_subplot(2, 2, 1)
        cumulative = (1 + aligned[['Original', 'New', 'Benchmark']]).cumprod()
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
    return {
        'ticker': ticker,
        'name': name,
        'allocation': allocation,
        'return_change': f"{metrics_after['Annualized Return (%)'] - metrics_before['Annualized Return (%)']:+.2f}%",
        'vol_change': f"{metrics_after['Annualized Volatility (%)'] - metrics_before['Annualized Volatility (%)']:+.2f}%",
        'sharpe_change': f"{metrics_after['Sharpe Ratio'] - metrics_before['Sharpe Ratio']:+.3f}",
        'beta_change': f"{metrics_after['Beta vs ACWI'] - metrics_before['Beta vs ACWI']:+.3f}",
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'correlations': correlations,
        'csv_path': csv_path,
        'chart_path': chart_path,
    }


def main():
    """Main execution function."""
    
    print("="*80)
    print(f"CANDIDATE STOCK ANALYSIS: {CANDIDATE_NAME} ({CANDIDATE_TICKER})")
    print(f"Target Allocation: {ALLOCATION_SIZE:.0%}")
    print("="*80)
    
    # Step 1: Build original portfolio weights
    print("\n" + "-"*60)
    print("STEP 1: Reconstructing Original Portfolio")
    print("-"*60)
    weights = build_portfolio_weights()
    
    # Step 2: Download all data
    print("\n" + "-"*60)
    print("STEP 2: Downloading Market Data")
    print("-"*60)
    
    all_tickers = list(weights.keys()) + [BENCHMARK_TICKER, CANDIDATE_TICKER, TECH_ETF_TICKER]
    all_tickers = list(set(all_tickers))  # Remove duplicates
    
    prices = download_data(all_tickers, LOOKBACK_YEARS)
    print(f"Data range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Step 3: Compute portfolio returns
    print("\n" + "-"*60)
    print("STEP 3: Computing Portfolio Returns")
    print("-"*60)
    
    old_portfolio, all_returns = compute_portfolio_returns(prices, weights, RESAMPLE_FREQ)
    old_portfolio.name = 'Original_Portfolio'
    
    # Get individual return series
    if RESAMPLE_FREQ == 'W':
        prices_resampled = prices.resample('W-FRI').last()
    else:
        prices_resampled = prices
    
    returns = prices_resampled.pct_change().dropna()
    benchmark_returns = returns[BENCHMARK_TICKER]
    candidate_returns = returns[CANDIDATE_TICKER]
    tech_returns = returns[TECH_ETF_TICKER]
    
    print(f"Original portfolio returns: {len(old_portfolio)} periods")
    
    # Step 4: Construct new portfolio
    print("\n" + "-"*60)
    print("STEP 4: Constructing Pro-Forma Portfolio")
    print("-"*60)
    
    new_portfolio = construct_new_portfolio(old_portfolio, candidate_returns, ALLOCATION_SIZE)
    print(f"New portfolio: {(1-ALLOCATION_SIZE)*100:.1f}% Original + {ALLOCATION_SIZE*100:.1f}% {CANDIDATE_NAME}")
    
    # Step 5: Calculate metrics
    print("\n" + "-"*60)
    print("STEP 5: Calculating Risk Metrics")
    print("-"*60)
    
    periods_per_year = 52 if RESAMPLE_FREQ == 'W' else 252
    
    metrics_before = calculate_metrics(old_portfolio, benchmark_returns, RISK_FREE_RATE, periods_per_year)
    metrics_after = calculate_metrics(new_portfolio, benchmark_returns, RISK_FREE_RATE, periods_per_year)
    metrics_candidate = calculate_metrics(candidate_returns, benchmark_returns, RISK_FREE_RATE, periods_per_year)
    metrics_benchmark = calculate_metrics(benchmark_returns, benchmark_returns, RISK_FREE_RATE, periods_per_year)
    
    # Step 6: Create comparison table
    print("\n" + "="*80)
    print("METRICS COMPARISON TABLE")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Original Portfolio': metrics_before,
        'New Portfolio': metrics_after,
        f'{CANDIDATE_NAME} (Standalone)': metrics_candidate,
        'ACWI Benchmark': metrics_benchmark
    }).T
    
    # Format and display
    print("\n")
    print(comparison_df.round(3).to_string())
    
    # Save to CSV
    comparison_df.to_csv(f'backtest_{CANDIDATE_TICKER}_metrics.csv')
    print(f"\nMetrics saved to: backtest_{CANDIDATE_TICKER}_metrics.csv")
    
    # Step 7: Correlation analysis
    print("\n" + "-"*60)
    print("STEP 6: Correlation Deep Dive")
    print("-"*60)
    
    correlations, corr_data = calculate_correlations(
        candidate_returns, old_portfolio, tech_returns, benchmark_returns
    )
    
    print(f"\n{CANDIDATE_NAME} Correlations:")
    for key, val in correlations.items():
        print(f"  • {key}: {val:.4f}")
    
    # Step 8: Impact summary
    print_impact_summary(metrics_before, metrics_after, metrics_candidate, 
                        correlations, CANDIDATE_NAME, ALLOCATION_SIZE)
    
    # Step 9: Visualization
    print("\n" + "-"*60)
    print("STEP 7: Generating Visualizations")
    print("-"*60)
    
    plot_analysis(old_portfolio, new_portfolio, candidate_returns, benchmark_returns,
                  corr_data, CANDIDATE_NAME, ALLOCATION_SIZE,
                  save_path=f'backtest_{CANDIDATE_TICKER}_analysis.png')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return comparison_df, correlations


if __name__ == "__main__":
    results, correlations = main()

