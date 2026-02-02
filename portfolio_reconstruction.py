"""
ECE Portfolio Analysis Engine
=============================
Portfolio Reconstruction and Risk Analysis
==========================================
Reconstructs a full portfolio from partial holdings (Top 10) and sector proxy ETFs,
then calculates key risk metrics.

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
# CONFIGURATION
# =============================================================================
from config import AnalysisConfig
from portfolio_loader import SECTOR_ETF_MAP

# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_data(tickers: list, config: AnalysisConfig) -> pd.DataFrame:
    """
    Download adjusted close prices for given tickers using yfinance.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.lookback_years * 365)
    
    print(f"Downloading data for {len(tickers)} tickers...")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # 1. First attempt with original tickers
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True
    )
    
    # Extract 'Close' prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers
    
    # 2. Identify failed tickers
    failed_tickers = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    
    if failed_tickers:
        print(f"   ⚠️ Initial download failed for: {failed_tickers}")
        print(f"   🔄 Retrying with ^ prefix...")
        
        # 3. Try fallback with ^ prefix for failed tickers
        retry_map = {f"^{t}": t for t in failed_tickers if not t.startswith('^')}
        
        if retry_map:
            retry_list = list(retry_map.keys())
            retry_data = yf.download(
                retry_list,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            
            if not retry_data.empty:
                if isinstance(retry_data.columns, pd.MultiIndex):
                    retry_prices = retry_data['Close']
                else:
                    retry_prices = retry_data[['Close']]
                    retry_prices.columns = retry_list
                
                # Merge successful retries
                for caret_ticker, original_ticker in retry_map.items():
                    if caret_ticker in retry_prices.columns and not retry_prices[caret_ticker].isna().all():
                        print(f"   ✅ Successfully recovered {original_ticker} as {caret_ticker}")
                        prices[original_ticker] = retry_prices[caret_ticker]
    
    # Final Validation
    nan_cols = [c for c in prices.columns if prices[c].isna().all()]
    missing_cols = [t for t in tickers if t not in prices.columns]
    bad_tickers = set(nan_cols + missing_cols)
    
    if bad_tickers:
        error_msg = f"Error: Could not retrieve data for: {', '.join(bad_tickers)}. Please check symbol validity."
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    return prices


def clean_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data: forward fill and drop remaining NaN rows.
    """
    # Forward fill missing values
    prices = prices.ffill()
    
    # Drop rows with any remaining NaN (typically at the start)
    prices = prices.dropna()
    
    print(f"Data cleaned. Shape: {prices.shape}")
    print(f"Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    
    return prices


# =============================================================================
# PORTFOLIO RECONSTRUCTION LOGIC
# =============================================================================

def calculate_sector_allocations(top_holdings: dict, target_sector_weights: dict) -> dict:
    """
    Calculate the weight used by Top 10 holdings per sector and remaining weight
    to be allocated to sector ETFs.
    """
    # Calculate weight used by Top 10 per sector
    sector_used_weights = {}
    for ticker, info in top_holdings.items():
        sector = info['sector']
        weight = info['weight']
        sector_used_weights[sector] = sector_used_weights.get(sector, 0) + weight
    
    print("\n" + "="*60)
    print("SECTOR ALLOCATION ANALYSIS")
    print("="*60)
    
    allocations = {}
    for sector, target_weight in target_sector_weights.items():
        used_weight = sector_used_weights.get(sector, 0)
        remaining_weight = max(0, target_weight - used_weight)  # Clip to 0 if negative
        
        allocations[sector] = {
            'target_weight': target_weight,
            'used_by_top10': used_weight,
            'remaining_for_etf': remaining_weight,
            'etf_proxy': SECTOR_ETF_MAP.get(sector, None)
        }
        
        print(f"\n{sector}:")
        print(f"  Target Weight:     {target_weight:6.2f}%")
        print(f"  Used by Top 10:    {used_weight:6.2f}%")
        print(f"  Remaining for ETF: {remaining_weight:6.2f}%")
        if remaining_weight < target_weight and used_weight > 0:
            print(f"  → Top 10 stocks cover {used_weight/target_weight*100:.1f}% of sector allocation")
    
    return allocations


def build_portfolio_weights(top_holdings: dict, target_sector_weights: dict) -> dict:
    """
    Build the complete portfolio weights combining Top 10 holdings and sector ETF proxies.
    """
    weights = {}
    
    # Add Top 10 holdings
    for ticker, info in top_holdings.items():
        weights[ticker] = info['weight']
    
    # Calculate sector allocations
    sector_allocations = calculate_sector_allocations(top_holdings, target_sector_weights)
    
    # Add sector ETF proxies for remaining weights
    for sector, alloc in sector_allocations.items():
        if alloc['remaining_for_etf'] > 0 and alloc['etf_proxy']:
            etf = alloc['etf_proxy']
            weights[etf] = weights.get(etf, 0) + alloc['remaining_for_etf']
    
    # Validate total weights
    total_weight = sum(weights.values())
    print(f"\n{'='*60}")
    print(f"PORTFOLIO WEIGHT VALIDATION")
    print(f"{'='*60}")
    print(f"Total Weight (before normalization): {total_weight:.2f}%")
    
    # Normalize if not exactly 100%
    if abs(total_weight - 100) > 0.01:
        print(f"Normalizing weights to 100%...")
        for ticker in weights:
            weights[ticker] = weights[ticker] / total_weight * 100
        print(f"Total Weight (after normalization): {sum(weights.values()):.2f}%")
    
    return weights


def display_portfolio_composition(weights: dict, top_holdings: dict):
    """
    Display the final portfolio composition in a formatted table.
    """
    print(f"\n{'='*60}")
    print("FINAL PORTFOLIO COMPOSITION")
    print(f"{'='*60}")
    
    # Separate Top 10 from ETFs
    top10_tickers = set(top_holdings.keys())
    
    print("\n--- Top 10 Holdings ---")
    top10_total = 0
    for ticker in sorted(top10_tickers, key=lambda x: weights.get(x, 0), reverse=True):
        if ticker in weights:
            sector = top_holdings[ticker]['sector']
            print(f"  {ticker:10s} {weights[ticker]:6.2f}%  ({sector})")
            top10_total += weights[ticker]
    print(f"  {'SUBTOTAL':10s} {top10_total:6.2f}%")
    
    print("\n--- Sector ETF Proxies ---")
    etf_total = 0
    etf_weights = {k: v for k, v in weights.items() if k not in top10_tickers}
    for ticker in sorted(etf_weights.keys(), key=lambda x: etf_weights[x], reverse=True):
        # Find which sector this ETF represents
        sector = [s for s, e in SECTOR_ETF_MAP.items() if e == ticker]
        sector_str = sector[0] if sector else "Unknown"
        print(f"  {ticker:10s} {weights[ticker]:6.2f}%  ({sector_str})")
        etf_total += weights[ticker]
    print(f"  {'SUBTOTAL':10s} {etf_total:6.2f}%")
    
    print(f"\n  {'TOTAL':10s} {top10_total + etf_total:6.2f}%")


def save_portfolio_composition(weights: dict, top_holdings: dict, save_path: str = 'portfolio_composition.txt'):
    """
    Save the portfolio composition to a formatted text file.
    """
    lines = []
    
    # Header
    header = f"{'Company Name':<40} {'Ticker':<10} {'Sector':<30} {'Region':<20} {'Weight (%)':<10}"
    separator = "=" * 110
    
    lines.append(separator)
    lines.append("PORTFOLIO COMPOSITION")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(separator)
    lines.append("")
    lines.append(header)
    lines.append("-" * 110)
    
    # Top 10 Holdings section
    lines.append("")
    lines.append("--- TOP 10 HOLDINGS ---")
    lines.append("")
    
    top10_tickers = set(top_holdings.keys())
    top10_total = 0
    
    for ticker in sorted(top10_tickers, key=lambda x: weights.get(x, 0), reverse=True):
        if ticker in weights:
            info = top_holdings[ticker]
            name = info['name']
            sector = info['sector']
            region = info['country']
            weight = weights[ticker]
            top10_total += weight
            
            # Format: Company Name (Exchange:Ticker)
            display_name = f"{name} ({ticker})"
            line = f"{display_name:<40} {ticker:<10} {sector:<30} {region:<20} {weight:.1f}%"
            lines.append(line)
    
    lines.append(f"{'SUBTOTAL':<40} {'':<10} {'':<30} {'':<20} {top10_total:.1f}%")
    
    # Sector ETF Proxies section
    lines.append("")
    lines.append("--- SECTOR ETF PROXIES (Rest of Portfolio) ---")
    lines.append("")
    
    etf_total = 0
    etf_weights = {k: v for k, v in weights.items() if k not in top10_tickers}
    
    for ticker in sorted(etf_weights.keys(), key=lambda x: etf_weights[x], reverse=True):
        # Find sector for this ETF
        sector_list = [s for s, e in SECTOR_ETF_MAP.items() if e == ticker]
        sector = sector_list[0] if sector_list else "Unknown"
        name = ETF_NAMES.get(ticker, ticker)
        region = "Global"  # ETFs are global
        weight = weights[ticker]
        etf_total += weight
        
        line = f"{name:<40} {ticker:<10} {sector:<30} {region:<20} {weight:.1f}%"
        lines.append(line)
    
    lines.append(f"{'SUBTOTAL':<40} {'':<10} {'':<30} {'':<20} {etf_total:.1f}%")
    
    # Total
    lines.append("")
    lines.append("-" * 110)
    lines.append(f"{'PORTFOLIO TOTAL':<40} {'':<10} {'':<30} {'':<20} {top10_total + etf_total:.1f}%")
    lines.append(separator)
    
    # Write to file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\nPortfolio composition saved to: {save_path}")
    
    return save_path


# =============================================================================
# PORTFOLIO RETURNS CONSTRUCTION
# =============================================================================

def compute_portfolio_returns(prices: pd.DataFrame, weights: dict, 
                              resample_freq: str = 'W') -> pd.Series:
    """
    Compute portfolio return series from constituent returns and weights.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data for all constituents
    weights : dict
        Portfolio weights (in %)
    resample_freq : str
        Resampling frequency ('D' for daily, 'W' for weekly)
    
    Returns
    -------
    pd.Series
        Portfolio return series
    """
    # Compute returns
    if resample_freq == 'W':
        prices_resampled = prices.resample('W-FRI').last()
    else:
        prices_resampled = prices
    
    returns = prices_resampled.pct_change().dropna()
    
    # Filter to only tickers we have in both prices and weights
    available_tickers = [t for t in weights.keys() if t in returns.columns]
    missing_tickers = [t for t in weights.keys() if t not in returns.columns]
    
    if missing_tickers:
        print(f"\nWarning: Missing data for tickers: {missing_tickers}")
        print("These will be excluded from the portfolio calculation.")
    
    # Normalize weights for available tickers
    available_weights = {t: weights[t] for t in available_tickers}
    total_available = sum(available_weights.values())
    normalized_weights = {t: w / total_available * 100 for t, w in available_weights.items()}
    
    # Compute weighted portfolio returns
    portfolio_returns = pd.Series(0, index=returns.index)
    for ticker, weight in normalized_weights.items():
        portfolio_returns += returns[ticker] * (weight / 100)
    
    portfolio_returns.name = 'Reconstructed_Portfolio'
    
    return portfolio_returns, returns


# =============================================================================
# RISK METRICS CALCULATION
# =============================================================================

def calculate_risk_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                           risk_free_rate: float = 0.04, 
                           periods_per_year: int = 52,
                           benchmark_name: str = "Benchmark") -> dict:
    """
    Calculate comprehensive risk metrics for the portfolio.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio return series
    benchmark_returns : pd.Series
        Benchmark return series
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year (52 for weekly, 252 for daily)
    benchmark_name : str
        Name of the benchmark (e.g. ACWI)
    
    Returns
    -------
    dict
        Dictionary of risk metrics
    """
    metrics = {}
    
    # Align series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]
    
    # 1. Annualized Return
    total_return = (1 + port_ret).prod() - 1
    n_periods = len(port_ret)
    years = n_periods / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1
    metrics['Annualized Return (%)'] = annualized_return * 100
    
    # 2. Annualized Volatility
    volatility = port_ret.std() * np.sqrt(periods_per_year)
    metrics['Annualized Volatility (%)'] = volatility * 100
    
    # 3. Sharpe Ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility != 0 else 0
    metrics['Sharpe Ratio'] = sharpe_ratio
    
    # 4. Beta vs Benchmark
    covariance = port_ret.cov(bench_ret)
    benchmark_variance = bench_ret.var()
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
    metrics[f'Beta vs {benchmark_name}'] = beta
    
    # 5. Alpha (annualized)
    bench_total_return = (1 + bench_ret).prod() - 1
    bench_annualized_return = (1 + bench_total_return) ** (1 / years) - 1
    alpha = annualized_return - (risk_free_rate + beta * (bench_annualized_return - risk_free_rate))
    metrics['Alpha (%)'] = alpha * 100
    
    # 6. Historical VaR (95%)
    var_95 = np.percentile(port_ret, 5)  # 5th percentile = 95% VaR
    metrics['VaR (95%, period)'] = var_95 * 100
    
    # Annualized VaR (approximate)
    var_95_annual = var_95 * np.sqrt(periods_per_year)
    metrics['VaR (95%, annualized)'] = var_95_annual * 100
    
    # 7. Maximum Drawdown
    cumulative = (1 + port_ret).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    metrics['Max Drawdown (%)'] = max_drawdown * 100
    
    # 8. Correlation with Benchmark
    correlation = port_ret.corr(bench_ret)
    metrics[f'Correlation vs {benchmark_name}'] = correlation
    
    # 9. Tracking Error
    tracking_diff = port_ret - bench_ret
    tracking_error = tracking_diff.std() * np.sqrt(periods_per_year)
    metrics['Tracking Error (%)'] = tracking_error * 100
    
    # 10. Information Ratio
    excess_annual = metrics['Annualized Return (%)'] - (bench_annualized_return * 100)
    info_ratio = (excess_annual / 100) / tracking_error if tracking_error != 0 else 0
    metrics['Information Ratio'] = info_ratio
    
    return metrics


def display_risk_metrics(metrics: dict):
    """
    Display risk metrics in a formatted table.
    
    Parameters
    ----------
    metrics : dict
        Risk metrics dictionary
    """
    print(f"\n{'='*60}")
    print("RISK METRICS SUMMARY")
    print(f"{'='*60}\n")
    
    # Create DataFrame for nice display
    df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    
    # Format values
    format_pct = ['Annualized Return (%)', 'Annualized Volatility (%)', 'Alpha (%)',
                  'VaR (95%, period)', 'VaR (95%, annualized)', 'Max Drawdown (%)', 
                  'Tracking Error (%)']
    
    for idx in df.index:
        if idx in format_pct:
            df.loc[idx, 'Formatted'] = f"{df.loc[idx, 'Value']:.2f}%"
        elif idx in ['Sharpe Ratio', 'Information Ratio'] or idx.startswith('Beta vs'):
            df.loc[idx, 'Formatted'] = f"{df.loc[idx, 'Value']:.3f}"
        elif idx.startswith('Correlation vs'):
            df.loc[idx, 'Formatted'] = f"{df.loc[idx, 'Value']:.4f}"
        else:
            df.loc[idx, 'Formatted'] = f"{df.loc[idx, 'Value']:.4f}"
    
    for idx, row in df.iterrows():
        print(f"  {idx:30s} : {row['Formatted']:>12s}")
    
    return df


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_cumulative_returns(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                            benchmark_name: str = 'ACWI',
                            save_path: str = None,
                            freq_label: str = 'Weekly',
                            periods_per_year: int = 52):
    """
    Plot cumulative returns comparison: Portfolio vs Benchmark.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio return series
    benchmark_returns : pd.Series
        Benchmark return series
    benchmark_name : str
        Name of the benchmark ticker (default: ACWI)
    save_path : str, optional
        Path to save the figure
    freq_label : str
        Label for frequency (e.g. 'Weekly', 'Daily')
    periods_per_year : int
        Annualization factor (52 for weekly, 252 for daily)
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Portfolio Analysis: Reconstructed Portfolio vs {benchmark_name} Benchmark', 
                 fontsize=14, fontweight='bold')
    
    # Align data
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ['Reconstructed Portfolio', f'{benchmark_name} Benchmark']
    
    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    cumulative = (1 + aligned).cumprod()
    cumulative.plot(ax=ax1, linewidth=2)
    ax1.set_title('Cumulative Returns', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return (1 = 100%)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add final values annotation
    for col in cumulative.columns:
        final_val = cumulative[col].iloc[-1]
        ax1.annotate(f'{final_val:.2f}', 
                     xy=(cumulative.index[-1], final_val),
                     xytext=(10, 0), textcoords='offset points',
                     fontsize=10, fontweight='bold')
    
    # 2. Rolling Volatility
    # Window: ~3 months (12 weeks or 63 days)
    window = 12 if periods_per_year == 52 else 63
    unit = 'Week' if periods_per_year == 52 else 'Day'
    
    ax2 = axes[0, 1]
    rolling_vol = aligned.rolling(window=window).std() * np.sqrt(periods_per_year) * 100
    rolling_vol.plot(ax=ax2, linewidth=2)
    ax2.set_title(f'{window}-{unit} Rolling Volatility (Annualized)', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility (%)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = axes[1, 0]
    for i, col in enumerate(aligned.columns):
        cum = (1 + aligned[col]).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max * 100
        ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.4, label=col)
        ax3.plot(drawdown.index, drawdown, linewidth=1)
    ax3.set_title('Drawdown Analysis', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Return Distribution
    ax4 = axes[1, 1]
    for col in aligned.columns:
        ax4.hist(aligned[col] * 100, bins=50, alpha=0.5, label=col, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax4.set_title(f'{freq_label} Return Distribution', fontweight='bold')
    ax4.set_xlabel(f'{freq_label} Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function for portfolio reconstruction and analysis.
    """
    from portfolio_loader import DEFAULT_TOP_HOLDINGS, DEFAULT_SECTOR_TARGETS
    
    config = AnalysisConfig()
    
    print("="*60)
    print("PORTFOLIO RECONSTRUCTION & RISK ANALYSIS")
    print("="*60)
    print("\nThis script reconstructs a portfolio from:")
    print("  - Known holdings")
    print("  - Sector ETF proxies for remaining weights")
    print(f"  - Benchmark: {config.benchmark_ticker}")
    
    # Step 1: Build portfolio weights
    print("\n" + "-"*60)
    print("STEP 1: Building Portfolio Weights")
    print("-"*60)
    weights = build_portfolio_weights(DEFAULT_TOP_HOLDINGS, DEFAULT_SECTOR_TARGETS)
    display_portfolio_composition(weights, DEFAULT_TOP_HOLDINGS)
    
    # Step 2: Collect all tickers
    all_tickers = list(weights.keys()) + [config.benchmark_ticker]
    print(f"\n\nTotal tickers to download: {len(all_tickers)}")
    
    # Step 3: Download data
    print("\n" + "-"*60)
    print("STEP 2: Downloading Price Data")
    print("-"*60)
    prices = download_data(all_tickers, config)
    prices = clean_data(prices)
    
    # Step 4: Compute portfolio returns
    print("\n" + "-"*60)
    print("STEP 3: Computing Portfolio Returns")
    print("-"*60)
    portfolio_returns, constituent_returns = compute_portfolio_returns(
        prices, weights, config.resample_freq
    )
    
    # Compute benchmark returns
    if config.resample_freq == 'W':
        benchmark_prices = prices[config.benchmark_ticker].resample('W-FRI').last()
    else:
        benchmark_prices = prices[config.benchmark_ticker]
    benchmark_returns = benchmark_prices.pct_change().dropna()
    benchmark_returns.name = config.benchmark_ticker
    
    print(f"Portfolio returns computed: {len(portfolio_returns)} periods")
    print(f"Frequency: {'Weekly' if config.resample_freq == 'W' else 'Daily'}")
    
    # Step 5: Calculate risk metrics
    print("\n" + "-"*60)
    print("STEP 4: Calculating Risk Metrics")
    print("-"*60)
    periods_per_year = 52 if config.resample_freq == 'W' else 252
    metrics = calculate_risk_metrics(
        portfolio_returns, benchmark_returns,
        risk_free_rate=config.risk_free_rate,
        periods_per_year=periods_per_year
    )
    display_risk_metrics(metrics)
    
    # Step 6: Create summary DataFrame
    print("\n" + "-"*60)
    print("STEP 5: Creating Summary Output")
    print("-"*60)
    
    summary = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    summary.set_index('Metric', inplace=True)
    
    # Save summary to CSV
    summary_path = 'portfolio_risk_metrics.csv'
    summary.to_csv(summary_path)
    print(f"\nRisk metrics saved to: {summary_path}")
    
    # Save portfolio composition to text file
    save_portfolio_composition(weights, DEFAULT_TOP_HOLDINGS, 'portfolio_composition.txt')
    
    # Step 7: Visualization
    print("\n" + "-"*60)
    print("STEP 6: Generating Visualizations")
    print("-"*60)
    freq_label = 'Weekly' if config.resample_freq == 'W' else 'Daily'
    periods = 52 if config.resample_freq == 'W' else 252
    
    plot_cumulative_returns(
        portfolio_returns, 
        benchmark_returns,
        benchmark_name=config.benchmark_ticker,
        save_path='portfolio_analysis_chart.png',
        freq_label=freq_label,
        periods_per_year=periods
    )
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print(f"  • Annualized Return:    {metrics['Annualized Return (%)']:.2f}%")
    print(f"  • Annualized Volatility: {metrics['Annualized Volatility (%)']:.2f}%")
    print(f"  • Sharpe Ratio:          {metrics['Sharpe Ratio']:.3f}")
    
    beta_key = f'Beta vs {config.benchmark_ticker}'
    if beta_key in metrics:
        print(f"  • {beta_key}:         {metrics[beta_key]:.3f}")
    
    # Dynamic VaR label
    var_label = f"VaR (95%, {freq_label.lower()}):"
    print(f"  • {var_label:21s} {metrics['VaR (95%, period)']:.2f}%")
    print(f"  • Max Drawdown:          {metrics['Max Drawdown (%)']:.2f}%")
    
    return summary, portfolio_returns, benchmark_returns


if __name__ == "__main__":
    summary, port_ret, bench_ret = main()
