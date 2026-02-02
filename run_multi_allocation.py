"""
Multi-Allocation Metrics Generator
===================================
Takes a ticker as input, finds optimal allocation using optimal_allocation.py,
then computes backtest metrics at 0.5% granularity from 0% to optimal allocation.

Usage: python run_multi_allocation.py <TICKER> [STOCK_NAME] [GRANULARITY_PCT]
Example: python run_multi_allocation.py COR "Cencora Inc." 0.5
"""
import pandas as pd
import numpy as np
import sys
import os
import argparse

# Ensure we can import from current directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import AnalysisConfig
import portfolio_loader as pl
from optimal_allocation import optimize_allocation
from backtest_candidate import run_backtest

def run_multi_allocation_analysis(ticker: str, name: str = None, granularity: float = 0.005,
                                  config: AnalysisConfig = None,
                                  holdings: dict = None, sector_targets: dict = None):
    """
    Run multi-allocation analysis for a given ticker.
    """
    if name is None:
        name = ticker
    
    if config is None:
        config = AnalysisConfig()
    if holdings is None:
        holdings = pl.DEFAULT_TOP_HOLDINGS
    if sector_targets is None:
        sector_targets = pl.DEFAULT_SECTOR_TARGETS
    
    gran_pct = granularity * 100
    print("="*70)
    print(f"MULTI-ALLOCATION ANALYSIS: {name} ({ticker}) - {gran_pct:.2f}% Step")
    print("="*70)
    
    # Step 1: Find optimal allocation
    print("\n[STEP 1] Finding optimal allocation...")
    try:
        opt_result = optimize_allocation(ticker, name, config, holdings, sector_targets)
        optimal_alloc = opt_result.recommended_allocation
        print(f"\n   ✓ Optimal Allocation: {optimal_alloc*100:.1f}%")
    except Exception as e:
        print(f"   ✗ Error finding optimal allocation: {e}")
        return None
    
    # Step 2: Generate allocation levels
    print(f"\n[STEP 2] Generating allocation levels ({gran_pct:.2f}% granularity)...")
    
    # Start from granularity, go up to optimal allocation (rounded up to nearest step)
    max_alloc = np.ceil(optimal_alloc / granularity) * granularity
    allocations = np.arange(granularity, max_alloc + 0.000001, granularity)
    allocations = [round(a, 4) for a in allocations if a <= max_alloc + 0.000001]
    
    # Add the exact optimal allocation if not already included
    if optimal_alloc not in allocations:
        allocations.append(optimal_alloc)
        allocations = sorted(allocations)
    
    print(f"   Testing {len(allocations)} allocation levels: {allocations[0]*100:.2f}% to {allocations[-1]*100:.2f}%")
    
    # Step 3: Run backtest at each allocation level
    print("\n[STEP 3] Running backtests...")
    
    all_results = []
    for i, alloc in enumerate(allocations, 1):
        pct = alloc * 100
        print(f"   [{i}/{len(allocations)}] {pct:.1f}%...", end=" ")
        
        try:
            results = run_backtest(ticker, name, 
                                   config=config, top_holdings=holdings, sector_targets=sector_targets,
                                   allocation=alloc, 
                                   output_dir='.', show_plot=False)
            
            # Create row with allocation and all metrics
            row = {'Allocation (%)': pct}
            
            # Add flag if this is the optimal allocation
            row['Is Optimal'] = 'Yes' if abs(alloc - optimal_alloc) < 0.001 else 'No'
            
            # Add all metrics from new portfolio
            for key, val in results['metrics_after'].items():
                row[key] = val
            
            # Add change metrics vs original
            row['Return Change (%)'] = results['metrics_after']['Annualized Return (%)'] - results['metrics_before']['Annualized Return (%)']
            row['Vol Change (%)'] = results['metrics_after']['Annualized Volatility (%)'] - results['metrics_before']['Annualized Volatility (%)']
            row['Sharpe Change'] = results['metrics_after']['Sharpe Ratio'] - results['metrics_before']['Sharpe Ratio']
            row['VaR Ann. Change (%)'] = results['metrics_after']['VaR (95%, annualized)'] - results['metrics_before']['VaR (95%, annualized)']
            row['Max DD Change (%)'] = results['metrics_after']['Max Drawdown (%)'] - results['metrics_before']['Max Drawdown (%)']
            
            all_results.append(row)
            
            sharpe = results['metrics_after']['Sharpe Ratio']
            print(f"Sharpe: {sharpe:.3f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Step 4: Create master DataFrame
    print("\n[STEP 4] Creating master CSV...")
    
    master_df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    benchmark = config.benchmark_ticker
    cols_order = [
        'Allocation (%)',
        'Is Optimal',
        'Annualized Return (%)',
        'Annualized Volatility (%)',
        'Sharpe Ratio',
        f'Beta vs {benchmark}',
        'Alpha (%)',
        'VaR (95%, period)',
        'VaR (95%, annualized)',
        'Max Drawdown (%)',
        f'Correlation vs {benchmark}',
        'Tracking Error (%)',
        'Information Ratio',
        'Return Change (%)',
        'Vol Change (%)',
        'Sharpe Change',
        'VaR Ann. Change (%)',
        'Max DD Change (%)'
    ]
    
    # Only include columns that exist
    cols_order = [c for c in cols_order if c in master_df.columns]
    master_df = master_df[cols_order]
    
    # Save to CSV
    output_path = f'{ticker}_master_metrics.csv'
    master_df.to_csv(output_path, index=False)
    
    print(f"\n   ✓ Saved: {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    summary_cols = ['Allocation (%)', 'Is Optimal', 'Annualized Return (%)', 
                    'Annualized Volatility (%)', 'Sharpe Ratio', 'VaR (95%, annualized)']
    print(master_df[summary_cols].round(2).to_string(index=False))
    
    # Highlight optimal row
    optimal_row = master_df[master_df['Is Optimal'] == 'Yes']
    if not optimal_row.empty:
        print("\n" + "="*70)
        print(f"OPTIMAL ALLOCATION: {optimal_alloc*100:.1f}%")
        print("="*70)
        for col in ['Annualized Return (%)', 'Sharpe Ratio', 'VaR (95%, annualized)', 'Max Drawdown (%)']:
            print(f"   {col}: {optimal_row[col].values[0]:.2f}")
    
    print(f"\n✓ Analysis complete. Results saved to: {output_path}")
    
    return master_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multi-allocation analysis')
    parser.add_argument('ticker', type=str, help='Ticker symbols')
    parser.add_argument('name', type=str, nargs='?', help='Company name')
    parser.add_argument('granularity', type=float, nargs='?', default=0.5, help='Granularity % (default: 0.5)')
    
    # Config args
    parser.add_argument('--risk-aversion', type=float, help='Risk aversion')
    parser.add_argument('--concentration-penalty', type=float, help='Concentration penalty')
    parser.add_argument('--min-recommended', type=float, help='Min recommended allocation')
    parser.add_argument('--min-allocation', type=float, help='Min allocation')
    parser.add_argument('--max-allocation', type=float, help='Max allocation')
    parser.add_argument('--benchmark', type=str, help='Benchmark ticker')
    parser.add_argument('--risk-free-rate', type=float, help='Risk free rate')
    parser.add_argument('--lookback-years', type=int, help='Lookback years')
    parser.add_argument('--resample-freq', type=str, choices=['D', 'W', 'M'], help='Resample frequency')
    parser.add_argument('--tech-etf', type=str, help='Tech ETF ticker')
    parser.add_argument('--n-simulations', type=int, help='N simulations')
    
    # Files
    parser.add_argument('--holdings-csv', type=str, help='Holdings CSV')
    parser.add_argument('--sectors-csv', type=str, help='Sectors CSV')
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    name = args.name if args.name else ticker
    granularity = args.granularity / 100.0
    
    # Init config
    config = AnalysisConfig()
    if args.risk_aversion: config.risk_aversion = args.risk_aversion
    if args.concentration_penalty: config.concentration_penalty = args.concentration_penalty
    if args.min_recommended: config.min_recommended_allocation = args.min_recommended
    if args.min_allocation: config.min_allocation = args.min_allocation
    if args.max_allocation: config.max_allocation = args.max_allocation
    if args.benchmark: config.benchmark_ticker = args.benchmark
    if args.risk_free_rate: config.risk_free_rate = args.risk_free_rate
    if args.lookback_years: config.lookback_years = args.lookback_years
    if args.resample_freq: config.resample_freq = args.resample_freq
    if args.tech_etf: config.tech_etf_ticker = args.tech_etf
    if args.n_simulations: config.n_simulations = args.n_simulations
    
    # Load portfolio
    if args.holdings_csv:
        holdings = pl.load_holdings_csv(args.holdings_csv)
    else:
        holdings = pl.DEFAULT_TOP_HOLDINGS
        
    if args.sectors_csv:
        sectors = pl.load_sector_targets_csv(args.sectors_csv)
    else:
        sectors = pl.DEFAULT_SECTOR_TARGETS
    
    run_multi_allocation_analysis(ticker, name, granularity, config, holdings, sectors)
