"""
ECE Portfolio Analysis Engine
=============================
Run Analysis from Screener CSV
==============================
Reads a screener CSV file and runs the full analysis pipeline
on all (or selected) stocks from the screener.

Author: Josh E. SOUSSAN
Usage: python run_from_screener.py [--csv path/to/screener.csv] [--top N] [--multi-alloc]
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

from config import AnalysisConfig
from portfolio_loader import load_holdings_csv, load_sector_targets_csv, DEFAULT_TOP_HOLDINGS, DEFAULT_SECTOR_TARGETS


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default screener file
DEFAULT_SCREENER = 'screener-results.csv'

# Maximum number of stocks to analyze (to avoid long runtimes)
MAX_STOCKS = 50

# Default allocation for each candidate (will be overridden by optimal allocation finder)
DEFAULT_ALLOCATION = 0.05

# Run settings
RUN_PORTFOLIO_RECONSTRUCTION = True
RUN_OPTIMAL_ALLOCATION = True
RUN_BACKTESTS = True
RUN_VALUATION = True
RUN_MULTI_ALLOCATION = False  # Off by default (expensive)

# Multi-allocation granularity
MULTI_ALLOC_STEP = 0.005  # 0.5%


# =============================================================================
# CSV PARSING
# =============================================================================

def load_screener(csv_path: str) -> pd.DataFrame:
    """Load and validate screener CSV."""
    print(f"\n📂 Loading screener: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Required columns
    required = ['symbol', 'companyName']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"   Found {len(df)} stocks in screener")
    
    # Display preview
    print(f"\n   Preview:")
    for i, row in df.head(5).iterrows():
        print(f"   {row['symbol']:6s} | {row['companyName'][:40]}")
    if len(df) > 5:
        print(f"   ... and {len(df) - 5} more")
    
    return df


def screener_to_candidates(df: pd.DataFrame, top_n: int = None) -> list:
    """
    Convert screener DataFrame to candidates list format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Screener data
    top_n : int, optional
        Limit to top N stocks (by market cap or order in file)
    
    Returns
    -------
    list
        List of dicts with ticker, name, allocation
    """
    if top_n:
        df = df.head(top_n)
    
    candidates = []
    for _, row in df.iterrows():
        ticker = row['symbol']
        name = row.get('companyName', ticker)
        
        # Clean up name (remove special chars, truncate)
        name = str(name).replace(',', '').replace('"', '')[:40]
        
        candidates.append({
            'ticker': ticker,
            'name': name,
            'allocation': DEFAULT_ALLOCATION,
        })
    
    return candidates


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_multi_allocation_for_screener(candidates: list, output_dir: Path, 
                                      config: AnalysisConfig, 
                                      holdings: dict, sector_targets: dict,
                                      granularity: float = 0.005):
    """
    Run multi-allocation analysis for all stocks in the screener.
    """
    import numpy as np
    from optimal_allocation import optimize_allocation
    from backtest_candidate import run_backtest
    
    gran_pct = granularity * 100
    print("\n" + "="*70)
    print(f"MULTI-ALLOCATION ANALYSIS ({gran_pct:.1f}% GRANULARITY)")
    print("="*70)
    
    for candidate in candidates:
        ticker = candidate['ticker']
        name = candidate['name']
        stock_dir = output_dir / ticker
        stock_dir.mkdir(exist_ok=True)
        
        print(f"\n📊 Processing {ticker}: {name}")
        
        try:
            # Step 1: Find optimal allocation
            print(f"   [1/3] Finding optimal allocation...")
            opt_result = optimize_allocation(ticker, name, config, holdings, sector_targets)
            optimal_alloc = opt_result.recommended_allocation
            print(f"         Optimal: {optimal_alloc*100:.1f}%")
            
            # Step 2: Generate allocation levels
            max_alloc = np.ceil(optimal_alloc / granularity) * granularity
            allocations = np.arange(granularity, max_alloc + 0.0001, granularity)
            allocations = [round(a, 4) for a in allocations if a <= max_alloc + 0.0001]
            if optimal_alloc not in allocations:
                allocations.append(optimal_alloc)
                allocations = sorted(allocations)
            
            print(f"   [2/3] Running {len(allocations)} backtests ({allocations[0]*100:.1f}% to {allocations[-1]*100:.1f}%)...")
            
            # Step 3: Run backtests
            all_results = []
            for alloc in allocations:
                results = run_backtest(ticker, name, 
                                       config=config, top_holdings=holdings, sector_targets=sector_targets,
                                       allocation=alloc, 
                                       output_dir=str(stock_dir), show_plot=False)
                
                row = {'Allocation (%)': alloc * 100}
                row['Is Optimal'] = 'Yes' if abs(alloc - optimal_alloc) < 0.001 else 'No'
                
                for key, val in results['metrics_after'].items():
                    row[key] = val
                
                row['Return Change (%)'] = results['metrics_after']['Annualized Return (%)'] - results['metrics_before']['Annualized Return (%)']
                row['Vol Change (%)'] = results['metrics_after']['Annualized Volatility (%)'] - results['metrics_before']['Annualized Volatility (%)']
                row['Sharpe Change'] = results['metrics_after']['Sharpe Ratio'] - results['metrics_before']['Sharpe Ratio']
                row['VaR Ann. Change (%)'] = results['metrics_after']['VaR (95%, annualized)'] - results['metrics_before']['VaR (95%, annualized)']
                row['Max DD Change (%)'] = results['metrics_after']['Max Drawdown (%)'] - results['metrics_before']['Max Drawdown (%)']
                
                all_results.append(row)
            
            # Save to ticker folder
            master_df = pd.DataFrame(all_results)
            
            # Dynamic column ordering based on actual benchmark ticker
            benchmark = config.benchmark_ticker
            cols_order = [
                'Allocation (%)', 'Is Optimal', 'Annualized Return (%)', 'Annualized Volatility (%)',
                'Sharpe Ratio', f'Beta vs {benchmark}', 'Alpha (%)', 'VaR (95%, period)', 'VaR (95%, annualized)',
                'Max Drawdown (%)', f'Correlation vs {benchmark}', 'Tracking Error (%)', 'Information Ratio',
                'Return Change (%)', 'Vol Change (%)', 'Sharpe Change', 'VaR Ann. Change (%)', 'Max DD Change (%)'
            ]
            cols_order = [c for c in cols_order if c in master_df.columns]
            master_df = master_df[cols_order]
            
            csv_path = stock_dir / f'{ticker}_multi_allocation.csv'
            master_df.to_csv(csv_path, index=False)
            print(f"   [3/3] ✓ Saved: {csv_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("MULTI-ALLOCATION ANALYSIS COMPLETE")
    print("="*70)


def run_from_screener(csv_path: str, 
                      config: AnalysisConfig,
                      holdings: dict,
                      sector_targets: dict,
                      top_n: int = None,
                      run_portfolio: bool = True,
                      run_optimal: bool = True,
                      run_backtests: bool = True,
                      run_valuations: bool = True,
                      multi_alloc_granularity: float = None):
    """
    Main function: load screener CSV and run full analysis.
    """
    from run_analysis import AnalysisOrchestrator, OUTPUT_BASE
    
    print("\n" + "🔍" * 35)
    print("   SCREENER-BASED ANALYSIS")
    print("🔍" * 35)
    
    # Load screener
    df = load_screener(csv_path)
    
    # Convert to candidates
    if top_n:
        print(f"\n   Limiting to top {top_n} stocks")
    candidates = screener_to_candidates(df, top_n)
    
    print(f"\n   Candidates to analyze: {len(candidates)}")
    for c in candidates:
        print(f"   • {c['ticker']}: {c['name']}")
    
    # Create orchestrator with screener candidates
    orchestrator = AnalysisOrchestrator(
        candidates=candidates,
        output_base=OUTPUT_BASE,
        config=config,
        holdings=holdings,
        sector_targets=sector_targets
    )
    
    # Run analysis
    print("\n" + "="*70)
    print("STARTING ANALYSIS PIPELINE")
    print("="*70)
    
    output_dir = orchestrator.run(
        run_portfolio=run_portfolio,
        run_optimal=run_optimal,
        run_backtests=run_backtests,
        run_valuations=run_valuations
    )
    
    # Run multi-allocation analysis if requested
    if multi_alloc_granularity is not None:
        run_multi_allocation_for_screener(
            candidates, output_dir, 
            config, holdings, sector_targets,
            multi_alloc_granularity
        )
    
    # Copy screener to output for reference
    screener_copy_path = output_dir / 'input_screener.csv'
    df.to_csv(screener_copy_path, index=False)
    print(f"\n📋 Input screener copied to: {screener_copy_path}")
    
    print(f"\n\n{'='*70}")
    print("SCREENER ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📂 Results saved to: {output_dir}")
    
    return orchestrator, output_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run portfolio analysis from screener CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--csv', '-c', default=DEFAULT_SCREENER, help=f'Path to screener CSV file (default: {DEFAULT_SCREENER})')
    parser.add_argument('--top', '-n', type=int, default=None, help=f'Analyze only top N stocks')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze ALL stocks')
    
    # Run flags
    parser.add_argument('--skip-portfolio', action='store_true', help='Skip portfolio reconstruction')
    parser.add_argument('--skip-optimal', action='store_true', help='Skip optimal allocation step')
    parser.add_argument('--skip-backtest', action='store_true', help='Skip backtest step')
    parser.add_argument('--skip-valuation', action='store_true', help='Skip valuation step')
    parser.add_argument('--only-valuation', action='store_true', help='Only run valuation')
    parser.add_argument('--multi-alloc', '-m', type=float, nargs='?', const=0.5, default=None, help='Run multi-allocation analysis with granularity')
    
    # Configuration - Dynamic Parameters
    parser.add_argument('--risk-aversion', type=float, help='Risk aversion (gamma)')
    parser.add_argument('--concentration-penalty', type=float, help='Concentration penalty')
    parser.add_argument('--min-recommended', type=float, help='Min recommended allocation')
    parser.add_argument('--min-allocation', type=float, help='Min allocation range')
    parser.add_argument('--max-allocation', type=float, help='Max allocation range')
    parser.add_argument('--benchmark', type=str, help='Benchmark ticker')
    parser.add_argument('--risk-free-rate', type=float, help='Risk free rate (e.g. 0.045 for 4.5%)')
    parser.add_argument('--lookback-years', type=int, help='Lookback years')
    parser.add_argument('--resample-freq', type=str, choices=['D', 'W', 'M'], help='Resample frequency')
    parser.add_argument('--correlation-ticker', type=str, dest='tech_etf', help='Correlation Ticker (e.g. IXN)')
    parser.add_argument('--n-simulations', type=int, help='Number of Monte Carlo simulations')
    
    # File inputs
    parser.add_argument('--holdings-csv', type=str, help='Path to holdings CSV')
    parser.add_argument('--sectors-csv', type=str, help='Path to sector targets CSV')
    
    args = parser.parse_args()
    
    # 1. Initialize Config (Default + Overrides)
    config = AnalysisConfig()
    if args.risk_aversion is not None: config.risk_aversion = args.risk_aversion
    if args.concentration_penalty is not None: config.concentration_penalty = args.concentration_penalty
    if args.min_recommended is not None: config.min_recommended_allocation = args.min_recommended
    if args.min_allocation is not None: config.min_allocation = args.min_allocation
    if args.max_allocation is not None: config.max_allocation = args.max_allocation
    if args.benchmark: config.benchmark_ticker = args.benchmark
    if args.risk_free_rate is not None: config.risk_free_rate = args.risk_free_rate
    if args.lookback_years: config.lookback_years = args.lookback_years
    if args.resample_freq: config.resample_freq = args.resample_freq
    if args.tech_etf: config.tech_etf_ticker = args.tech_etf
    if args.n_simulations: config.n_simulations = args.n_simulations
    
    # 2. Load Portfolio Data
    if args.holdings_csv:
        holdings = load_holdings_csv(args.holdings_csv)
    else:
        holdings = DEFAULT_TOP_HOLDINGS
        
    if args.sectors_csv:
        sector_targets = load_sector_targets_csv(args.sectors_csv)
    else:
        sector_targets = DEFAULT_SECTOR_TARGETS
    
    # 3. Determine run flags
    if args.only_valuation:
        run_portfolio = False
        run_optimal = False
        run_backtests = False
        run_valuations = True
    else:
        run_portfolio = not args.skip_portfolio
        run_optimal = not args.skip_optimal
        run_backtests = not args.skip_backtest
        run_valuations = not args.skip_valuation
    
    # 4. Stocks limit
    if args.all:
        top_n = None
        print(f"\n⚠️  WARNING: Processing ALL stocks.")
    elif args.top is not None:
        top_n = args.top
    else:
        top_n = MAX_STOCKS
    
    # 5. Multi-alloc
    multi_alloc_granularity = None
    if args.multi_alloc is not None:
        multi_alloc_granularity = args.multi_alloc / 100
        print(f"\n📊 Multi-allocation analysis enabled: {args.multi_alloc}% granularity")
    
    # 6. Run
    orchestrator, output_dir = run_from_screener(
        csv_path=args.csv,
        config=config,
        holdings=holdings,
        sector_targets=sector_targets,
        top_n=top_n,
        run_portfolio=run_portfolio,
        run_optimal=run_optimal,
        run_backtests=run_backtests,
        run_valuations=run_valuations,
        multi_alloc_granularity=multi_alloc_granularity
    )
    
    return orchestrator


if __name__ == "__main__":
    orchestrator = main()
