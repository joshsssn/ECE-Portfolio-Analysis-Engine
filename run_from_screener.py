"""
ECE Portfolio Analysis Engine
=============================
Run Analysis from Screener CSV
==============================
Reads a screener CSV file and runs the full analysis pipeline
on all (or selected) stocks from the screener.

Author: Josh E. SOUSSAN
Usage: python run_from_screener.py [--csv path/to/screener.csv] [--top N]
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys


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

def run_from_screener(csv_path: str, 
                      top_n: int = None,
                      run_portfolio: bool = True,
                      run_optimal: bool = True,
                      run_backtests: bool = True,
                      run_valuations: bool = True):
    """
    Main function: load screener CSV and run full analysis.
    
    Parameters
    ----------
    csv_path : str
        Path to screener CSV file
    top_n : int
        Limit to top N stocks
    run_portfolio : bool
        Run portfolio reconstruction
    run_optimal : bool
        Run optimal allocation finder
    run_backtests : bool
        Run backtests
    run_valuations : bool
        Run valuation engine
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
        output_base=OUTPUT_BASE
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_from_screener.py
  python run_from_screener.py --csv my_screener.csv
  python run_from_screener.py --top 5
  python run_from_screener.py --csv results.csv --top 3 --skip-portfolio
        """
    )
    
    parser.add_argument(
        '--csv', '-c',
        default=DEFAULT_SCREENER,
        help=f'Path to screener CSV file (default: {DEFAULT_SCREENER})'
    )
    
    parser.add_argument(
        '--top', '-n',
        type=int,
        default=None,
        help=f'Analyze only top N stocks (default: {MAX_STOCKS} unless --all is used)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Analyze ALL stocks in the CSV (⚠️ can take hours for large files)'
    )
    
    parser.add_argument(
        '--skip-portfolio',
        action='store_true',
        help='Skip portfolio reconstruction step'
    )
    
    parser.add_argument(
        '--skip-optimal',
        action='store_true',
        help='Skip optimal allocation step'
    )
    
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip backtest step'
    )
    
    parser.add_argument(
        '--skip-valuation',
        action='store_true',
        help='Skip valuation step'
    )
    
    parser.add_argument(
        '--only-valuation',
        action='store_true',
        help='Only run valuation (skip portfolio, optimal, backtest)'
    )
    
    args = parser.parse_args()
    
    # Handle shortcuts
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
    
    # Determine how many stocks to analyze
    if args.all:
        top_n = None  # No limit
        print(f"\n⚠️  WARNING: Processing ALL stocks. This may take a long time!")
    elif args.top is not None:
        top_n = args.top
    else:
        top_n = MAX_STOCKS  # Default
    
    # Run
    orchestrator, output_dir = run_from_screener(
        csv_path=args.csv,
        top_n=top_n,
        run_portfolio=run_portfolio,
        run_optimal=run_optimal,
        run_backtests=run_backtests,
        run_valuations=run_valuations
    )
    
    return orchestrator


if __name__ == "__main__":
    orchestrator = main()
