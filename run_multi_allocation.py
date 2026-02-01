"""
Multi-Allocation Metrics Generator
===================================
Takes a ticker as input, finds optimal allocation using optimal_allocation.py,
then computes backtest metrics at 0.5% granularity from 0% to optimal allocation.

Usage: python run_multi_allocation.py <TICKER> [STOCK_NAME]
Example: python run_multi_allocation.py COR "Cencora Inc."
"""
import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from current directory
sys.path.insert(0, '.')

from optimal_allocation import optimize_allocation
from backtest_candidate import run_backtest

def run_multi_allocation_analysis(ticker: str, name: str = None):
    """
    Run multi-allocation analysis for a given ticker.
    
    1. Find optimal allocation using optimize_allocation()
    2. Generate allocations from 0.5% to optimal at 0.5% granularity
    3. Run backtest at each allocation level
    4. Save master CSV with all metrics
    """
    if name is None:
        name = ticker
    
    print("="*70)
    print(f"MULTI-ALLOCATION ANALYSIS: {name} ({ticker})")
    print("="*70)
    
    # Step 1: Find optimal allocation
    print("\n[STEP 1] Finding optimal allocation...")
    try:
        opt_result = optimize_allocation(ticker, name)
        optimal_alloc = opt_result.recommended_allocation
        print(f"\n   ✓ Optimal Allocation: {optimal_alloc*100:.1f}%")
    except Exception as e:
        print(f"   ✗ Error finding optimal allocation: {e}")
        return None
    
    # Step 2: Generate allocation levels at 0.5% granularity
    print("\n[STEP 2] Generating allocation levels (0.5% granularity)...")
    
    # Start from 0.5% (0.005), go up to optimal allocation (rounded up to nearest 0.5%)
    max_alloc = np.ceil(optimal_alloc * 200) / 200  # Round up to nearest 0.5%
    allocations = np.arange(0.005, max_alloc + 0.001, 0.005)  # 0.5% = 0.005
    allocations = [round(a, 4) for a in allocations if a <= max_alloc + 0.001]
    
    # Add the exact optimal allocation if not already included
    if optimal_alloc not in allocations:
        allocations.append(optimal_alloc)
        allocations = sorted(allocations)
    
    print(f"   Testing {len(allocations)} allocation levels: {allocations[0]*100:.1f}% to {allocations[-1]*100:.1f}%")
    
    # Step 3: Run backtest at each allocation level
    print("\n[STEP 3] Running backtests...")
    
    all_results = []
    for i, alloc in enumerate(allocations, 1):
        pct = alloc * 100
        print(f"   [{i}/{len(allocations)}] {pct:.1f}%...", end=" ")
        
        try:
            results = run_backtest(ticker, name, allocation=alloc, output_dir='.', show_plot=False)
            
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
    cols_order = [
        'Allocation (%)',
        'Is Optimal',
        'Annualized Return (%)',
        'Annualized Volatility (%)',
        'Sharpe Ratio',
        'Beta vs ACWI',
        'Alpha (%)',
        'VaR (95%, period)',
        'VaR (95%, annualized)',
        'Max Drawdown (%)',
        'Correlation vs ACWI',
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
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_multi_allocation.py <TICKER> [STOCK_NAME]")
        print("Example: python run_multi_allocation.py COR \"Cencora Inc.\"")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    name = sys.argv[2] if len(sys.argv) > 2 else ticker
    
    # Run analysis
    run_multi_allocation_analysis(ticker, name)
