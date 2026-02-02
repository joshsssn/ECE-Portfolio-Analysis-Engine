"""
ECE Portfolio Analysis Engine
=============================
Master Analysis Orchestrator
=============================
Single entry point to run the full analysis pipeline:
1. Portfolio Reconstruction
2. Candidate Backtesting
3. Valuation Engine (DCF + Monte Carlo + Relative)

All outputs are organized into timestamped folders.

Author: Josh E. SOUSSAN
Usage: python run_analysis.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
from config import AnalysisConfig
from portfolio_loader import DEFAULT_TOP_HOLDINGS, DEFAULT_SECTOR_TARGETS

# Output directory base
OUTPUT_BASE = Path("analysis_outputs")

# Output directory base
OUTPUT_BASE = Path("analysis_outputs")


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class AnalysisOrchestrator:
    """Master orchestrator for the full analysis pipeline."""
    
    def __init__(self, candidates: list, output_base: Path = OUTPUT_BASE, config: AnalysisConfig = None,
                 holdings: dict = None, sector_targets: dict = None):
        self.candidates = candidates
        self.output_base = output_base
        self.config = config if config else AnalysisConfig()
        self.holdings = holdings if holdings else DEFAULT_TOP_HOLDINGS
        self.sector_targets = sector_targets if sector_targets else DEFAULT_SECTOR_TARGETS
        
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_base / f"run_{self.run_timestamp}"
        
        # Shared folders (portfolio is global, summary aggregates all)
        self.portfolio_dir = self.output_dir / "0_portfolio"
        self.summary_dir = self.output_dir / "summary"
        
        # Per-stock folders (created dynamically)
        self.stock_dirs = {}
        
        # Results storage
        self.portfolio_metrics = None
        self.optimal_results = {}
        self.backtest_results = {}
        self.valuation_results = {}
        
    def get_stock_dir(self, ticker: str) -> Path:
        """Get or create the output directory for a specific stock."""
        if ticker not in self.stock_dirs:
            stock_dir = self.output_dir / ticker
            stock_dir.mkdir(parents=True, exist_ok=True)
            self.stock_dirs[ticker] = stock_dir
        return self.stock_dirs[ticker]
        
    def setup_directories(self):
        """Create base output directory structure."""
        # Only create shared folders here; stock folders created on-demand
        for d in [self.portfolio_dir, self.summary_dir]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"   Structure: [TICKER]/optimal.png, backtest.png, valuation_*.png")
    
    def run_portfolio_reconstruction(self):
        """Run portfolio reconstruction and save outputs."""
        print("\n" + "="*70)
        print("STEP 1: PORTFOLIO RECONSTRUCTION")
        print("="*70)
        
        try:
            # Import module
            import portfolio_reconstruction as pr
            
            # Build weights
            portfolio_weights = pr.build_portfolio_weights(self.holdings, self.sector_targets)
            
            # Download data
            tickers = list(portfolio_weights.keys()) + [self.config.benchmark_ticker]
            prices = pr.download_data(tickers, self.config)
            
            # Compute returns (returns tuple: portfolio_returns, all_returns)
            portfolio_returns, all_returns = pr.compute_portfolio_returns(prices, portfolio_weights, self.config.resample_freq)
            
            # Resample benchmark
            if self.config.resample_freq == 'W':
                benchmark_prices = prices[self.config.benchmark_ticker].resample('W-FRI').last()
            else:
                benchmark_prices = prices[self.config.benchmark_ticker]
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Calculate metrics
            periods = 52 if self.config.resample_freq == 'W' else 252
            metrics = pr.calculate_risk_metrics(portfolio_returns, benchmark_returns, self.config.risk_free_rate, periods)
            self.portfolio_metrics = metrics
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(self.portfolio_dir / "portfolio_risk_metrics.csv", index=False)
            
            # Save weights
            weights_df = pd.DataFrame([
                {'Ticker': k, 'Weight': v} for k, v in portfolio_weights.items()
            ])
            weights_df.to_csv(self.portfolio_dir / "portfolio_weights.csv", index=False)
            
            # Generate simple chart
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Cumulative returns
            cum_port = (1 + portfolio_returns).cumprod()
            cum_bench = (1 + benchmark_returns).cumprod()
            axes[0, 0].plot(cum_port.index, cum_port.values, label='Portfolio')
            axes[0, 0].plot(cum_bench.index, cum_bench.values, label=self.config.benchmark_ticker)
            axes[0, 0].set_title('Cumulative Returns')
            axes[0, 0].legend()
            
            # Drawdown
            dd = (cum_port - cum_port.cummax()) / cum_port.cummax()
            axes[0, 1].fill_between(dd.index, dd.values, 0, alpha=0.5, color='red')
            axes[0, 1].set_title('Portfolio Drawdown')
            
            # Rolling volatility
            rolling_window = periods # 1 year window
            rolling_vol = portfolio_returns.rolling(rolling_window).std() * np.sqrt(periods) * 100
            unit = "Week" if self.config.resample_freq == 'W' else "Day"
            axes[1, 0].plot(rolling_vol.index, rolling_vol.values)
            axes[1, 0].set_title(f'Rolling {rolling_window}-{unit} Volatility (%)')
            
            # Metrics bar chart - use correct key names from calculate_risk_metrics
            metric_names = ['Sharpe Ratio', 'Alpha (%)', 'Information Ratio']
            metric_vals = [
                metrics.get('Sharpe Ratio', 0),
                metrics.get('Alpha (annualized)', 0),  # Correct key name
                metrics.get('Information Ratio', 0)
            ]
            axes[1, 1].bar(metric_names, metric_vals, color=['steelblue', 'green', 'orange'])
            axes[1, 1].set_title('Key Metrics')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            plt.savefig(self.portfolio_dir / "portfolio_analysis_chart.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\n✅ Portfolio reconstruction complete")
            print(f"   Annualized Return: {metrics.get('Annualized Return', 0)*100:.2f}%")
            print(f"   Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}")
            print(f"   Alpha: {metrics.get('Alpha (annualized)', 0):.2f}%")
            print(f"   Saved to: {self.portfolio_dir}")
            
        except Exception as e:
            print(f"❌ Portfolio reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_optimal_allocation(self):
        """Find optimal allocation for each candidate using Sharpe + Risk Budgeting."""
        print("\n" + "="*70)
        print("STEP 2: OPTIMAL ALLOCATION FINDER")
        print("="*70)
        
        try:
            from optimal_allocation import optimize_allocation, plot_optimization
        except ImportError as e:
            print(f"❌ Could not import optimal_allocation module: {e}")
            return
        
        for candidate in self.candidates:
            ticker = candidate['ticker']
            name = candidate['name']
            
            print(f"\n--- Finding optimal allocation for {ticker} ({name}) ---")
            
            try:
                result = optimize_allocation(ticker, name, self.config, self.holdings, self.sector_targets)
                
                # Update candidate with optimal allocation
                candidate['optimal_allocation'] = result.recommended_allocation
                candidate['allocation_method'] = result.recommendation_method
                
                self.optimal_results[ticker] = result
                
                # Save chart to stock folder
                stock_dir = self.get_stock_dir(ticker)
                chart_path = str(stock_dir / "optimal.png")
                plot_optimization(result, save_path=chart_path)
                
                # Save individual summary
                summary_data = {
                    'Ticker': ticker,
                    'Name': name,
                    'Sharpe Optimal (%)': f"{result.sharpe_optimal_allocation * 100:.1f}%",
                    'Min Vol Optimal (%)': f"{result.risk_budget_optimal_allocation * 100:.1f}%",
                    'Recommended (%)': f"{result.recommended_allocation * 100:.1f}%",
                    'Method': result.recommendation_method,
                    'Original Sharpe': f"{result.original_sharpe:.3f}",
                    'New Sharpe': f"{result.new_sharpe:.3f}",
                    'Original Vol (%)': f"{result.original_volatility:.2f}%",
                    'New Vol (%)': f"{result.new_volatility:.2f}%",
                }
                pd.DataFrame([summary_data]).to_csv(
                    stock_dir / "optimal_summary.csv", index=False
                )
                
                print(f"   ✅ {ticker}: Optimal allocation = {result.recommended_allocation*100:.1f}% ({result.recommendation_method})")
                
            except Exception as e:
                print(f"   ❌ {ticker} optimization failed: {e}")
                self.optimal_results[ticker] = {'error': str(e)}
        
        # Save combined summary
        if self.optimal_results:
            rows = []
            for ticker, result in self.optimal_results.items():
                if hasattr(result, 'recommended_allocation'):
                    rows.append({
                        'Ticker': ticker,
                        'Optimal Allocation': f"{result.recommended_allocation * 100:.1f}%",
                        'Method': result.recommendation_method,
                        'Sharpe Change': f"{result.new_sharpe - result.original_sharpe:+.3f}",
                        'Vol Change': f"{result.new_volatility - result.original_volatility:+.2f}%",
                    })
            if rows:
                pd.DataFrame(rows).to_csv(self.summary_dir / "optimal_summary.csv", index=False)
                print(f"\n   📊 Combined summary saved to: {self.summary_dir}")
    
    def run_backtests(self):
        """Run backtests for all candidate stocks."""
        print("\n" + "="*70)
        print("STEP 3: CANDIDATE BACKTESTS")
        print("="*70)
        
        try:
            from backtest_candidate import run_backtest
        except ImportError as e:
            print(f"❌ Could not import backtest module: {e}")
            return
        
        for candidate in self.candidates:
            ticker = candidate['ticker']
            name = candidate['name']
            # Use optimal allocation if found, otherwise use default
            allocation = candidate.get('optimal_allocation', candidate['allocation'])
            
            # Get stock-specific output folder
            stock_dir = self.get_stock_dir(ticker)
            
            print(f"\n--- Backtesting {ticker} ({name}) at {allocation:.1%} allocation ---")
            
            try:
                result = run_backtest(
                    ticker=ticker,
                    name=name,
                    config=self.config,
                    top_holdings=self.holdings,
                    sector_targets=self.sector_targets,
                    allocation=allocation,
                    output_dir=str(stock_dir),
                    show_plot=False
                )
                
                self.backtest_results[ticker] = result
                print(f"   ✅ {ticker} backtest complete")
                
            except Exception as e:
                print(f"   ❌ {ticker} backtest failed: {e}")
                self.backtest_results[ticker] = {'error': str(e)}
    
    def run_valuations(self):
        """Run valuation engine for all candidate stocks."""
        print("\n" + "="*70)
        print("STEP 4: VALUATION ANALYSIS")
        print("="*70)
        
        try:
            from valuation_engine import ValuationEngine, RelativeValuation
        except ImportError as e:
            print(f"❌ Could not import valuation module: {e}")
            return
        
        engine = ValuationEngine(config=self.config)
        
        for candidate in self.candidates:
            ticker = candidate['ticker']
            name = candidate['name']
            
            print(f"\n--- Valuing {ticker} ({name}) ---")
            
            try:
                dcf_result, rel_result = engine.analyze_full(ticker, verbose=True)
                
                # Get financial data for this ticker to capture missing fields
                financial_data = engine.financial_data.get(ticker)
                
                self.valuation_results[ticker] = {
                    'dcf': dcf_result,
                    'relative': rel_result,
                    'missing_fields': financial_data.missing_fields if financial_data else [],
                    'data_warnings': financial_data.data_quality_warnings if financial_data else [],
                }
                
                # Get stock-specific output folder
                stock_dir = self.get_stock_dir(ticker)
                
                # Save DCF chart
                engine.plot_valuation(ticker, save_path=str(stock_dir / "valuation_dcf.png"))
                
                # Save relative chart
                if rel_result and rel_result.is_valid:
                    RelativeValuation.plot_regression(
                        rel_result, 
                        save_path=str(stock_dir / "valuation_relative.png")
                    )
                
                print(f"   ✅ {ticker} valuation complete")
                
            except Exception as e:
                print(f"   ❌ {ticker} valuation failed: {e}")
                import traceback
                traceback.print_exc()
                self.valuation_results[ticker] = {'error': str(e)}
        
        # Save summary
        try:
            summary = engine.get_summary()
            summary.to_csv(self.summary_dir / "valuation_summary.csv", index=False)
        except Exception as e:
            print(f"   ⚠️ Could not save valuation summary: {e}")
    
    def generate_master_summary(self):
        """Generate a master summary report combining all analyses."""
        print("\n" + "="*70)
        print("STEP 5: GENERATING MASTER SUMMARY")
        print("="*70)
        
        summary_rows = []
        
        for candidate in self.candidates:
            ticker = candidate['ticker']
            name = candidate['name']
            # Use optimal allocation if found
            allocation = candidate.get('optimal_allocation', candidate['allocation'])
            method = candidate.get('allocation_method', 'Default')
            
            row = {
                'Ticker': ticker,
                'Name': name,
                'Optimal Allocation': f"{allocation:.1%}",
                'Method': method,
            }
            
            # Add backtest results
            if ticker in self.backtest_results and 'error' not in self.backtest_results[ticker]:
                bt = self.backtest_results[ticker]
                row['BT: Return Impact'] = bt.get('return_change', 'N/A')
                row['BT: Vol Impact'] = bt.get('vol_change', 'N/A')
                row['BT: Sharpe Impact'] = bt.get('sharpe_change', 'N/A')
            
            # Add valuation results
            if ticker in self.valuation_results and 'error' not in self.valuation_results[ticker]:
                val = self.valuation_results[ticker]
                dcf = val.get('dcf')
                rel = val.get('relative')
                
                if dcf and dcf.is_valid:
                    row['Current Price'] = f"${dcf.current_price:.2f}"
                    row['DCF Fair Value'] = f"${dcf.mc_mean:.2f}"
                    row['DCF Margin of Safety'] = f"{dcf.margin_of_safety:.1f}%"
                    row['Win Probability'] = f"{dcf.win_probability:.0f}%"
                
                if rel and rel.is_valid:
                    row['P/E Discount'] = f"{rel.pe_discount_pct:+.1f}%"
                    row['EV/EBITDA Discount'] = f"{rel.ev_discount_pct:+.1f}%"
                
                # Add data quality info
                missing = val.get('missing_fields', [])
                warnings = val.get('data_warnings', [])
                if missing:
                    row['Missing Data'] = ', '.join(missing[:3]) + ('...' if len(missing) > 3 else '')
                if warnings:
                    row['Data Warnings'] = ', '.join(warnings[:2]) + ('...' if len(warnings) > 2 else '')
            
            summary_rows.append(row)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.summary_dir / "master_summary.csv", index=False)
        
        # Create text report
        report = self._generate_text_report(summary_df)
        with open(self.summary_dir / "analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Master summary saved to: {self.summary_dir}")
        print(f"\n📊 ANALYSIS COMPLETE!")
        print(f"   All outputs in: {self.output_dir}")
    
    def _generate_text_report(self, summary_df):
        """Generate a text report."""
        lines = [
            "=" * 70,
            "PORTFOLIO ANALYSIS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "PORTFOLIO RECONSTRUCTION",
            "-" * 40,
        ]
        
        if self.portfolio_metrics:
            for k, v in self.portfolio_metrics.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                else:
                    lines.append(f"  {k}: {v}")
        
        lines.extend([
            "",
            "CANDIDATE ANALYSIS SUMMARY",
            "-" * 40,
        ])
        
        for _, row in summary_df.iterrows():
            lines.append(f"\n  {row['Ticker']} - {row['Name']}")
            lines.append(f"    Optimal Allocation: {row.get('Optimal Allocation', 'N/A')} ({row.get('Method', 'N/A')})")
            lines.append(f"    Current Price: {row.get('Current Price', 'N/A')}")
            lines.append(f"    DCF Fair Value: {row.get('DCF Fair Value', 'N/A')}")
            lines.append(f"    Margin of Safety: {row.get('DCF Margin of Safety', 'N/A')}")
            lines.append(f"    Win Probability: {row.get('Win Probability', 'N/A')}")
            if 'BT: Sharpe Impact' in row:
                lines.append(f"    Backtest Sharpe Impact: {row.get('BT: Sharpe Impact', 'N/A')}")
            if 'Missing Data' in row and row['Missing Data']:
                lines.append(f"    ⚠️ Missing Data: {row['Missing Data']}")
            if 'Data Warnings' in row and row['Data Warnings']:
                lines.append(f"    ⚠️ Data Issues: {row['Data Warnings']}")
        
        lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def run(self, 
            run_portfolio: bool = True,
            run_optimal: bool = True,
            run_backtests: bool = True, 
            run_valuations: bool = True):
        """Run the full analysis pipeline."""
        
        print("\n" + "🚀" * 35)
        print("   MASTER ANALYSIS ORCHESTRATOR")
        print("🚀" * 35)
        print(f"\nCandidates: {[c['ticker'] for c in self.candidates]}")
        
        self.setup_directories()
        
        if run_portfolio:
            self.run_portfolio_reconstruction()
        
        if run_optimal:
            self.run_optimal_allocation()
        
        if run_backtests:
            self.run_backtests()
        
        if run_valuations:
            self.run_valuations()
        
        self.generate_master_summary()
        
        return self.output_dir


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point."""
    
    # Example Default Candidate Stocks
    candidate_stocks = [
        {'ticker': 'UNH', 'name': 'UnitedHealth Group', 'allocation': 0.05},
        {'ticker': 'TMO', 'name': 'Thermo Fisher', 'allocation': 0.03},
        {'ticker': 'V', 'name': 'Visa Inc.', 'allocation': 0.04},
    ]

    orchestrator = AnalysisOrchestrator(
        candidates=candidate_stocks,
        output_base=OUTPUT_BASE,
        config=AnalysisConfig(),
        holdings=DEFAULT_TOP_HOLDINGS,
        sector_targets=DEFAULT_SECTOR_TARGETS
    )
    
    output_dir = orchestrator.run(
        run_portfolio=True,
        run_optimal=True,
        run_backtests=True,
        run_valuations=True
    )
    
    print(f"\n\n{'='*70}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📂 Results saved to: {output_dir}")
    print("\nFolder structure:")
    print("  ├── 0_portfolio/")
    print("  │   ├── portfolio_risk_metrics.csv")
    print("  │   ├── portfolio_weights.csv")
    print("  │   └── portfolio_analysis_chart.png")
    print("  ├── [TICKER]/")
    print("  │   ├── optimal.png")
    print("  │   ├── backtest.png")
    print("  │   ├── valuation_dcf.png")
    print("  │   └── ...")
    print("  └── summary/")
    print("      ├── master_summary.csv")
    print("      └── analysis_report.txt")
    
    return orchestrator


if __name__ == "__main__":
    orchestrator = main()

