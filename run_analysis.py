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
from portfolio_reconstruction import build_portfolio_weights
from finoracle_wrapper import FinOracleWrapper
from finoracle_utils import get_ric
import argparse

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
                 holdings: dict = None, sector_targets: dict = None, sprint1_options: dict = None):
        self.candidates = candidates
        self.output_base = output_base
        self.config = config if config else AnalysisConfig()
        self.holdings = holdings if holdings else DEFAULT_TOP_HOLDINGS
        self.sector_targets = sector_targets if sector_targets else DEFAULT_SECTOR_TARGETS
        
        # Sprint 1 options with defaults
        self.sprint1_options = sprint1_options or {
            'enable_stress_test': False,
            'stress_portfolio_value': 1000000,
            'use_ledoit_wolf': True,
            'enable_rebalancing': False,
            'rebalancing_portfolio_value': None,
            'min_trade_value': 100,
            'round_to_lots': False,
        }
        
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
        self.finoracle_results = {} # Store forecast results
        self.stress_test_results = {}
        self.stress_test_results = {}
        self.rebalancing_plans = {}
        self.reconstructed_weights = None
        self.latest_prices = None
        
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
        print(f"\n[DIR] Output directory: {self.output_dir}")
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
            self.reconstructed_weights = portfolio_weights
            
            # Download data
            tickers = list(portfolio_weights.keys()) + [self.config.benchmark_ticker]
            prices = pr.download_data(tickers, self.config)
            prices = pr.clean_data(prices)
            self.latest_prices = prices.iloc[-1]
            
            # Compute returns (returns tuple: portfolio_returns, all_returns)
            portfolio_returns, all_returns = pr.compute_portfolio_returns(prices, portfolio_weights, self.config.resample_freq)
            self.all_returns = all_returns  # Save for rebalanced stress testing
            
            # Resample benchmark
            if self.config.resample_freq == 'W':
                benchmark_prices = prices[self.config.benchmark_ticker].resample('W-FRI').last()
            else:
                benchmark_prices = prices[self.config.benchmark_ticker]
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Calculate metrics
            periods = 52 if self.config.resample_freq == 'W' else 252
            metrics = pr.calculate_risk_metrics(
                portfolio_returns, 
                benchmark_returns, 
                self.config.risk_free_rate, 
                periods,
                benchmark_name=self.config.benchmark_ticker
            )
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
                metrics.get('Alpha (%)', 0),  
                metrics.get('Information Ratio', 0)
            ]
            axes[1, 1].bar(metric_names, metric_vals, color=['steelblue', 'green', 'orange'])
            axes[1, 1].set_title('Key Metrics')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            plt.savefig(self.portfolio_dir / "portfolio_analysis_chart.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\n[OK] Portfolio reconstruction complete")
            print(f"   Annualized Return: {metrics.get('Annualized Return (%)', 0):.2f}%")
            print(f"   Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}")
            print(f"   Alpha: {metrics.get('Alpha (%)', 0):.2f}%")
            print(f"   Saved to: {self.portfolio_dir}")
            
            # Store returns for stress testing
            self.portfolio_returns = portfolio_returns
            self.benchmark_returns = benchmark_returns
            
        except Exception as e:
            print(f"[ERROR] Portfolio reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            return  # Don't run stress tests if reconstruction failed
            
        # Run stress tests if enabled (outside try/except for isolation)
        self.run_stress_tests(self.portfolio_returns, self.benchmark_returns)
    
    
    def run_stress_tests(self, portfolio_returns, benchmark_returns):
        """Run stress tests on the portfolio if enabled."""
        # Debug logging to file for reliable inspection
        with open('stress_debug.log', 'w') as f:
            f.write(f"sprint1_options: {self.sprint1_options}\n")
            f.write(f"enable_stress_test: {self.sprint1_options.get('enable_stress_test', False)}\n")
            f.write(f"portfolio_returns is None: {portfolio_returns is None}\n")
            f.write(f"benchmark_returns is None: {benchmark_returns is None}\n")
        
        print(f"DEBUG: Checking stress test options: {self.sprint1_options}")
        if not self.sprint1_options.get('enable_stress_test', False):
            print("DEBUG: Stress test disabled in options")
            return
            
        print("\n" + "="*70)
        print(">> STRESS TESTING (Sprint 1)")
        print("="*70)
        
        try:
            from stress_testing import run_all_stress_tests, display_stress_test_results
            
            portfolio_value = self.sprint1_options.get('stress_portfolio_value', 1000000)
            
            results = run_all_stress_tests(
                portfolio_returns, 
                benchmark_returns,
                portfolio_value=portfolio_value
            )
            self.stress_test_results = results
            
            # Display results
            display_stress_test_results(results)
            
            # Save to CSV
            rows = []
            for scenario_id, r in results.items():  # Iterate over dict items
                rows.append({
                    'Scenario': r.scenario_name,  # Fixed attribute name
                    'Market Drawdown (%)': f"{r.market_drawdown*100:.1f}%",
                    'Estimated Portfolio Drawdown (%)': f"{r.estimated_drawdown*100:.1f}%",
                    'Expected Loss ($)': f"${abs(r.estimated_loss_usd):,.0f}",  # Fixed attribute name
                    'Portfolio Beta': f"{r.portfolio_beta:.2f}",
                    'Recovery Time (months)': r.recovery_estimate_months,  # Fixed attribute name
                })
            pd.DataFrame(rows).to_csv(self.portfolio_dir / "stress_test_results.csv", index=False)
            
            print(f"\n[OK] Stress test results saved to: {self.portfolio_dir / 'stress_test_results.csv'}")
            
        except Exception as e:
            print(f"[ERROR] Stress testing failed: {e}")
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
            print(f"   [ERROR] Could not import optimal_allocation module: {e}")
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
                
                print(f"   [OK] {ticker}: Optimal allocation = {result.recommended_allocation*100:.1f}% ({result.recommendation_method})")
                
                # Sprint 1: Rebalancing - Generate orders to buy this candidate
                if self.sprint1_options.get('enable_rebalancing', False) and result.recommended_allocation > 0:
                     try:
                        # Debug logging to file
                        print(f"   >> Generating rebalancing orders for {ticker}...")
                        from rebalancing import calculate_rebalancing_trades, display_rebalancing_plan
                        
                        # Build target weights
                        # Logic:
                        # 1. New candidate gets 'target_alloc'
                        # 2. Existing holdings:
                        #    - If current_total + new > 100%: Scale down existing pro-rata
                        #    - If current_total + new <= 100%: Keep existing (use cash)
                        
                        target_alloc_pct = result.recommended_allocation * 100.0
                        target_weights_100 = {ticker: target_alloc_pct}
                        
                        # Extract existing weights
                        # Use reconstructed weights (FULL portfolio incl. ETFs) if available
                        if self.reconstructed_weights:
                            holding_weights = self.reconstructed_weights.copy()
                        else:
                            # Fallback to Top 10 only (should not happen if flow is right)
                            holding_weights = {}
                            for h_ticker, h_data in self.holdings.items():
                                 if isinstance(h_data, dict):
                                     holding_weights[h_ticker] = h_data.get('weight', 0)
                                 else:
                                     holding_weights[h_ticker] = h_data

                        current_total_weight = sum(holding_weights.values())
                        
                        if current_total_weight + target_alloc_pct > 100.0:
                            # Must scale down to fit (Cap at 100%)
                            available_for_existing = 100.0 - target_alloc_pct
                            scale_factor = available_for_existing / current_total_weight
                        else:
                            # Keep existing as is
                            scale_factor = 1.0

                        for h_ticker, h_weight in holding_weights.items():
                            target_weights_100[h_ticker] = h_weight * scale_factor
                            
                        # Current prices (need to include candidate price)
                        prices_map = self.latest_prices.to_dict() if self.latest_prices is not None else {}
                        
                        # Ensure candidate ticker is in prices_map
                        if ticker not in prices_map:
                            try:
                                import yfinance as yf
                                data = yf.download(ticker, period="5d", progress=False)
                                if not data.empty:
                                    if isinstance(data.columns, pd.MultiIndex):
                                        latest_price = data['Close'].iloc[-1].iloc[0] if hasattr(data['Close'].iloc[-1], 'iloc') else data['Close'].iloc[-1]
                                    else:
                                        latest_price = data['Close'].iloc[-1]
                                    prices_map[ticker] = float(latest_price)
                            except Exception as e:
                                print(f"[WARN] Failed to fetch price for {ticker}: {e}")
                        





                        # Convert holdings (weights % -> USD)
                        # holding_weights now contains FULL portfolio (Top 10 + Sector ETFs)
                        total_pv = self.sprint1_options.get('rebalancing_portfolio_value', 100000)
                        current_holdings_usd = {}
                        
                        for h_ticker, h_weight in holding_weights.items():
                             # Convert % weight to USD
                             current_holdings_usd[h_ticker] = (h_weight / 100.0) * total_pv

                        # Run rebalancing
                        plan = calculate_rebalancing_trades(
                            current_holdings=current_holdings_usd,
                            target_weights=target_weights_100,  # Passed as % (0-100)
                            current_prices=prices_map,
                            portfolio_value=total_pv,
                            min_trade_value=self.sprint1_options.get('min_trade_value', 100),
                            round_lots=self.sprint1_options.get('round_to_lots', False)
                        )

                        
                        self.rebalancing_plans[ticker] = plan
                        
                        # Save plan to CSV
                        plan_df = pd.DataFrame([vars(o) for o in plan.orders])
                        plan_df.to_csv(stock_dir / "rebalancing_orders.csv", index=False)
                        print(f"   [OK] Rebalancing plan saved. {len(plan.orders)} trades generated.")

                        # -----------------------------------------------------------------
                        # ENHANCEMENT: Stress Test on Rebalanced Portfolio
                        # -----------------------------------------------------------------
                        if self.sprint1_options.get('enable_stress_test', False) and hasattr(self, 'all_returns'):
                            try:
                                print(f"   >> Running stress test on rebalanced portfolio (Pro-Forma)...")
                                from stress_testing import run_all_stress_tests, stress_test_summary_df
                                import yfinance as yf
                                
                                # 1. Fetch Candidate History
                                start_date = pd.Timestamp.now() - pd.DateOffset(years=self.config.lookback_years)
                                cand_hist = yf.download(ticker, start=start_date, progress=False)['Close']
                                if isinstance(cand_hist, pd.DataFrame): cand_hist = cand_hist.iloc[:, 0]
                                cand_ret = cand_hist.pct_change().dropna()
                                
                                # 2. Align with existing portfolio history
                                aligned_data = self.all_returns.copy()
                                aligned_data[ticker] = cand_ret
                                aligned_data = aligned_data.dropna()
                                
                                if not aligned_data.empty:
                                    # 3. Calculate Weighted Returns (Pro-Forma)
                                    pro_forma_ret = pd.Series(0.0, index=aligned_data.index)
                                    for t_res, w_pct in target_weights_100.items():
                                        if t_res in aligned_data.columns:
                                            pro_forma_ret += aligned_data[t_res] * (w_pct / 100.0)
                                    
                                    # 4. Run Stress Test
                                    bench_ret = self.benchmark_returns.loc[aligned_data.index] if hasattr(self, 'benchmark_returns') else None
                                    
                                    if bench_ret is not None and not bench_ret.empty:
                                        pf_results = run_all_stress_tests(pro_forma_ret, bench_ret, portfolio_value=total_pv)
                                        pf_df = stress_test_summary_df(pf_results)
                                        pf_df.to_csv(stock_dir / "stress_test_rebalanced.csv")
                                        print(f"   [OK] Rebalanced stress test saved to: stress_test_rebalanced.csv")
                                    else:
                                        print(f"   [WARN] Missing benchmark data for Pro-Forma stress test.")
                                else:
                                    print(f"   [WARN] Not enough overlapping history for Pro-Forma stress test.")
                            except Exception as e:
                                print(f"   [ERROR] Rebalanced stress test failed: {e}")
                        
                     except Exception as e:
                        print(f"   [ERROR] Rebalancing generation failed: {e}")
                        with open('rebalancing_crash.txt', 'w') as f:
                             import traceback
                             traceback.print_exc(file=f)

            except Exception as e:
                print(f"   [ERROR] {ticker} optimization failed: {e}")
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
                print(f"\n   [STATS] Combined summary saved to: {self.summary_dir}")
    
    def run_backtests(self):
        """Run backtests for all candidate stocks."""
        print("\n" + "="*70)
        print("STEP 3: CANDIDATE BACKTESTS")
        print("="*70)
        
        try:
            from backtest_candidate import run_backtest
        except ImportError as e:
            print(f"   [ERROR] Could not import backtest module: {e}")
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
                print(f"   [OK] {ticker} backtest complete")
                
            except Exception as e:
                print(f"   [ERROR] {ticker} backtest failed: {e}")
                self.backtest_results[ticker] = {'error': str(e)}
    
    def run_valuations(self):
        """Run valuation engine for all candidate stocks."""
        print("\n" + "="*70)
        print("STEP 4: VALUATION ANALYSIS")
        print("="*70)
        
        try:
            from valuation_engine import ValuationEngine, RelativeValuation
        except ImportError as e:
            print(f"   [ERROR] Could not import valuation module: {e}")
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
                
                print(f"   [OK] {ticker} valuation complete")
                
            except Exception as e:
                print(f"   [ERROR] {ticker} valuation failed: {e}")
                import traceback
                traceback.print_exc()
                self.valuation_results[ticker] = {'error': str(e)}
        
        # Save summary
        try:
            summary = engine.get_summary()
            summary.to_csv(self.summary_dir / "valuation_summary.csv", index=False)
        except Exception as e:
            print(f"   [WARN] Could not save valuation summary: {e}")
    
    
    def run_forecasting(self):
        """Run FinOracle forecasting for all candidate stocks."""
        if not self.config.enable_finoracle:
            return

        print("\n" + "="*70)
        print("STEP 4.5: FINORACLE FORECASTING")
        print("="*70)
        
        wrapper = FinOracleWrapper()
        
        for candidate in self.candidates:
            ticker = candidate['ticker']
            name = candidate['name']
            
            # Convert to RIC
            ric = get_ric(ticker)
            
            try:
                # Run Wrapper
                # Note: We pass the stock-specific output dir so it creates 'finoracle/' inside it
                stock_dir = self.get_stock_dir(ticker)
                
                result = wrapper.run_forecast(
                    ticker=ticker,
                    ric=ric,
                    output_dir=stock_dir, # Wrapper will add /finoracle
                    config=self.config
                )
                
                self.finoracle_results[ticker] = result
                
                if 'error' in result:
                    print(f"   [ERROR] {ticker}: {result['error']}")
                else:
                    print(f"   [OK] {ticker}: Forecast={result['forecast_price']:.2f}, Return={result['expected_return']:.2%}")
                    
            except Exception as e:
                print(f"   [ERROR] {ticker} forecast failed: {e}")
                self.finoracle_results[ticker] = {'error': str(e)}

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
            
            # Add FinOracle results
            if ticker in self.finoracle_results and 'error' not in self.finoracle_results[ticker]:
                fo = self.finoracle_results[ticker]
                row['FinOracle Return'] = f"{fo.get('expected_return', 0):.2%}"
                row['FinOracle Price'] = f"${fo.get('forecast_price', 0):.2f}"
                row['FinOracle Horizon'] = f"{fo.get('horizon_days')}d"
            
            summary_rows.append(row)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.summary_dir / "master_summary.csv", index=False)
        
        # Create text report
        report = self._generate_text_report(summary_df)
        with open(self.summary_dir / "analysis_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n[OK] Master summary saved to: {self.summary_dir}")
        print(f"\n[STATS] ANALYSIS COMPLETE!")
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
                lines.append(f"    [WARN] Missing Data: {row['Missing Data']}")
            if 'Data Warnings' in row and row['Data Warnings']:
                lines.append(f"    [WARN] Data Issues: {row['Data Warnings']}")
        
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
            run_valuations: bool = True,
            run_forecasting: bool = False):
        """Run the full analysis pipeline."""
        
        print("\n" + "=" * 35)
        print("   MASTER ANALYSIS ORCHESTRATOR")
        print("=" * 35)
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

        if run_forecasting:
            print(f"\n[DEBUG] run_forecasting=True, config.enable_finoracle={self.config.enable_finoracle}")
            self.run_forecasting()
        else:
            print(f"\n[DEBUG] run_forecasting=False, skipping FinOracle step")
        
        self.generate_master_summary()
        self.save_config_summary()
        
        return self.output_dir

    def save_config_summary(self):
        """Save configuration parameters to text file."""
        try:
            params = self.config.to_dict()
            with open(self.output_dir / "parameters.txt", "w") as f:
                f.write("="*50 + "\n")
                f.write("ANALYSIS RUN PARAMETERS\n")
                f.write("="*50 + "\n\n")
                for k, v in sorted(params.items()):
                    f.write(f"{k}: {v}\n")
            print(f"   [CONF] Parameters saved to: parameters.txt")
        except Exception as e:
            print(f"   [WARN] Failed to save parameters: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description="ECE Portfolio Analysis Engine")
    
    # Standard flags
    parser.add_argument("--ticker", type=str, help="Run for specific ticker (adds to default list)")
    parser.add_argument("--name", type=str, help="Name for the specific ticker")
    parser.add_argument("--allocation", type=float, default=0.05, help="Allocation for specific ticker")
    
    # FinOracle flags
    parser.add_argument("--finoracle", action="store_true", help="Enable FinOracle forecasting")
    # Data Fetching
    parser.add_argument("--fo-freq", type=str, default='d', help="Data frequency: tick, 1min, 5min, 1h, d, w, m")
    parser.add_argument("--fo-days", type=int, default=None, help="Fetch last N days of data (overrides --fo-years)")
    parser.add_argument("--fo-years", type=int, default=5, help="Fetch last N years of data (default: 5)")
    parser.add_argument("--fo-start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--fo-end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--fo-skip-fetch", action="store_true", help="Reuse existing data.csv")
    # Model Configuration
    parser.add_argument("--fo-context", type=int, default=128, help="Context length L (32-1024)")
    parser.add_argument("--fo-horizon", type=int, default=16, help="Forecast horizon H (1-256)")
    parser.add_argument("--fo-optimize", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--fo-trials", type=int, default=20, help="Optuna trials for optimization")
    parser.add_argument("--fo-folds", type=int, default=3, help="CV folds for optimization")
    parser.add_argument("--fo-cpu", action="store_true", help="Force CPU (no GPU)")
    parser.add_argument("--fo-skip-inference", action="store_true", help="Skip model run (re-visualize old results)")
    
    args = parser.parse_args()
    
    # Initialize Config
    config = AnalysisConfig()
    
    # Update config with FinOracle args
    if args.finoracle:
        config.enable_finoracle = True
        config.finoracle_freq = args.fo_freq
        config.finoracle_days = args.fo_days
        config.finoracle_years = args.fo_years
        config.finoracle_start = args.fo_start
        config.finoracle_end = args.fo_end
        config.finoracle_skip_fetch = args.fo_skip_fetch
        config.finoracle_context_len = args.fo_context
        config.finoracle_horizon_len = args.fo_horizon
        config.finoracle_optimize = args.fo_optimize
        config.finoracle_trials = args.fo_trials
        config.finoracle_folds = args.fo_folds
        config.finoracle_use_gpu = not args.fo_cpu
        config.finoracle_skip_inference = args.fo_skip_inference
    
    # Build Candidate List
    candidate_stocks = [
        {'ticker': 'UNH', 'name': 'UnitedHealth Group', 'allocation': 0.05},
        {'ticker': 'TMO', 'name': 'Thermo Fisher', 'allocation': 0.03},
        {'ticker': 'V', 'name': 'Visa Inc.', 'allocation': 0.04},
    ]
    
    if args.ticker:
        candidate_stocks.append({
            'ticker': args.ticker,
            'name': args.name if args.name else args.ticker,
            'allocation': args.allocation
        })

    orchestrator = AnalysisOrchestrator(
        candidates=candidate_stocks,
        output_base=OUTPUT_BASE,
        config=config,
        holdings=DEFAULT_TOP_HOLDINGS,
        sector_targets=DEFAULT_SECTOR_TARGETS
    )
    
    output_dir = orchestrator.run(
        run_portfolio=True,
        run_optimal=True,
        run_backtests=True,
        run_valuations=True,
        run_forecasting=config.enable_finoracle
    )
    
    print(f"\n\n{'='*70}")
    print("ALL ANALYSES COMPLETE!")
    print(f"{'='*70}")
    print(f"\n[DIR] Results saved to: {output_dir}")
    
    return orchestrator


if __name__ == "__main__":
    main()

