"""
ECE Portfolio Analysis Engine
=============================
Valuation Engine - State of the Art Stock Valuation
====================================================
Comprehensive valuation tool using DCF + Monte Carlo simulation.
Identifies stocks trading below intrinsic value with probabilistic analysis.

Author: Josh E. SOUSSAN
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Risk-free rate (10Y Treasury proxy)
RISK_FREE_RATE = 0.045  # 4.5%

# Equity Risk Premium
EQUITY_RISK_PREMIUM = 0.05  # 5.0%

# Projection period
PROJECTION_YEARS = 10

# Terminal growth rate (base)
TERMINAL_GROWTH_BASE = 0.025  # 2.5%

# Monte Carlo simulations
N_SIMULATIONS = 10000

# Growth rate cap
MAX_GROWTH_RATE = 0.10  # 10% cap

# Sanity bounds for DCF output (relative to current price)
DCF_UPPER_BOUND_MULTIPLIER = 5.0  # Flag if DCF > 5x current price
DCF_LOWER_BOUND_MULTIPLIER = 0.10  # Flag if DCF < 10% of current price


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FinancialData:
    """Container for fetched financial data."""
    ticker: str
    current_price: float = 0.0
    shares_outstanding: float = 0.0
    market_cap: float = 0.0
    
    # Income Statement
    revenue: float = 0.0
    revenue_growth_3y: float = 0.0
    revenue_growth_5y: float = 0.0
    ebit: float = 0.0
    ebit_margin: float = 0.0
    interest_expense: float = 0.0
    
    # Cash Flow
    free_cash_flow: float = 0.0
    fcf_margin: float = 0.0
    
    # Balance Sheet
    total_debt: float = 0.0
    cash: float = 0.0
    
    # Risk
    beta: float = 1.0
    
    # Calculated
    cost_of_equity: float = 0.0
    cost_of_debt: float = 0.0
    wacc: float = 0.0
    
    # Validation
    is_valid: bool = True
    error_message: str = ""
    
    # Data quality tracking - list of fields that were missing/defaulted
    missing_fields: list = field(default_factory=list)
    data_quality_warnings: list = field(default_factory=list)


@dataclass
class ValuationResult:
    """Container for valuation results."""
    ticker: str
    current_price: float
    
    # DCF Base Case
    dcf_value: float = 0.0
    dcf_per_share: float = 0.0
    
    # Monte Carlo Results
    mc_mean: float = 0.0
    mc_median: float = 0.0
    mc_std: float = 0.0
    mc_p10: float = 0.0  # Pessimistic
    mc_p25: float = 0.0
    mc_p75: float = 0.0
    mc_p90: float = 0.0  # Optimistic
    
    # Key Metrics
    margin_of_safety: float = 0.0
    win_probability: float = 0.0
    
    # Sensitivity Matrix
    sensitivity_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Simulation data for plotting
    simulation_values: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Relative Valuation Results
    relative_pe_implied: float = 0.0
    relative_ev_ebitda_implied: float = 0.0
    relative_pe_discount: float = 0.0  # % below regression line
    relative_ev_discount: float = 0.0
    
    is_valid: bool = True
    error_message: str = ""


# =============================================================================
# DATA FETCHING MODULE
# =============================================================================

class DataFetcher:
    """Fetches and processes financial data from yfinance."""
    
    @staticmethod
    def fetch(ticker: str) -> FinancialData:
        """Fetch all required financial data for a ticker."""
        data = FinancialData(ticker=ticker)
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic Info - track if missing
            data.current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            if data.current_price == 0:
                data.missing_fields.append('current_price')
            
            data.shares_outstanding = info.get('sharesOutstanding', 0)
            if data.shares_outstanding == 0:
                data.missing_fields.append('shares_outstanding')
            
            data.market_cap = info.get('marketCap', 0)
            if data.market_cap == 0:
                data.missing_fields.append('market_cap')
            
            data.beta = info.get('beta', 1.0) or 1.0
            if info.get('beta') is None:
                data.missing_fields.append('beta (defaulted to 1.0)')
            
            if data.current_price == 0:
                # Try to get from history
                hist = stock.history(period='5d')
                if not hist.empty:
                    data.current_price = hist['Close'].iloc[-1]
                    data.data_quality_warnings.append('price from history fallback')
                    data.missing_fields.remove('current_price') if 'current_price' in data.missing_fields else None
            
            # Financials
            try:
                income = stock.financials
                if income.empty:
                    data.missing_fields.append('income_statement')
                    data.data_quality_warnings.append('No income statement data available')
                else:
                    # Revenue
                    if 'Total Revenue' in income.index:
                        revenues = income.loc['Total Revenue'].dropna().values
                        if len(revenues) > 0:
                            data.revenue = revenues[0]
                            
                            # Calculate growth from annual data
                            if len(revenues) >= 4:
                                data.revenue_growth_3y = DataFetcher._calculate_cagr(
                                    revenues[-1], revenues[0], min(3, len(revenues)-1)
                                )
                            
                            if len(revenues) >= 5:
                                data.revenue_growth_5y = DataFetcher._calculate_cagr(
                                    revenues[-1], revenues[0], min(5, len(revenues)-1)
                                )
                            else:
                                # FALLBACK 1: Try quarterly data for 5Y growth
                                try:
                                    quarterly = stock.quarterly_financials
                                    if not quarterly.empty and 'Total Revenue' in quarterly.index:
                                        q_revenues = quarterly.loc['Total Revenue'].dropna().values
                                        # Need ~20 quarters for 5Y
                                        if len(q_revenues) >= 16:  # At least 4Y of quarters
                                            years_span = len(q_revenues) / 4
                                            data.revenue_growth_5y = DataFetcher._calculate_cagr(
                                                q_revenues[-1], q_revenues[0], years_span
                                            )
                                            data.data_quality_warnings.append(f'5Y growth from {len(q_revenues)} quarters')
                                except Exception:
                                    pass
                            
                            # FALLBACK 2: Use yfinance info fields
                            if data.revenue_growth_5y == 0:
                                # Try revenueGrowth from info (this is typically trailing)
                                yf_growth = info.get('revenueGrowth')
                                if yf_growth and yf_growth > 0:
                                    data.revenue_growth_5y = yf_growth
                                    data.data_quality_warnings.append('5Y growth from yfinance revenueGrowth')
                            
                            # FALLBACK 3: Use 3Y growth as proxy for 5Y
                            if data.revenue_growth_5y == 0 and data.revenue_growth_3y > 0:
                                data.revenue_growth_5y = data.revenue_growth_3y * 0.9  # Slightly dampen for conservatism
                                data.data_quality_warnings.append('5Y growth estimated from 3Y (0.9x)')
                            
                            # If still missing after all fallbacks, flag it
                            if data.revenue_growth_5y == 0:
                                data.missing_fields.append('revenue_growth_5y (all fallbacks failed)')
                            if data.revenue_growth_3y == 0:
                                data.missing_fields.append('revenue_growth_3y (insufficient history)')
                        else:
                            data.missing_fields.append('revenue')
                    else:
                        data.missing_fields.append('revenue')
                    
                    # EBIT
                    if 'EBIT' in income.index:
                        ebit_vals = income.loc['EBIT'].dropna().values
                        data.ebit = ebit_vals[0] if len(ebit_vals) > 0 else 0
                    elif 'Operating Income' in income.index:
                        oi_vals = income.loc['Operating Income'].dropna().values
                        data.ebit = oi_vals[0] if len(oi_vals) > 0 else 0
                        data.data_quality_warnings.append('Using Operating Income as EBIT proxy')
                    else:
                        data.missing_fields.append('ebit')
                    
                    if data.ebit < 0:
                        data.data_quality_warnings.append(f'Negative EBIT: {data.ebit/1e9:.2f}B')
                    
                    if data.revenue > 0:
                        data.ebit_margin = data.ebit / data.revenue
                    
                    # Interest Expense
                    if 'Interest Expense' in income.index:
                        ie = income.loc['Interest Expense'].dropna().values
                        data.interest_expense = abs(ie[0]) if len(ie) > 0 else 0
                    else:
                        data.missing_fields.append('interest_expense')
            except Exception as e:
                print(f"  Warning: Could not fetch income statement for {ticker}: {e}")
                data.data_quality_warnings.append(f'Income statement error: {str(e)[:50]}')
            
            # Cash Flow
            try:
                cf = stock.cashflow
                if cf.empty:
                    data.missing_fields.append('cash_flow_statement')
                else:
                    if 'Free Cash Flow' in cf.index:
                        fcf_values = cf.loc['Free Cash Flow'].dropna().values
                        data.free_cash_flow = fcf_values[0] if len(fcf_values) > 0 else 0
                    else:
                        # Calculate FCF = Operating CF - CapEx
                        if 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
                            ocf = cf.loc['Operating Cash Flow'].dropna().values
                            capex = cf.loc['Capital Expenditure'].dropna().values
                            if len(ocf) > 0 and len(capex) > 0:
                                data.free_cash_flow = ocf[0] + capex[0]  # CapEx is negative
                                data.data_quality_warnings.append('FCF calculated from OCF - CapEx')
                            else:
                                data.missing_fields.append('free_cash_flow')
                        else:
                            data.missing_fields.append('free_cash_flow')
                    
                    if data.free_cash_flow < 0:
                        data.data_quality_warnings.append(f'Negative FCF: {data.free_cash_flow/1e9:.2f}B')
                    
                    if data.revenue > 0:
                        data.fcf_margin = data.free_cash_flow / data.revenue
            except Exception as e:
                print(f"  Warning: Could not fetch cash flow for {ticker}: {e}")
                data.data_quality_warnings.append(f'Cash flow error: {str(e)[:50]}')
            
            # Balance Sheet
            try:
                bs = stock.balance_sheet
                if bs.empty:
                    data.missing_fields.append('balance_sheet')
                else:
                    if 'Total Debt' in bs.index:
                        debt_values = bs.loc['Total Debt'].dropna().values
                        data.total_debt = debt_values[0] if len(debt_values) > 0 else 0
                    elif 'Long Term Debt' in bs.index:
                        debt_values = bs.loc['Long Term Debt'].dropna().values
                        data.total_debt = debt_values[0] if len(debt_values) > 0 else 0
                        data.data_quality_warnings.append('Using Long Term Debt only (no Total Debt)')
                    else:
                        data.missing_fields.append('total_debt')
                    
                    if 'Cash And Cash Equivalents' in bs.index:
                        cash_values = bs.loc['Cash And Cash Equivalents'].dropna().values
                        data.cash = cash_values[0] if len(cash_values) > 0 else 0
                    else:
                        data.missing_fields.append('cash')
            except Exception as e:
                print(f"  Warning: Could not fetch balance sheet for {ticker}: {e}")
                data.data_quality_warnings.append(f'Balance sheet error: {str(e)[:50]}')
            
            # Calculate WACC
            data = DataFetcher._calculate_wacc(data)
            
            # Final validation
            if data.current_price <= 0 or data.shares_outstanding <= 0:
                data.is_valid = False
                data.error_message = f"Missing critical data: {', '.join(data.missing_fields[:3])}"
            elif data.free_cash_flow <= 0 and data.revenue <= 0:
                data.is_valid = False
                data.error_message = "No positive FCF or Revenue data"
            
            # Summary warning if many fields missing
            if len(data.missing_fields) > 5:
                data.data_quality_warnings.append(f'{len(data.missing_fields)} data fields missing')
                
        except Exception as e:
            data.is_valid = False
            data.error_message = str(e)
        
        return data
    
    @staticmethod
    def _calculate_cagr(start_value: float, end_value: float, years: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return 0.0
        return (end_value / start_value) ** (1 / years) - 1
    
    @staticmethod
    def _calculate_wacc(data: FinancialData) -> FinancialData:
        """Calculate Weighted Average Cost of Capital."""
        # Cost of Equity (CAPM)
        data.cost_of_equity = RISK_FREE_RATE + data.beta * EQUITY_RISK_PREMIUM
        
        # Cost of Debt
        if data.total_debt > 0 and data.interest_expense > 0:
            data.cost_of_debt = min(data.interest_expense / data.total_debt, 0.15)  # Cap at 15%
        else:
            data.cost_of_debt = RISK_FREE_RATE + 0.02  # Default spread
        
        # After-tax cost of debt (assume 25% tax rate)
        after_tax_cost_of_debt = data.cost_of_debt * (1 - 0.25)
        
        # Weights
        total_capital = data.market_cap + data.total_debt
        if total_capital > 0:
            equity_weight = data.market_cap / total_capital
            debt_weight = data.total_debt / total_capital
        else:
            equity_weight = 1.0
            debt_weight = 0.0
        
        # WACC
        data.wacc = equity_weight * data.cost_of_equity + debt_weight * after_tax_cost_of_debt
        
        # Sanity check
        data.wacc = max(0.05, min(data.wacc, 0.20))  # Between 5% and 20%
        
        return data


# =============================================================================
# DCF VALUATION MODULE
# =============================================================================

class DCFValuation:
    """Discounted Cash Flow valuation engine."""
    
    @staticmethod
    def calculate(data: FinancialData, 
                  growth_override: Optional[float] = None,
                  wacc_override: Optional[float] = None,
                  terminal_growth_override: Optional[float] = None,
                  fcf_override: Optional[float] = None) -> float:
        """
        Calculate DCF intrinsic value.
        
        Returns enterprise value.
        
        Parameters:
        - fcf_override: If provided, use this as base FCF (for Monte Carlo margin simulation)
        """
        # Base FCF - use override if provided (for Monte Carlo)
        if fcf_override is not None and fcf_override > 0:
            base_fcf = fcf_override
        elif data.free_cash_flow > 0:
            base_fcf = data.free_cash_flow
        elif data.ebit > 0:
            # Estimate FCF from EBIT (after-tax approximation)
            base_fcf = data.ebit * 0.75  # Rough proxy
        else:
            # No positive FCF or EBIT - cannot value
            return 0.0
        
        # Growth rate - use weighted average instead of max for more conservative estimate
        if growth_override is not None:
            growth = growth_override
        else:
            # Use weighted average of 3Y and 5Y growth (favor recent)
            # Handle cases where one might be 0 or negative
            g3y = max(data.revenue_growth_3y, 0.0)
            g5y = max(data.revenue_growth_5y, 0.0)
            
            if g3y > 0 and g5y > 0:
                # Weighted average: 60% weight on 3Y (more recent)
                hist_growth = 0.6 * g3y + 0.4 * g5y
            elif g3y > 0:
                hist_growth = g3y
            elif g5y > 0:
                hist_growth = g5y
            else:
                hist_growth = 0.02  # Default floor
            
            growth = min(hist_growth, MAX_GROWTH_RATE)
            growth = max(growth, 0.02)  # Floor at 2%
        
        # WACC
        wacc = wacc_override if wacc_override is not None else data.wacc
        
        # Terminal growth
        terminal_growth = terminal_growth_override if terminal_growth_override is not None else TERMINAL_GROWTH_BASE
        
        # Project FCF with decaying growth
        projected_fcf = []
        current_fcf = base_fcf
        
        for year in range(1, PROJECTION_YEARS + 1):
            # Decay growth rate over time
            year_growth = growth * (1 - (year - 1) / (PROJECTION_YEARS * 2))
            year_growth = max(year_growth, terminal_growth)
            
            current_fcf = current_fcf * (1 + year_growth)
            projected_fcf.append(current_fcf)
        
        # Discount projected FCF
        pv_fcf = sum([
            fcf / ((1 + wacc) ** (i + 1))
            for i, fcf in enumerate(projected_fcf)
        ])
        
        # Terminal Value (Gordon Growth)
        # GUARD: Ensure wacc > terminal_growth to prevent division by zero or negative
        if wacc <= terminal_growth:
            # Invalid: terminal growth >= WACC makes Gordon Growth formula undefined
            # Cap terminal_growth to maintain a minimum spread
            terminal_growth = wacc - 0.01  # Minimum 1% spread
        
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** PROJECTION_YEARS)
        
        # Enterprise Value
        enterprise_value = pv_fcf + pv_terminal
        
        return enterprise_value
    
    @staticmethod
    def to_equity_value(enterprise_value: float, data: FinancialData, 
                        current_price: Optional[float] = None) -> float:
        """
        Convert enterprise value to equity value per share.
        
        Parameters:
        - current_price: If provided, used for sanity checking (optional)
        
        Returns:
        - Per-share equity value, or 0.0 if invalid
        """
        # Handle zero or negative enterprise value
        if enterprise_value <= 0:
            return 0.0
        
        equity_value = enterprise_value - data.total_debt + data.cash
        
        # If equity value is negative (debt exceeds EV + cash), return 0
        if equity_value <= 0:
            return 0.0
        
        if data.shares_outstanding > 0:
            per_share = equity_value / data.shares_outstanding
            
            # Sanity check if current price provided
            if current_price is not None and current_price > 0:
                # Cap extreme values to prevent unrealistic outputs
                upper_bound = current_price * DCF_UPPER_BOUND_MULTIPLIER
                lower_bound = current_price * DCF_LOWER_BOUND_MULTIPLIER
                
                if per_share > upper_bound:
                    # Very high valuation - cap it but still return a value
                    per_share = min(per_share, upper_bound)
                # Note: we don't raise the floor, just return 0 if truly invalid
            
            return per_share
        return 0.0


# =============================================================================
# MONTE CARLO SIMULATION MODULE
# =============================================================================

class MonteCarloValuation:
    """Monte Carlo simulation for probabilistic valuation."""
    
    @staticmethod
    def simulate(data: FinancialData, n_simulations: int = N_SIMULATIONS) -> np.ndarray:
        """
        Run Monte Carlo simulation to generate distribution of fair values.
        
        Randomizes:
        - Revenue Growth ~ Normal(Base, 2%)
        - FCF Margin ~ Normal(Base, 1%) - NOW ACTUALLY USED!
        - WACC ~ Normal(Base, 0.5%)
        - Terminal Growth ~ Normal(2.5%, 0.2%)
        
        FCF for each simulation is computed as: Revenue × Margin
        This creates meaningful variation in the valuation distribution.
        """
        # Check if we have valid data for simulation
        if data.revenue <= 0:
            # Cannot simulate without revenue
            return np.array([])
        
        # Base parameters - use weighted average for growth
        g3y = max(data.revenue_growth_3y, 0.0)
        g5y = max(data.revenue_growth_5y, 0.0)
        if g3y > 0 and g5y > 0:
            base_growth = min(0.6 * g3y + 0.4 * g5y, MAX_GROWTH_RATE)
        else:
            base_growth = min(max(g3y, g5y, 0.02), MAX_GROWTH_RATE)
        
        # Use FCF margin if available, otherwise EBIT margin
        if data.free_cash_flow > 0 and data.revenue > 0:
            base_margin = data.free_cash_flow / data.revenue
        elif data.ebit_margin > 0:
            base_margin = data.ebit_margin * 0.75  # Approximate FCF margin from EBIT
        else:
            base_margin = 0.05  # Floor at 5%
        
        base_margin = max(base_margin, 0.01)  # Ensure positive
        base_wacc = data.wacc
        
        # Generate random parameters
        np.random.seed(42)  # For reproducibility
        
        growth_samples = np.random.normal(base_growth, 0.02, n_simulations)
        margin_samples = np.random.normal(base_margin, 0.02, n_simulations)  # Increased std for more variation
        wacc_samples = np.random.normal(base_wacc, 0.005, n_simulations)
        terminal_samples = np.random.normal(TERMINAL_GROWTH_BASE, 0.002, n_simulations)
        
        # Clip to reasonable ranges
        growth_samples = np.clip(growth_samples, 0.0, 0.20)
        margin_samples = np.clip(margin_samples, 0.01, 0.40)  # FCF margin between 1% and 40%
        wacc_samples = np.clip(wacc_samples, 0.04, 0.20)
        terminal_samples = np.clip(terminal_samples, 0.01, 0.04)
        
        # Ensure terminal < WACC
        terminal_samples = np.minimum(terminal_samples, wacc_samples - 0.01)
        
        # Run simulations - NOW USING MARGIN SAMPLES!
        fair_values = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            # Compute FCF for this simulation from Revenue × Margin
            simulated_fcf = data.revenue * margin_samples[i]
            
            ev = DCFValuation.calculate(
                data,
                growth_override=growth_samples[i],
                wacc_override=wacc_samples[i],
                terminal_growth_override=terminal_samples[i],
                fcf_override=simulated_fcf  # <-- NOW USING THE MARGIN!
            )
            fair_values[i] = DCFValuation.to_equity_value(ev, data, data.current_price)
        
        # Remove any invalid values
        fair_values = fair_values[fair_values > 0]
        fair_values = fair_values[~np.isnan(fair_values)]
        fair_values = fair_values[~np.isinf(fair_values)]
        
        return fair_values


# =============================================================================
# SENSITIVITY ANALYSIS MODULE
# =============================================================================

class SensitivityAnalysis:
    """Generate sensitivity matrices for valuation inputs."""
    
    @staticmethod
    def generate_matrix(data: FinancialData) -> pd.DataFrame:
        """
        Generate 2D sensitivity matrix for WACC vs Terminal Growth.
        """
        base_wacc = data.wacc
        
        # WACC variations: -1%, -0.5%, Base, +0.5%, +1%
        wacc_variations = [base_wacc - 0.01, base_wacc - 0.005, base_wacc, 
                          base_wacc + 0.005, base_wacc + 0.01]
        wacc_labels = [f"{(w*100):.1f}%" for w in wacc_variations]
        
        # Terminal Growth variations
        tg_variations = [TERMINAL_GROWTH_BASE - 0.01, TERMINAL_GROWTH_BASE - 0.005,
                        TERMINAL_GROWTH_BASE, TERMINAL_GROWTH_BASE + 0.005, 
                        TERMINAL_GROWTH_BASE + 0.01]
        tg_labels = [f"{(t*100):.1f}%" for t in tg_variations]
        
        # Calculate matrix
        matrix = []
        for wacc in wacc_variations:
            row = []
            for tg in tg_variations:
                if tg >= wacc:
                    row.append(np.nan)  # Invalid combination
                    continue
                ev = DCFValuation.calculate(data, wacc_override=wacc, terminal_growth_override=tg)
                per_share = DCFValuation.to_equity_value(ev, data, data.current_price)
                row.append(per_share)
            matrix.append(row)
        
        df = pd.DataFrame(matrix, index=wacc_labels, columns=tg_labels)
        df.index.name = 'WACC'
        df.columns.name = 'Terminal Growth'
        
        return df


# =============================================================================
# RELATIVE VALUATION MODULE (Trading Comps with Regression)
# =============================================================================

@dataclass
class RelativeValuationResult:
    """Container for relative valuation results."""
    ticker: str
    current_price: float = 0.0
    
    # Actual Multiples
    pe_ratio: float = 0.0
    ev_ebitda: float = 0.0
    revenue_growth: float = 0.0
    
    # Regression Implied Values
    pe_implied_by_growth: float = 0.0
    ev_implied_by_growth: float = 0.0
    
    # Discount/Premium vs Regression Line
    pe_discount_pct: float = 0.0  # Positive = Undervalued
    ev_discount_pct: float = 0.0
    
    # Fair Price based on regression
    fair_price_pe: float = 0.0
    fair_price_ev: float = 0.0
    
    # Regression Statistics
    pe_r_squared: float = 0.0
    ev_r_squared: float = 0.0
    
    # Peers data
    peers_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    is_valid: bool = True
    error_message: str = ""


class RelativeValuation:
    """
    Relative Valuation using Trading Comps with Linear Regression.
    
    State of the Art approach: Instead of simple average comparisons,
    we regress Growth vs Multiple for the sector. A stock below the
    regression line is statistically undervalued relative to its growth.
    """
    
    # Default peer groups by sector/industry
    SECTOR_PEERS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'CRM', 'ADBE', 'ORCL', 'CSCO', 'INTC'],
        'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY'],
        'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
        'Consumer': ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'COST', 'WMT'],
        'Industrials': ['HON', 'UPS', 'CAT', 'BA', 'GE', 'MMM', 'LMT', 'RTX', 'DE', 'UNP'],
        'Payments': ['V', 'MA', 'PYPL', 'SQ', 'FIS', 'FISV', 'GPN', 'ADYEN.AS', 'COIN', 'AFRM'],  # Payment processors
        'Insurance': ['BRK-B', 'PGR', 'ALL', 'TRV', 'MET', 'AIG', 'CB', 'AFL', 'HIG', 'PRU'],
        'Default': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI'],
    }
    
    # Industries that should use specific peer groups
    INDUSTRY_OVERRIDES = {
        'Credit Services': 'Payments',
        'Financial Data & Stock Exchanges': 'Payments',
        'Insurance': 'Insurance',
        'Insurance—Diversified': 'Insurance',
        'Insurance—Life': 'Insurance',
        'Insurance—Property & Casualty': 'Insurance',
    }
    
    @staticmethod
    def fetch_peer_data(peers: List[str], use_pb_fallback: bool = True) -> pd.DataFrame:
        """
        Fetch valuation multiples and growth for a list of peers.
        
        For banks/financials: uses P/B ratio since EBITDA doesn't apply.
        For other sectors: uses P/E and EV/EBITDA.
        
        Returns DataFrame with: Ticker, P/E, EV/EBITDA (or P/B), Revenue Growth, Market Cap
        """
        data = []
        
        for ticker in peers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Get key metrics
                pe = info.get('trailingPE') or info.get('forwardPE', 0) or 0
                ev = info.get('enterpriseValue', 0) or 0
                ebitda = info.get('ebitda', 0) or 0
                ev_ebitda = ev / ebitda if ebitda > 0 else 0
                
                # P/B ratio for financials
                pb = info.get('priceToBook', 0) or 0
                
                # Revenue growth
                rev_growth = info.get('revenueGrowth', 0) or 0
                
                # Earnings growth
                earnings_growth = info.get('earningsGrowth', 0) or info.get('earningsQuarterlyGrowth', 0) or rev_growth
                
                price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                market_cap = info.get('marketCap', 0)
                
                # More flexible filter: require P/E OR (P/B for financials)
                has_valid_pe = pe > 0 and pe < 100
                has_valid_ev = ev_ebitda > 0 and ev_ebitda < 50
                has_valid_pb = pb > 0 and pb < 10
                
                # Accept if has P/E and (EV/EBITDA or P/B as fallback)
                if has_valid_pe and (has_valid_ev or (use_pb_fallback and has_valid_pb)):
                    data.append({
                        'Ticker': ticker,
                        'Price': price,
                        'Market_Cap': market_cap,
                        'PE_Ratio': pe,
                        'EV_EBITDA': ev_ebitda if has_valid_ev else pb,  # Use P/B as fallback
                        'PB_Ratio': pb,
                        'Revenue_Growth': rev_growth * 100,
                        'Earnings_Growth': earnings_growth * 100,
                        'Is_PB_Fallback': not has_valid_ev and has_valid_pb,
                    })
            except Exception as e:
                continue  # Skip failed tickers
        
        return pd.DataFrame(data)
    
    @staticmethod
    def run_regression(df: pd.DataFrame, x_col: str, y_col: str) -> Tuple[float, float, float, float]:
        """
        Run linear regression: y = a + b*x
        
        Returns: (slope, intercept, r_squared, std_error)
        """
        if len(df) < 3:
            return 0, 0, 0, 0
        
        x = df[x_col].values
        y = df[y_col].values
        
        # Remove NaN and inf
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 3:
            return 0, 0, 0, 0
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return slope, intercept, r_value ** 2, std_err
    
    @staticmethod
    def analyze(ticker: str, peers: Optional[List[str]] = None, 
                verbose: bool = True) -> RelativeValuationResult:
        """
        Perform relative valuation analysis using regression.
        
        Parameters
        ----------
        ticker : str
            Target stock to value
        peers : list, optional
            List of peer tickers. If None, auto-selects based on sector.
        verbose : bool
            Print detailed output
        
        Returns
        -------
        RelativeValuationResult
        """
        result = RelativeValuationResult(ticker=ticker)
        
        try:
            # Fetch target stock data
            stock = yf.Ticker(ticker)
            info = stock.info
            
            result.current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            result.pe_ratio = info.get('trailingPE') or info.get('forwardPE', 0) or 0
            
            ev = info.get('enterpriseValue', 0)
            ebitda = info.get('ebitda', 0)
            result.ev_ebitda = ev / ebitda if ebitda and ebitda > 0 else 0
            
            result.revenue_growth = (info.get('revenueGrowth', 0) or 0) * 100
            
            # Determine sector and peers using industry override first
            sector = info.get('sector', 'Default')
            industry = info.get('industry', '')
            
            if peers is None:
                # Check industry-specific overrides first
                if industry in RelativeValuation.INDUSTRY_OVERRIDES:
                    peer_group = RelativeValuation.INDUSTRY_OVERRIDES[industry]
                    peers = RelativeValuation.SECTOR_PEERS[peer_group]
                    if verbose:
                        print(f"   Industry '{industry}' -> Using {peer_group} peers")
                # Then check sector
                elif 'Tech' in sector:
                    peers = RelativeValuation.SECTOR_PEERS['Technology']
                elif 'Health' in sector:
                    peers = RelativeValuation.SECTOR_PEERS['Healthcare']
                elif 'Financial' in sector:
                    peers = RelativeValuation.SECTOR_PEERS['Financials']
                elif 'Consumer' in sector:
                    peers = RelativeValuation.SECTOR_PEERS['Consumer']
                elif 'Industrial' in sector:
                    peers = RelativeValuation.SECTOR_PEERS['Industrials']
                else:
                    peers = RelativeValuation.SECTOR_PEERS['Default']
            
            # Ensure target is in peers for comparison
            if ticker not in peers:
                peers = [ticker] + peers[:9]  # Keep max 10 peers
            
            if verbose:
                print(f"\n   Fetching peer data for {len(peers)} companies...")
            
            # Fetch peer data
            peer_df = RelativeValuation.fetch_peer_data(peers)
            result.peers_data = peer_df
            
            if len(peer_df) < 4:
                result.is_valid = False
                result.error_message = "Insufficient peer data for regression"
                return result
            
            if verbose:
                print(f"   Successfully fetched data for {len(peer_df)} peers")
            
            # Run P/E vs Growth Regression
            pe_slope, pe_intercept, pe_r2, pe_stderr = RelativeValuation.run_regression(
                peer_df, 'Revenue_Growth', 'PE_Ratio'
            )
            result.pe_r_squared = pe_r2
            
            # Calculate implied P/E based on target's growth
            result.pe_implied_by_growth = pe_intercept + pe_slope * result.revenue_growth
            result.pe_implied_by_growth = max(5, min(result.pe_implied_by_growth, 50))  # Sanity bounds
            
            # Calculate discount/premium
            if result.pe_ratio > 0:
                result.pe_discount_pct = (result.pe_implied_by_growth - result.pe_ratio) / result.pe_ratio * 100
            
            # Fair price based on P/E regression
            if result.pe_ratio > 0:
                eps = result.current_price / result.pe_ratio
                result.fair_price_pe = eps * result.pe_implied_by_growth
            
            # Run EV/EBITDA vs Growth Regression
            ev_slope, ev_intercept, ev_r2, ev_stderr = RelativeValuation.run_regression(
                peer_df, 'Revenue_Growth', 'EV_EBITDA'
            )
            result.ev_r_squared = ev_r2
            
            # Calculate implied EV/EBITDA
            result.ev_implied_by_growth = ev_intercept + ev_slope * result.revenue_growth
            result.ev_implied_by_growth = max(3, min(result.ev_implied_by_growth, 30))
            
            # Calculate EV discount
            if result.ev_ebitda > 0:
                result.ev_discount_pct = (result.ev_implied_by_growth - result.ev_ebitda) / result.ev_ebitda * 100
            
            # Fair price based on EV/EBITDA regression
            if result.ev_ebitda > 0 and ebitda > 0:
                shares = info.get('sharesOutstanding', 1)
                total_debt = info.get('totalDebt', 0) or 0
                cash = info.get('totalCash', 0) or 0
                implied_ev = result.ev_implied_by_growth * ebitda
                implied_equity = implied_ev - total_debt + cash
                result.fair_price_ev = max(0, implied_equity / shares) if shares > 0 else 0
            
            if verbose:
                print(f"\n   --- Regression Results ---")
                print(f"   P/E Regression (R²={pe_r2:.2f}):")
                print(f"     Actual P/E: {result.pe_ratio:.1f}x")
                print(f"     Implied P/E (by growth): {result.pe_implied_by_growth:.1f}x")
                print(f"     Discount: {result.pe_discount_pct:+.1f}% {'(Undervalued)' if result.pe_discount_pct > 0 else '(Overvalued)'}")
                print(f"\n   EV/EBITDA Regression (R²={ev_r2:.2f}):")
                print(f"     Actual EV/EBITDA: {result.ev_ebitda:.1f}x")
                print(f"     Implied EV/EBITDA: {result.ev_implied_by_growth:.1f}x")
                print(f"     Discount: {result.ev_discount_pct:+.1f}% {'(Undervalued)' if result.ev_discount_pct > 0 else '(Overvalued)'}")
            
        except Exception as e:
            result.is_valid = False
            result.error_message = str(e)
        
        return result
    
    @staticmethod
    def plot_regression(result: RelativeValuationResult, save_path: Optional[str] = None):
        """
        Plot Growth vs Multiple regression with target stock highlighted.
        """
        if result.peers_data.empty:
            print("No peer data available for plotting.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Relative Valuation: {result.ticker} vs Sector Peers', 
                     fontsize=14, fontweight='bold')
        
        peer_df = result.peers_data
        
        # Plot 1: P/E vs Growth
        ax1 = axes[0]
        other_peers = peer_df[peer_df['Ticker'] != result.ticker]
        target_data = peer_df[peer_df['Ticker'] == result.ticker]
        
        # Scatter peers
        ax1.scatter(other_peers['Revenue_Growth'], other_peers['PE_Ratio'], 
                   s=100, alpha=0.6, color='steelblue', label='Peers', edgecolors='black')
        
        # Highlight target
        if not target_data.empty:
            ax1.scatter(target_data['Revenue_Growth'], target_data['PE_Ratio'],
                       s=200, color='red', marker='*', label=result.ticker, edgecolors='black', zorder=5)
        
        # Regression line
        x_range = np.linspace(peer_df['Revenue_Growth'].min() - 5, 
                              peer_df['Revenue_Growth'].max() + 5, 100)
        slope, intercept, _, _ = RelativeValuation.run_regression(peer_df, 'Revenue_Growth', 'PE_Ratio')
        y_pred = intercept + slope * x_range
        ax1.plot(x_range, y_pred, 'g--', linewidth=2, label='Regression Line')
        
        # Mark implied value
        ax1.scatter([result.revenue_growth], [result.pe_implied_by_growth], 
                   s=150, color='green', marker='D', label=f'Implied P/E: {result.pe_implied_by_growth:.1f}x',
                   edgecolors='black', zorder=5)
        
        # Labels for each point
        for _, row in peer_df.iterrows():
            ax1.annotate(row['Ticker'], (row['Revenue_Growth'], row['PE_Ratio']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Revenue Growth (%)')
        ax1.set_ylabel('P/E Ratio')
        ax1.set_title(f'P/E vs Growth (R²={result.pe_r_squared:.2f})\n'
                     f'Discount: {result.pe_discount_pct:+.1f}%')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: EV/EBITDA vs Growth
        ax2 = axes[1]
        
        ax2.scatter(other_peers['Revenue_Growth'], other_peers['EV_EBITDA'],
                   s=100, alpha=0.6, color='steelblue', label='Peers', edgecolors='black')
        
        if not target_data.empty:
            ax2.scatter(target_data['Revenue_Growth'], target_data['EV_EBITDA'],
                       s=200, color='red', marker='*', label=result.ticker, edgecolors='black', zorder=5)
        
        # Regression line
        slope, intercept, _, _ = RelativeValuation.run_regression(peer_df, 'Revenue_Growth', 'EV_EBITDA')
        y_pred = intercept + slope * x_range
        ax2.plot(x_range, y_pred, 'g--', linewidth=2, label='Regression Line')
        
        ax2.scatter([result.revenue_growth], [result.ev_implied_by_growth],
                   s=150, color='green', marker='D', label=f'Implied EV/EBITDA: {result.ev_implied_by_growth:.1f}x',
                   edgecolors='black', zorder=5)
        
        for _, row in peer_df.iterrows():
            ax2.annotate(row['Ticker'], (row['Revenue_Growth'], row['EV_EBITDA']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Revenue Growth (%)')
        ax2.set_ylabel('EV/EBITDA')
        ax2.set_title(f'EV/EBITDA vs Growth (R²={result.ev_r_squared:.2f})\n'
                     f'Discount: {result.ev_discount_pct:+.1f}%')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nRelative valuation chart saved to: {save_path}")
        
        plt.show()


# =============================================================================
# VALUATION ENGINE (Main Orchestrator)
# =============================================================================

class ValuationEngine:
    """Main valuation engine orchestrating all modules."""
    
    def __init__(self):
        self.results: Dict[str, ValuationResult] = {}
        self.financial_data: Dict[str, FinancialData] = {}
        self.relative_results: Dict[str, RelativeValuationResult] = {}
    
    def analyze(self, ticker: str, verbose: bool = True) -> ValuationResult:
        """Run full valuation analysis on a single ticker."""
        result = ValuationResult(ticker=ticker, current_price=0.0)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {ticker}")
            print('='*60)
        
        # 1. Fetch Data
        if verbose:
            print("\n1. Fetching financial data...")
        data = DataFetcher.fetch(ticker)
        self.financial_data[ticker] = data
        
        if not data.is_valid:
            result.is_valid = False
            result.error_message = data.error_message
            if verbose:
                print(f"   ❌ Error: {data.error_message}")
            return result
        
        result.current_price = data.current_price
        
        if verbose:
            print(f"   Price: ${data.current_price:.2f}")
            print(f"   Revenue: ${data.revenue/1e9:.2f}B")
            print(f"   FCF: ${data.free_cash_flow/1e9:.2f}B")
            print(f"   Beta: {data.beta:.2f}")
            print(f"   WACC: {data.wacc*100:.2f}%")
            
            # Print data quality warnings
            if data.missing_fields:
                print(f"   ⚠️ Missing data: {', '.join(data.missing_fields[:5])}")
                if len(data.missing_fields) > 5:
                    print(f"      ... and {len(data.missing_fields) - 5} more fields")
            if data.data_quality_warnings:
                for warning in data.data_quality_warnings[:3]:
                    print(f"   ⚠️ {warning}")
                if len(data.data_quality_warnings) > 3:
                    print(f"      ... and {len(data.data_quality_warnings) - 3} more warnings")
        
        # 2. Base DCF
        if verbose:
            print("\n2. Running DCF valuation...")
        result.dcf_value = DCFValuation.calculate(data)
        result.dcf_per_share = DCFValuation.to_equity_value(result.dcf_value, data, data.current_price)
        
        if verbose:
            print(f"   DCF Value: ${result.dcf_per_share:.2f} per share")
        
        # 3. Monte Carlo
        if verbose:
            print("\n3. Running Monte Carlo simulation (10,000 trials)...")
        simulations = MonteCarloValuation.simulate(data)
        result.simulation_values = simulations
        
        if len(simulations) > 0:
            result.mc_mean = np.mean(simulations)
            result.mc_median = np.median(simulations)
            result.mc_std = np.std(simulations)
            result.mc_p10 = np.percentile(simulations, 10)
            result.mc_p25 = np.percentile(simulations, 25)
            result.mc_p75 = np.percentile(simulations, 75)
            result.mc_p90 = np.percentile(simulations, 90)
            
            # Win probability
            result.win_probability = np.mean(simulations > data.current_price) * 100
            
            # Margin of safety
            result.margin_of_safety = (result.mc_mean - data.current_price) / data.current_price * 100
        
        if verbose:
            print(f"   Mean Value: ${result.mc_mean:.2f}")
            print(f"   P10 (Pessimistic): ${result.mc_p10:.2f}")
            print(f"   P90 (Optimistic): ${result.mc_p90:.2f}")
            print(f"   Win Probability: {result.win_probability:.1f}%")
        
        # 4. Sensitivity Analysis
        if verbose:
            print("\n4. Generating sensitivity matrix...")
        result.sensitivity_matrix = SensitivityAnalysis.generate_matrix(data)
        
        self.results[ticker] = result
        
        if verbose:
            print("\n" + "-"*60)
            print("VALUATION SUMMARY")
            print("-"*60)
            if result.margin_of_safety > 20:
                signal = "🟢 UNDERVALUED"
            elif result.margin_of_safety > 0:
                signal = "🟡 FAIRLY VALUED"
            else:
                signal = "🔴 OVERVALUED"
            
            print(f"   Current Price: ${result.current_price:.2f}")
            print(f"   Fair Value (MC Mean): ${result.mc_mean:.2f}")
            print(f"   Margin of Safety: {result.margin_of_safety:.1f}%")
            print(f"   Signal: {signal}")
        
        return result
    
    def analyze_relative(self, ticker: str, peers: Optional[List[str]] = None,
                         verbose: bool = True) -> RelativeValuationResult:
        """
        Run relative valuation (Trading Comps) with regression analysis.
        
        Parameters
        ----------
        ticker : str
            Stock to analyze
        peers : list, optional
            Custom peer list. If None, auto-selects by sector.
        verbose : bool
            Print detailed output
        """
        if verbose:
            print(f"\n5. Running Relative Valuation (Trading Comps)...")
        
        rel_result = RelativeValuation.analyze(ticker, peers, verbose)
        self.relative_results[ticker] = rel_result
        
        # Update main result with relative metrics
        if ticker in self.results and rel_result.is_valid:
            self.results[ticker].relative_pe_implied = rel_result.pe_implied_by_growth
            self.results[ticker].relative_ev_ebitda_implied = rel_result.ev_implied_by_growth
            self.results[ticker].relative_pe_discount = rel_result.pe_discount_pct
            self.results[ticker].relative_ev_discount = rel_result.ev_discount_pct
        
        return rel_result
    
    def analyze_full(self, ticker: str, peers: Optional[List[str]] = None,
                     verbose: bool = True) -> Tuple[ValuationResult, RelativeValuationResult]:
        """
        Run complete analysis: DCF + Monte Carlo + Relative Valuation.
        """
        dcf_result = self.analyze(ticker, verbose)
        rel_result = self.analyze_relative(ticker, peers, verbose)
        
        if verbose and rel_result.is_valid:
            print("\n" + "-"*60)
            print("COMBINED VALUATION SUMMARY")
            print("-"*60)
            print(f"   Current Price: ${dcf_result.current_price:.2f}")
            print(f"\n   Intrinsic (DCF Monte Carlo):")
            print(f"     Fair Value: ${dcf_result.mc_mean:.2f}")
            print(f"     Margin of Safety: {dcf_result.margin_of_safety:.1f}%")
            print(f"\n   Relative (vs Sector Peers):")
            print(f"     P/E Implied Price: ${rel_result.fair_price_pe:.2f}")
            print(f"     P/E Discount: {rel_result.pe_discount_pct:+.1f}%")
            print(f"     EV/EBITDA Implied Price: ${rel_result.fair_price_ev:.2f}")
            print(f"     EV/EBITDA Discount: {rel_result.ev_discount_pct:+.1f}%")
            
            # Combined signal
            avg_discount = (dcf_result.margin_of_safety + rel_result.pe_discount_pct + rel_result.ev_discount_pct) / 3
            if avg_discount > 15:
                combined_signal = "🟢 STRONG BUY"
            elif avg_discount > 0:
                combined_signal = "🟡 HOLD/ACCUMULATE"
            else:
                combined_signal = "🔴 OVERVALUED"
            
            print(f"\n   Combined Signal: {combined_signal}")
        
        return dcf_result, rel_result
    
    def analyze_batch(self, tickers: List[str], verbose: bool = False) -> pd.DataFrame:
        """Analyze multiple tickers and return summary DataFrame."""
        print(f"\nAnalyzing {len(tickers)} stocks...")
        
        for ticker in tickers:
            try:
                self.analyze(ticker, verbose=verbose)
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
        
        return self.get_summary()
    
    def get_summary(self) -> pd.DataFrame:
        """Generate master summary DataFrame."""
        rows = []
        
        for ticker, result in self.results.items():
            if not result.is_valid:
                continue
            
            rows.append({
                'Ticker': ticker,
                'Current Price': result.current_price,
                'DCF Value': result.dcf_per_share,
                'MC Mean': result.mc_mean,
                'MC P10 (Bear)': result.mc_p10,
                'MC P90 (Bull)': result.mc_p90,
                'Margin of Safety (%)': result.margin_of_safety,
                'Win Probability (%)': result.win_probability,
            })
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            df = df.sort_values('Margin of Safety (%)', ascending=False)
        
        return df
    
    def plot_valuation(self, ticker: str, save_path: Optional[str] = None):
        """Plot Monte Carlo histogram and sensitivity heatmap for a ticker."""
        if ticker not in self.results:
            print(f"No results for {ticker}. Run analyze() first.")
            return
        
        result = self.results[ticker]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Valuation Analysis: {ticker}', fontsize=14, fontweight='bold')
        
        # 1. Monte Carlo Histogram
        ax1 = axes[0]
        if len(result.simulation_values) > 0:
            ax1.hist(result.simulation_values, bins=50, alpha=0.7, color='steelblue', 
                    edgecolor='black', density=True)
            ax1.axvline(result.current_price, color='red', linestyle='--', 
                       linewidth=2, label=f'Current Price: ${result.current_price:.2f}')
            ax1.axvline(result.mc_mean, color='green', linestyle='-', 
                       linewidth=2, label=f'MC Mean: ${result.mc_mean:.2f}')
            ax1.axvline(result.mc_p10, color='orange', linestyle=':', 
                       linewidth=1.5, label=f'P10: ${result.mc_p10:.2f}')
            ax1.axvline(result.mc_p90, color='orange', linestyle=':', 
                       linewidth=1.5, label=f'P90: ${result.mc_p90:.2f}')
            
            # Fill area above current price
            ax1.fill_betweenx([0, ax1.get_ylim()[1] * 0.8], result.current_price, 
                             result.simulation_values.max(), alpha=0.2, color='green')
            
            ax1.set_xlabel('Fair Value per Share ($)')
            ax1.set_ylabel('Density')
            ax1.set_title(f'Monte Carlo Fair Value Distribution\n'
                         f'Win Probability: {result.win_probability:.1f}%')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
        
        # 2. Sensitivity Heatmap
        ax2 = axes[1]
        if not result.sensitivity_matrix.empty:
            sns.heatmap(result.sensitivity_matrix, annot=True, fmt='.0f', 
                       cmap='RdYlGn', center=result.current_price,
                       ax=ax2, cbar_kws={'label': 'Fair Value ($)'})
            ax2.set_title('Sensitivity: WACC vs Terminal Growth\n'
                         f'(Current Price: ${result.current_price:.2f})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution with example stocks."""
    
    # Example tickers to analyze
    TICKERS = ['UNH', 'TMO', 'V', 'JNJ', 'PG']
    
    print("="*60)
    print("VALUATION ENGINE - State of the Art DCF Analysis")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  • Risk-Free Rate: {RISK_FREE_RATE*100:.1f}%")
    print(f"  • Equity Risk Premium: {EQUITY_RISK_PREMIUM*100:.1f}%")
    print(f"  • Projection Period: {PROJECTION_YEARS} years")
    print(f"  • Terminal Growth: {TERMINAL_GROWTH_BASE*100:.1f}%")
    print(f"  • Monte Carlo Simulations: {N_SIMULATIONS:,}")
    
    # Initialize engine
    engine = ValuationEngine()
    
    # Analyze stocks
    for ticker in TICKERS:
        engine.analyze(ticker, verbose=True)
    
    # Generate summary
    print("\n" + "="*60)
    print("MASTER SUMMARY")
    print("="*60)
    
    summary = engine.get_summary()
    print("\n")
    print(summary.to_string(index=False))
    
    # Save summary
    summary.to_csv('valuation_summary.csv', index=False)
    print("\nSummary saved to: valuation_summary.csv")
    
    # Plot first stock
    if TICKERS:
        print(f"\nGenerating visualization for {TICKERS[0]}...")
        engine.plot_valuation(TICKERS[0], save_path=f'valuation_{TICKERS[0]}_chart.png')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return engine


if __name__ == "__main__":
    engine = main()
