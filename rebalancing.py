"""
ECE Portfolio Analysis Engine
=============================
Rebalancing Optimizer Module
============================
Convert target allocations to executable trade orders.
Answers: "What exactly do I need to buy/sell to reach target weights?"

Author: Josh E. SOUSSAN
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import yfinance as yf


@dataclass
class TradeOrder:
    """Single trade order."""
    ticker: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    estimated_value: float
    current_weight: float
    target_weight: float
    weight_change: float
    current_price: float


@dataclass
class RebalancingPlan:
    """Complete rebalancing plan with all trades."""
    orders: List[TradeOrder]
    total_buys: float
    total_sells: float
    net_cash_flow: float
    turnover: float
    portfolio_value: float
    num_trades: int


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch current prices for a list of tickers.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols
    
    Returns
    -------
    dict
        ticker -> current price
    """
    prices = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Try to get the most recent price
            hist = stock.history(period='1d')
            if not hist.empty:
                prices[ticker] = hist['Close'].iloc[-1]
            else:
                # Fallback to info
                info = stock.info
                prices[ticker] = info.get('regularMarketPrice', 0) or info.get('previousClose', 0)
        except Exception as e:
            print(f"Warning: Could not get price for {ticker}: {e}")
            prices[ticker] = 0
    
    return prices


def calculate_rebalancing_trades(
    current_holdings: Dict[str, float],  # ticker -> current_value (USD)
    target_weights: Dict[str, float],    # ticker -> target_weight (%)
    current_prices: Dict[str, float] = None,  # ticker -> price (optional, will fetch)
    portfolio_value: float = None,       # Total portfolio value (optional, calculated from holdings)
    min_trade_value: float = 100.0,      # Minimum trade size to execute
    round_lots: bool = False             # If True, round to 100 share lots
) -> RebalancingPlan:
    """
    Generate trade orders to reach target allocation.
    
    Parameters
    ----------
    current_holdings : dict
        Current positions: ticker -> market value in USD
    target_weights : dict
        Target weights: ticker -> weight as percentage (0-100)
    current_prices : dict, optional
        If not provided, will fetch from yfinance
    portfolio_value : float, optional
        If not provided, calculated as sum of current_holdings
    min_trade_value : float
        Trades smaller than this are ignored
    round_lots : bool
        If True, round to 100-share lots (for institutional)
    
    Returns
    -------
    RebalancingPlan
        Complete rebalancing plan with trade orders
    
    Example
    -------
    >>> current = {'AAPL': 50000, 'MSFT': 30000, 'GOOGL': 20000}
    >>> target = {'AAPL': 40, 'MSFT': 35, 'GOOGL': 25}
    >>> plan = calculate_rebalancing_trades(current, target)
    >>> for order in plan.orders:
    ...     print(f"{order.action} {order.shares} {order.ticker}")
    """
    # Calculate portfolio value if not provided
    if portfolio_value is None:
        portfolio_value = sum(current_holdings.values())
    
    if portfolio_value <= 0:
        raise ValueError("Portfolio value must be positive")
    
    # Get all tickers
    all_tickers = list(set(current_holdings.keys()) | set(target_weights.keys()))
    
    # Fetch prices if not provided
    if current_prices is None:
        current_prices = get_current_prices(all_tickers)
    
    orders = []
    total_buys = 0
    total_sells = 0
    total_weight_change = 0
    
    for ticker in all_tickers:
        current_value = current_holdings.get(ticker, 0.0)
        target_weight = target_weights.get(ticker, 0.0)
        
        current_weight = (current_value / portfolio_value) * 100 if portfolio_value > 0 else 0
        target_value = portfolio_value * (target_weight / 100)
        delta = target_value - current_value
        
        # Track weight change for turnover calculation
        total_weight_change += abs(target_weight - current_weight)
        
        # Skip small trades
        if abs(delta) < min_trade_value:
            continue
        
        price = current_prices.get(ticker, 0)
        if not (price > 0):  # Handles 0, negative, and NaN
            print(f"Warning: Skipping {ticker} - price unavailable")
            continue
        
        # Calculate shares
        shares = int(abs(delta) / price)
        
        # Round to lots if requested
        if round_lots and shares >= 100:
            shares = (shares // 100) * 100
        
        if shares == 0:
            continue
        
        action = 'BUY' if delta > 0 else 'SELL'
        trade_value = shares * price
        
        if action == 'BUY':
            total_buys += trade_value
        else:
            total_sells += trade_value
        
        orders.append(TradeOrder(
            ticker=ticker,
            action=action,
            shares=shares,
            estimated_value=trade_value,
            current_weight=current_weight,
            target_weight=target_weight,
            weight_change=target_weight - current_weight,
            current_price=price
        ))
    
    # Sort by absolute trade value (largest first)
    orders.sort(key=lambda x: x.estimated_value, reverse=True)
    
    # Calculate turnover (one-way, as percentage of portfolio)
    turnover = total_weight_change / 2  # percentage points
    
    return RebalancingPlan(
        orders=orders,
        total_buys=total_buys,
        total_sells=total_sells,
        net_cash_flow=total_sells - total_buys,
        turnover=turnover,
        portfolio_value=portfolio_value,
        num_trades=len(orders)
    )


def display_rebalancing_plan(plan: RebalancingPlan):
    """
    Display rebalancing plan in a formatted table.
    """
    print("\n" + "="*80)
    print("REBALANCING PLAN")
    print("="*80)
    print(f"\nPortfolio Value: ${plan.portfolio_value:,.2f}")
    print(f"Number of Trades: {plan.num_trades}")
    print(f"Turnover: {plan.turnover:.1f}%")
    print()
    
    if not plan.orders:
        print("No trades needed - portfolio is already at target weights.")
        return
    
    # Separate buys and sells
    buys = [o for o in plan.orders if o.action == 'BUY']
    sells = [o for o in plan.orders if o.action == 'SELL']
    
    if sells:
        print("--- SELL ORDERS ---")
        for o in sells:
            print(f"  [SELL] {o.shares:>6,} {o.ticker:<6} @ ${o.current_price:>8,.2f} = ${o.estimated_value:>12,.2f}")
            print(f"       Weight: {o.current_weight:>5.1f}% -> {o.target_weight:>5.1f}% ({o.weight_change:+.1f}%)")
        print(f"\n  Total Sells: ${plan.total_sells:,.2f}")
    
    if buys:
        print("\n--- BUY ORDERS ---")
        for o in buys:
            print(f"  [BUY]  {o.shares:>6,} {o.ticker:<6} @ ${o.current_price:>8,.2f} = ${o.estimated_value:>12,.2f}")
            print(f"       Weight: {o.current_weight:>5.1f}% -> {o.target_weight:>5.1f}% ({o.weight_change:+.1f}%)")
        print(f"\n  Total Buys: ${plan.total_buys:,.2f}")
    
    print("\n" + "-"*80)
    if plan.net_cash_flow >= 0:
        print(f"[CASH] Net Cash Generated: ${plan.net_cash_flow:,.2f}")
    else:
        print(f"[REQ] Net Cash Required:  ${abs(plan.net_cash_flow):,.2f}")
    print("="*80)


def rebalancing_summary_df(plan: RebalancingPlan) -> pd.DataFrame:
    """
    Convert rebalancing plan to DataFrame for export.
    """
    if not plan.orders:
        return pd.DataFrame()
    
    data = []
    for o in plan.orders:
        data.append({
            'Ticker': o.ticker,
            'Action': o.action,
            'Shares': o.shares,
            'Price ($)': o.current_price,
            'Value ($)': o.estimated_value,
            'Current Weight (%)': o.current_weight,
            'Target Weight (%)': o.target_weight,
            'Weight Change (%)': o.weight_change
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("REBALANCING OPTIMIZER - DEMO")
    print("="*60)
    
    # Example: Current holdings
    current_holdings = {
        'AAPL': 50000,   # $50k in Apple
        'MSFT': 30000,   # $30k in Microsoft
        'GOOGL': 15000,  # $15k in Google
        'AMZN': 5000,    # $5k in Amazon
    }
    
    # Target allocation
    target_weights = {
        'AAPL': 35,   # Want 35%
        'MSFT': 35,   # Want 35%
        'GOOGL': 20,  # Want 20%
        'AMZN': 10,   # Want 10%
    }
    
    print("\nCurrent Holdings:")
    for t, v in current_holdings.items():
        print(f"  {t}: ${v:,.0f}")
    
    print("\nTarget Weights:")
    for t, w in target_weights.items():
        print(f"  {t}: {w}%")
    
    # Generate rebalancing plan
    plan = calculate_rebalancing_trades(current_holdings, target_weights)
    display_rebalancing_plan(plan)
