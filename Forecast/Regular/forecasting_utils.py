
import os
import pandas as pd
import numpy as np
import sys

# Import fetch_data from FTS inference
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Forecast/FinCast-fts/Inference')))
try:
    import fetch_data
    print("✅ Successfully imported fetch_data")
except ImportError as e:
    print(f"❌ Failed to import fetch_data: {e}")
    fetch_data = None

def get_market_data(ticker, freq='d', days=None, years=None, output_dir=None):
    """
    Fetch market data using the shared Refinitiv fetcher.
    Auto-caches to the specified output_dir or analysis_outputs/[TICKER]/forecast/data.csv
    
    Args:
        ticker: Stock ticker or RIC
        freq: Data frequency
        days: Number of days to fetch
        years: Number of years to fetch
        output_dir: Optional. If provided, saves data here. Otherwise uses analysis_outputs/[TICKER]/forecast/
    """
    # 1. Determine output path
    if output_dir:
        stock_dir = output_dir
    else:
        # Fallback: Use global structure ECE/analysis_outputs/[TICKER]/forecast/
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../analysis_outputs'))
        stock_dir = os.path.join(base_dir, ticker, 'forecast')
    
    os.makedirs(stock_dir, exist_ok=True)
    csv_path = os.path.join(stock_dir, 'data.csv')
    
    # 2. Fetch or Load
    print(f"Fetching data for {ticker}...")
    
    if fetch_data:
        # Convert simple ticker to RIC if needed
        ric = ticker
        if not "." in ric and not "=" in ric:
             ric = f"{ticker}.O" 

        # Try to connect
        try:
             fetch_data.connect()
        except Exception as e:
             print(f"⚠️ Connection attempt failed: {e}")

        try:
            df = fetch_data.fetch_close_prices(ric=ric, freq=freq, days=days, years=years)
            if not df.empty:
                # Save to cache
                saved_path = fetch_data.save_csv(df, ric, freq, stock_dir)
                return df
        except Exception as e:
            print(f"⚠️ Fetch failed: {e}")
    
    # Fallback: Try loading existing cache
    if os.path.exists(csv_path):
        print(f"⚠️ Using cached data from {csv_path}")
        return pd.read_csv(csv_path)
        
    return pd.DataFrame()

def add_technical_indicators(df):
    """Add standard technical indicators for ARIMAX/ML models."""
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange

    if df.empty: return df
    
    df = df.copy()
    close = df['Close']
    
    # Trend
    df['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
    df['EMA_20'] = EMAIndicator(close, window=20).ema_indicator()
    df['MACD'] = MACD(close).macd()
    
    # Momentum
    df['RSI'] = RSIIndicator(close).rsi()
    df['Stoch_k'] = StochasticOscillator(df['Close'], df['Close'], df['Close']).stoch() # Approximation using Close as High/Low
    
    # Volatility
    bb = BollingerBands(close)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    df.dropna(inplace=True)
    return df

def save_forecast_results(df, results, output_dir):
    """
    Save consolidated forecast results to CSV and JSON.
    Handles different forecast lengths by padding shorter ones with NaN.
    """
    try:
        if not results: return
        
        # Determine max horizon from all results
        max_horizon = max(len(res.get('forecast', [])) for res in results)
        if max_horizon == 0: return
        
        # Generate future dates based on max horizon
        last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Exchange Date'].iloc[-1])
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(max_horizon)]
        
        # Create Forecast DataFrame
        res_df = pd.DataFrame(index=future_dates)
        res_df.index.name = 'Date'
        
        for res in results:
            model_name = res.get('model', 'Unknown')
            forecast = res.get('forecast', [])
            if not forecast: continue
            
            # Pad shorter forecasts with NaN
            padded = list(forecast) + [float('nan')] * (max_horizon - len(forecast))
            res_df[model_name] = padded
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'combined_forecasts.csv')
        res_df.to_csv(csv_path)
        print(f"   \U0001F4BE Saved combined forecasts to {csv_path}")
        
        return res_df
    except Exception as e:
        print(f"   \u274C Error saving forecast results: {e}")
        return None

def plot_single_forecast(df, result, ticker, output_dir, tail_points=30):
    """
    Generate FTS-style dark-themed plots for a SINGLE model's forecast.
    Produces two plots per model:
      1. Context + Forecast (history tail + forecast line)
      2. Zoomed view (last few points + forecast)
    """
    import matplotlib.pyplot as plt
    
    model_name = result.get('model', 'Unknown')
    forecast = result.get('forecast', [])
    if not forecast:
        return
    
    # Extract close prices
    if 'Close' in df.columns:
        close_values = pd.to_numeric(df['Close'], errors='coerce').dropna().values
    else:
        return
    
    H = len(forecast)
    safe_name = model_name.replace(' ', '_').lower()
    
    # =========================================================================
    # PLOT 1: Context + Forecast (FTS style)
    # =========================================================================
    try:
        ctx = close_values[-tail_points:]
        L = len(ctx)
        
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        
        x_ctx = np.arange(L)
        x_fut = np.arange(L, L + H)
        
        # Historical context
        ax.plot(x_ctx, ctx, color='#4cc9f0', linewidth=1.8, label='Historical (Context)')
        
        # Mean forecast
        ax.plot(x_fut, forecast, color='#e63946',
                linewidth=2.5, marker='o', markersize=4, label=f'Forecast ({model_name})')
        
        # Connect history to forecast with dashed line
        ax.plot([x_ctx[-1], x_fut[0]],
                [ctx[-1], forecast[0]],
                color='#888', linestyle='--', linewidth=1.5)
        
        # Vertical separator
        ax.axvline(x=L - 0.5, color='#888', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(L - 1, ax.get_ylim()[1], ' Forecast \u2192', color='#aaa',
                fontsize=9, va='top', ha='left')
        
        ax.set_title(f"{ticker}  |  {model_name}  |  Context={L}, Horizon={H}",
                     fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_xlabel("Time (relative index)", fontsize=12, color='#ccc')
        ax.set_ylabel("Price", fontsize=12, color='#ccc')
        ax.tick_params(colors='#aaa')
        ax.grid(True, alpha=0.2, color='#555')
        ax.legend(loc='best', frameon=True, facecolor='#1a1a2e',
                  edgecolor='#555', labelcolor='white', fontsize=9)
        
        for spine in ax.spines.values():
            spine.set_color('#333')
        
        ctx_path = os.path.join(output_dir, f"{ticker}_{safe_name}_context.png")
        plt.savefig(ctx_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print(f"   \U0001F4CA {model_name} context plot \u2192 {ctx_path}")
    except Exception as e:
        print(f"   \u274C Error plotting {model_name} context: {e}")
    
    # =========================================================================
    # PLOT 2: Zoomed view (last 10 points + forecast)
    # =========================================================================
    try:
        zoom_pts = 10
        tail = close_values[-zoom_pts:]
        T = len(tail)
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#16213e')
        
        x_hist = np.arange(T)
        x_fut = np.arange(T, T + H)
        
        # Recent history
        ax.plot(x_hist, tail, color='#4cc9f0', linewidth=2.5,
                marker='o', markersize=5, label='Recent History')
        
        # Forecast
        ax.plot(x_fut, forecast, color='#e63946',
                linewidth=2.5, marker='o', markersize=5, label=f'Forecast ({model_name})')
        
        # Connect
        ax.plot([x_hist[-1], x_fut[0]],
                [tail[-1], forecast[0]],
                color='#888', linestyle='--', linewidth=1.5)
        
        # Vertical separator
        ax.axvline(x=T - 0.5, color='#f77f00', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.text(T - 0.3, ax.get_ylim()[1], '  Forecast \u2192',
                color='#f77f00', fontsize=10, fontweight='bold', va='top', ha='left')
        
        # Summary annotation
        start_price = tail[-1]
        end_price = forecast[-1]
        change_pct = ((end_price / start_price) - 1) * 100
        trend = 'UP' if change_pct > 0 else 'DOWN'
        
        ax.annotate(f"{trend}  {change_pct:+.2f}%\nLast: ${start_price:.2f} -> ${end_price:.2f}",
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, color='white', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#333', alpha=0.8))
        
        ax.set_title(f"{ticker}  |  {model_name}  |  Zoomed View  |  Horizon={H}",
                     fontsize=16, fontweight='bold', color='white', pad=15)
        ax.set_xlabel("Steps", fontsize=12, color='#ccc')
        ax.set_ylabel("Price", fontsize=12, color='#ccc')
        ax.tick_params(colors='#aaa')
        ax.grid(True, alpha=0.2, color='#555')
        ax.legend(loc='best', frameon=True, facecolor='#1a1a2e',
                  edgecolor='#555', labelcolor='white', fontsize=9)
        
        for spine in ax.spines.values():
            spine.set_color('#333')
        
        zoom_path = os.path.join(output_dir, f"{ticker}_{safe_name}_zoomed.png")
        plt.savefig(zoom_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close()
        print(f"   \U0001F50D {model_name} zoomed plot \u2192 {zoom_path}")
    except Exception as e:
        print(f"   \u274C Error plotting {model_name} zoomed: {e}")

def plot_all_forecasts(df, results, ticker, output_dir):
    """
    Generate individual FTS-style plots for EACH model result.
    """
    for res in results:
        if 'error' not in res:
            plot_single_forecast(df, res, ticker, output_dir)
