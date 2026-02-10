"""
FinCast Data Fetcher — Pull close prices from Refinitiv for any ticker/frequency/range.

Usage:
    python fetch_data.py AAPL.O                           # daily, last 5 years
    python fetch_data.py AAPL.O --freq 1min --days 30     # 1-min bars, last 30 days
    python fetch_data.py MSFT.O --freq 1h --days 90       # hourly, last 90 days
    python fetch_data.py VOD.L  --freq d --years 10       # daily, last 10 years
    python fetch_data.py EUR=   --freq 5min --days 5      # FX, 5-min, last 5 days

Supported frequencies:
    tick, 1min, 5min, 10min, 15min, 30min, 1h, d, w, m, q, y

Output:
    Saves a clean CSV (Exchange Date, Close) ready for FinCast inference.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import pandas as pd

# ─── Refinitiv Data Library ───────────────────────────────────────────────────
try:
    import refinitiv.data as rd
    from refinitiv.data.content import historical_pricing
except ImportError:
    print("ERROR: refinitiv-data not installed.")
    print("Run:  uv pip install refinitiv-data --python .\\.venv\\Scripts\\python.exe")
    sys.exit(1)


# ─── Frequency mapping for FinCast config ────────────────────────────────────
FREQ_TO_FINCAST = {
    "tick": "s",
    "1min": "t",
    "5min": "t",
    "10min": "t",
    "15min": "t",
    "30min": "t",
    "1h": "h",
    "d": "d",
    "daily": "d",
    "w": "w",
    "weekly": "w",
    "m": "m",
    "monthly": "m",
    "q": "q",
    "quarterly": "q",
    "y": "y",
    "yearly": "y",
}

# Refinitiv interval mapping
FREQ_TO_RD_INTERVAL = {
    "tick": "tick",
    "1min": "minute",
    "5min": "five_minutes",
    "10min": "ten_minutes",
    "15min": "fifteen_minutes",
    "30min": "thirty_minutes",
    "1h": "hourly",
    "d": "daily",
    "daily": "daily",
    "w": "weekly",
    "weekly": "weekly",
    "m": "monthly",
    "monthly": "monthly",
    "q": "quarterly",
    "quarterly": "quarterly",
    "y": "yearly",
    "yearly": "yearly",
}


def connect():
    """Open a Refinitiv Data session (Desktop/Workspace)."""
    try:
        rd.open_session()
        print("✅ Connected to Refinitiv")
    except Exception as e:
        print(f"❌ Failed to connect to Refinitiv: {e}")
        print("\nMake sure:")
        print("  1. Refinitiv Workspace is running")
        print("  2. You are logged in")
        print("  3. The API proxy is enabled (Workspace > Settings > API)")
        sys.exit(1)


def fetch_close_prices(
    ric: str,
    freq: str = "d",
    days: int = None,
    years: int = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Fetch close prices from Refinitiv.
    
    Args:
        ric: Refinitiv Instrument Code (e.g., 'AAPL.O', 'VOD.L', 'EUR=')
        freq: Frequency string (1min, 5min, 1h, d, w, m, etc.)
        days: Number of days of history to pull
        years: Number of years of history to pull
        start_date: Explicit start date (YYYY-MM-DD)
        end_date: Explicit end date (YYYY-MM-DD or None for today)
    
    Returns:
        DataFrame with 'Exchange Date' and 'Close' columns.
    """
    # Determine date range
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    elif days:
        start_dt = end_dt - timedelta(days=days)
    elif years:
        start_dt = end_dt - timedelta(days=years * 365)
    else:
        # Default durations based on frequency
        if freq in ("d", "daily", "w", "weekly"):
            start_dt = end_dt - timedelta(days=365) # 1 Year for Daily/Weekly
        elif freq in ("m", "monthly", "q", "quarterly"):
            start_dt = end_dt - timedelta(days=5 * 365) # 5 Years for Monthly/Quarterly
        elif freq in ("y", "yearly"):
            start_dt = end_dt - timedelta(days=10 * 365)# 10 Years for Yearly
        else:
            start_dt = end_dt - timedelta(days=30)      # 30 Days for Intraday
    
    rd_interval = FREQ_TO_RD_INTERVAL.get(freq, "daily")
    
    print(f"\n{'='*60}")
    print(f"FETCHING DATA")
    print(f"{'='*60}")
    print(f"  RIC:        {ric}")
    print(f"  Frequency:  {freq} → Refinitiv interval: {rd_interval}")
    print(f"  Range:      {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    
    # --- Fetch via Refinitiv Data Library ---
    # First try: let the library return ALL available fields so we can see what's there
    df = None
    
    # Strategy 1: No fields specified (auto-discovery)
    try:
        print("\n[FETCH] Requesting data (auto-discovery)...")
        df = rd.get_history(
            universe=ric,
            interval=rd_interval,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"[WARN] Auto-discovery failed: {e}")
    
    # Strategy 2: Explicit fields if auto-discovery returned nothing useful
    if df is None or df.empty:
        for fields_attempt in [["CLOSE"], ["TRDPRC_1"], ["CLOSE", "TRDPRC_1"]]:
            try:
                print(f"[FETCH] Trying fields: {fields_attempt}...")
                df = rd.get_history(
                    universe=ric,
                    fields=fields_attempt,
                    interval=rd_interval,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                )
                if df is not None and not df.empty:
                    break
            except Exception:
                continue
    
    if df is None or df.empty:
        print("❌ No data returned. Check that:")
        print(f"   - '{ric}' is a valid RIC")
        print(f"   - Data exists for the requested period")
        print(f"   - Your Refinitiv license covers {rd_interval} data")
        return pd.DataFrame()
    
    # --- Debug: show what we got ---
    print(f"\n[DEBUG] Raw DataFrame shape: {df.shape}")
    print(f"[DEBUG] Columns returned: {df.columns.tolist()}")
    print(f"[DEBUG] Index name: {df.index.name}")
    print(f"[DEBUG] First 3 rows:")
    print(df.head(3).to_string())
    
    # --- Auto-detect close column ---
    close_col = None
    # Try exact matches first (case-insensitive)
    col_map = {c.upper(): c for c in df.columns}
    # Prioritized list of columns that usually represent the "Close" or "Last" price
    candidates = [
        "CLOSE", 
        "TRDPRC_1", 
        "MID_PRICE", 
        "PRICE", 
        "CLOSING PRICE", 
        "LAST", 
        "SETTLE",
        "MID"
    ]
    for candidate in candidates:
        if candidate in col_map:
            close_col = col_map[candidate]
            break
    
    # If still not found, try partial match
    if close_col is None:
        for c in df.columns:
            if "close" in c.lower() or "trdprc" in c.lower() or "last" in c.lower():
                close_col = c
                break
    
    # Last resort: use the first numeric column
    if close_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                close_col = c
                print(f"[WARN] No 'Close' column found, using first numeric column: '{close_col}'")
                break
    
    if close_col is None:
        print(f"❌ No usable price column found. Available columns: {df.columns.tolist()}")
        return pd.DataFrame()
    
    print(f"\n[INFO] Using column '{close_col}' as Close price")
    
    # Build clean output
    result = pd.DataFrame()
    result["Exchange Date"] = df.index
    result["Close"] = pd.to_numeric(df[close_col], errors="coerce").values
    result = result.dropna(subset=["Close"]).reset_index(drop=True)
    
    # Sort chronologically
    result = result.sort_values("Exchange Date").reset_index(drop=True)
    
    if len(result) == 0:
        print("❌ All rows were NaN after filtering. No usable data.")
        return pd.DataFrame()
    
    print(f"\n✅ Fetched {len(result)} data points")
    print(f"   First: {result['Exchange Date'].iloc[0]}")
    print(f"   Last:  {result['Exchange Date'].iloc[-1]}")
    print(f"   Close range: [{result['Close'].min():.4f} - {result['Close'].max():.4f}]")
    
    return result


def save_csv(df: pd.DataFrame, ric: str, freq: str, output_dir: str) -> str:
    """Save DataFrame to a FinCast-ready CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean RIC for filename (replace special chars)
    safe_ric = ric.replace(".", "_").replace("=", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_ric}_{freq}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    df.to_csv(filepath, index=False)
    print(f"\n📁 Saved to: {filepath}")
    
    # Also save as data.csv for direct FinCast use
    data_csv_path = os.path.join(output_dir, "data.csv")
    df.to_csv(data_csv_path, index=False)
    print(f"📁 Also saved as: {data_csv_path} (ready for FinCast)")
    
    # Print FinCast config hint
    fincast_freq = FREQ_TO_FINCAST.get(freq, "d")
    print(f"\n💡 FinCast config tip:")
    print(f'   config.data_frequency = "{fincast_freq}"')
    print(f'   config.data_path = r"{data_csv_path}"')
    
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Fetch close prices from Refinitiv for FinCast inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_data.py AAPL.O                            # daily, default 5y
  python fetch_data.py AAPL.O --freq 1min --days 30      # 1-min bars, 30 days
  python fetch_data.py MSFT.O --freq 1h --days 90        # hourly, 90 days
  python fetch_data.py VOD.L  --freq d --years 10        # daily, 10 years
  python fetch_data.py EUR=   --freq 5min --days 5       # FX 5-min, 5 days
  python fetch_data.py 0#.FCHI --freq d --years 5        # CAC 40 index

Supported frequencies:
  tick, 1min, 5min, 10min, 15min, 30min, 1h, d, w, m, q, y
        """,
    )
    
    parser.add_argument("ric", type=str, help="Refinitiv Instrument Code (e.g., AAPL.O, VOD.L, EUR=)")
    parser.add_argument("--freq", type=str, default="d", help="Frequency (1min, 5min, 1h, d, w, m) [default: d]")
    parser.add_argument("--days", type=int, default=None, help="Number of days of history")
    parser.add_argument("--years", type=int, default=None, help="Number of years of history")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=None, help="Output directory [default: Inference/]")
    
    args = parser.parse_args()
    
    # Default output dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(base_dir, "output")
    
    # Connect to Refinitiv
    connect()
    
    try:
        # Fetch data
        df = fetch_close_prices(
            ric=args.ric,
            freq=args.freq,
            days=args.days,
            years=args.years,
            start_date=args.start,
            end_date=args.end,
        )
        
        if df.empty:
            print("\n❌ No data fetched. Exiting.")
            sys.exit(1)
        
        # Save
        save_csv(df, args.ric, args.freq, output_dir)
        
    finally:
        rd.close_session()
        print("\n🔌 Refinitiv session closed.")


if __name__ == "__main__":
    main()
