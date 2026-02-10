"""
FinOracle Utilities
===================
Helper functions for FinOracle integration.
"""

import sys

def get_ric(ticker: str) -> str:
    """
    Convert a standard ticker (e.g., 'AAPL', 'MSFT', 'EUR=') to a Refinitiv RIC (e.g., 'AAPL.O').
    
    Uses the Refinitiv Data Library's Symbol Conversion if available.
    Falls back to simple heuristics if the library is not installed or the lookup fails.
    """
    # 1. Try Refinitiv Symbol Conversion
    try:
        import refinitiv.data.content.symbol_conversion as symbol_conversion
        import refinitiv.data as rd
        
        # Ensure session is open (it might already be open by fetch_data, but good to check)
        try:
            rd.get_session()
        except:
            # If no session, try to open one - assumes desktop/workspace is running
            # However, we don't want to crash here if it fails, just fallback.
            try:
                rd.open_session()
            except:
                pass

        print(f"[FinOracle] Converting '{ticker}' to RIC via Refinitiv API...")
        response = symbol_conversion.Definition(
            symbols=[ticker],
            from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
            to_symbol_types=[symbol_conversion.SymbolTypes.RIC],
        ).get_data()
        
        df = response.data.df
        if not df.empty and 'RIC' in df.columns:
            ric = df['RIC'].iloc[0]
            if ric and isinstance(ric, str):
                print(f"[FinOracle] Resolved: {ticker} -> {ric}")
                return ric
                
    except ImportError:
        # Refinitiv library not found in ECE env -- this is expected.
        # The bridge script will handle resolution if needed.
        pass
    except Exception as e:
        print(f"[FinOracle] Symbol conversion failed: {e}. Using heuristics.")

    # 2. Heuristic Fallbacks
    # Already a RIC? (Has a dot or is a known format like 'EUR=')
    if "." in ticker or "=" in ticker:
        return ticker
        
    # Common Suffixes
    # Note: This is a simplified fallback. Ideally, the API should work.
    return f"{ticker}.O"  # Default to US implementation
