"""
FinOracle Bridge Script
========================
This script runs INSIDE the FinCast .venv (which has torch, etc.).
It receives a JSON config on stdin, runs the FinCast inference pipeline,
and outputs a JSON result on stdout.

Called by finoracle_wrapper.py via subprocess.
"""

import sys
import os
import json
import traceback
import warnings

# Suppress annoying warnings from Refinitiv/Pandas
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# ─── CRITICAL: Redirect stdout → stderr BEFORE any imports ───────────
# FinCast prints "Loaded PyTorch FinCast..." at import time, which would
# pollute stdout and break JSON parsing. Only our final JSON goes to real stdout.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

# Add FinCast src to path
INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(INFERENCE_DIR, "..", "src"))

import numpy as np
import pandas as pd
import torch

from inference_pipeline import (
    create_config, run_inference, optimize_config, DEFAULT_CONFIG,
    _slice_to_horizon, _pick_last_window_indices, _save_outputs_to_csv,
)
from fetch_data import fetch_close_prices


def _json_safe(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _resolve_ric(ric: str) -> str:
    """
    Resolve a potentially wrong RIC using the Refinitiv Symbol Conversion API.
    If the RIC already has a dot (e.g., 'AAPL.O'), try it first via API validation.
    If it fails, try the Symbol Conversion API or common exchange suffixes.
    """
    import refinitiv.data as rd

    # If it already looks like a proper RIC with a dot, validate it first
    # by trying a quick historical data probe
    if "." in ric or "=" in ric:
        try:
            test = rd.get_history(universe=ric, interval="daily", count=1)
            if test is not None and not test.empty:
                print(f"   [RIC] {ric} validated OK")
                return ric
        except Exception:
            pass
        print(f"   [RIC] {ric} not found, trying symbol conversion...")

    # Extract the base ticker (strip any exchange suffix)
    base_ticker = ric.split(".")[0].split("=")[0]

    # Try Refinitiv Symbol Conversion API
    try:
        from refinitiv.data.content import symbol_conversion
        response = symbol_conversion.Definition(
            symbols=[base_ticker],
            from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
            to_symbol_types=[symbol_conversion.SymbolTypes.RIC],
        ).get_data()
        df = response.data.df
        if not df.empty and 'RIC' in df.columns:
            resolved = df['RIC'].iloc[0]
            if resolved and isinstance(resolved, str):
                print(f"   [RIC] Resolved via API: {base_ticker} -> {resolved}")
                return resolved
    except Exception as e:
        print(f"   [RIC] Symbol conversion API failed: {e}")

    # Fallback: try common exchange suffixes
    suffixes = [".N", ".O", ".PA", ".L", ".DE", ".TO", ".AX", ".HK", ".T"]
    for suffix in suffixes:
        candidate = base_ticker + suffix
        try:
            test = rd.get_history(universe=candidate, interval="daily", count=1)
            if test is not None and not test.empty:
                print(f"   [RIC] Found via probe: {base_ticker} -> {candidate}")
                return candidate
        except Exception:
            continue

    print(f"   [RIC] Could not resolve {base_ticker}, using original: {ric}")
    return ric


def run_bridge(cfg: dict) -> dict:
    """
    Main bridge logic — mirrors the old FinOracleWrapper.run_forecast().
    
    Args:
        cfg: dict with keys matching AnalysisConfig finoracle_* fields plus:
             - ticker, ric, output_dir, model_path
    
    Returns:
        dict with forecast results or error.
    """
    ticker = cfg["ticker"]
    ric = cfg["ric"]
    output_dir = cfg["output_dir"]
    model_path = cfg.get("model_path", os.path.join(INFERENCE_DIR, "v1.pth"))

    # Check model
    if not os.path.exists(model_path):
        return {"error": f"Model weights not found at: {model_path}"}

    # Setup
    finoracle_dir = os.path.join(output_dir, "finoracle")
    os.makedirs(finoracle_dir, exist_ok=True)
    raw_data_path = os.path.join(finoracle_dir, "data.csv")

    # --- 1. Fetch Data ---
    skip_fetch = cfg.get("skip_fetch", False)
    if skip_fetch and os.path.exists(raw_data_path):
        print(f"   [Data] Skipping fetch - reusing {raw_data_path}")
        df = pd.read_csv(raw_data_path)
    else:
        # Open Refinitiv session before fetching
        from fetch_data import connect
        connect()

        # Resolve RIC using Refinitiv API (the FinCast venv has refinitiv-data)
        ric = _resolve_ric(ric)

        fetch_kwargs = {"ric": ric, "freq": cfg.get("freq", "d")}
        days = cfg.get("days")
        start = cfg.get("start")
        end = cfg.get("end")
        years = cfg.get("years", 5)

        if days is not None:
            fetch_kwargs["days"] = days
        elif start is not None:
            fetch_kwargs["start"] = start
            if end is not None:
                fetch_kwargs["end"] = end
        else:
            fetch_kwargs["days"] = years * 365

        df = fetch_close_prices(**fetch_kwargs)
        if df.empty:
            return {"error": f"No data fetched for {ric}"}
        df.to_csv(raw_data_path, index=False)
        print(f"   [Data] Fetched {len(df)} rows. Saved to {raw_data_path}")

    # --- 2. Skip inference? ---
    if cfg.get("skip_inference", False):
        print("   [Skip] Inference skipped.")
        import glob
        existing = glob.glob(os.path.join(finoracle_dir, "finoracle_*_full_*.csv"))
        if existing:
            return {
                "status": "skipped",
                "ticker": ticker,
                "ric": ric,
                "output_dir": finoracle_dir,
                "csv_path": existing[0],
            }
        return {"error": "skip_inference set but no existing results found"}

    # --- 3. Configure ---
    context_len = cfg.get("context_len", 128)
    horizon_len = cfg.get("horizon_len", 32)

    if cfg.get("optimize", False):
        trials = cfg.get("trials", 20)
        folds = cfg.get("folds", 3)
        print(f"   [Opt] Running optimization (Trials={trials}, Folds={folds})...")
        best_params = optimize_config(
            data=df, model_path=model_path,
            n_trials=trials, n_folds=folds,
        )
        context_len = best_params["context_len"]
        horizon_len = best_params["horizon_len"]
        print(f"   [Opt] Best Params: L={context_len}, H={horizon_len}")

    use_gpu = cfg.get("use_gpu", True)
    backend = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"

    fc_config = create_config(
        data_path=raw_data_path,
        model_path=model_path,
        output_path=finoracle_dir,
        context_len=context_len,
        horizon_len=horizon_len,
        data_frequency=cfg.get("freq", "d"),
        backend=backend,
        save_output=True,
        plt_outputs=True,
    )

    # --- 4. Run Inference ---
    preds, mapping, full_outputs, _ = run_inference(fc_config)

    H = horizon_len
    mean_sliced = _slice_to_horizon(preds, H)
    full_sliced = _slice_to_horizon(full_outputs, H)
    pick_idx = _pick_last_window_indices(mapping)

    mean_sel = mean_sliced[pick_idx, :]
    full_sel = None if full_sliced is None else full_sliced[pick_idx, :, :]

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"finoracle_{ticker}"

    mean_csv, full_csv = _save_outputs_to_csv(
        mean_sel, full_sel, mapping.iloc[pick_idx], finoracle_dir, prefix=prefix
    )

    # --- 5. Plot ---
    try:
        sys.path.insert(0, INFERENCE_DIR)
        from pretty_show_forecast import (
            plot_forecast_only, plot_context_and_forecast,
            plot_zoomed_forecast, _load_forecast,
        )
        forecast_target_csv = full_csv if full_csv else mean_csv
        row_name, forecast_df, _ = _load_forecast(forecast_target_csv)

        plot_forecast_only(
            f"{ticker} Forecast", forecast_df,
            os.path.join(finoracle_dir, f"{prefix}_forecast.png"),
        )
        plot_context_and_forecast(
            f"{ticker} Context", forecast_df, raw_data_path,
            os.path.join(finoracle_dir, f"{prefix}_context.png"),
            context_len,
        )
        plot_zoomed_forecast(
            f"{ticker} Zoomed", forecast_df, raw_data_path,
            os.path.join(finoracle_dir, f"{prefix}_zoomed.png"),
        )
        print(f"   [Plot] Plots saved to {finoracle_dir}")
    except ImportError:
        print("   [Warn] pretty_show_forecast not found, skipping plot.")
    except Exception as e:
        print(f"   [Warn] Plotting failed: {e}")

    # --- 6. Summary ---
    forecast_mean = preds[0, :H]
    last_price = float(df["Close"].iloc[-1])
    final_pred = float(forecast_mean[-1])
    ret = (final_pred / last_price) - 1

    # Save run_config.txt
    config_path = os.path.join(finoracle_dir, "run_config.txt")
    with open(config_path, "w") as f:
        # Save bridge config + resolved RIC + final metrics
        final_cfg = cfg.copy()
        final_cfg.update({
            "resolved_ric": ric,
            "context_used": int(context_len),
            "horizon_used": int(horizon_len),
            "model_used": model_path
        })
        json.dump(final_cfg, f, indent=4, default=_json_safe)
    print(f"   [Config] Saved run config to {config_path}")

    return {
        "status": "success",
        "ticker": ticker,
        "ric": ric,
        "current_price": last_price,
        "forecast_price": final_pred,
        "expected_return": ret,
        "horizon_days": int(horizon_len),
        "context_used": int(context_len),
        "output_dir": finoracle_dir,
        "csv_path": full_csv if full_csv else mean_csv,
    }


# ─────────────────────────────────────────────────────────────────────
# Entry point — read JSON from stdin, write JSON to stdout
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        cfg = json.load(sys.stdin)
        result = run_bridge(cfg)
    except Exception as e:
        traceback.print_exc()
        result = {"error": str(e)}
    finally:
        # Close Refinitiv session if it was opened
        try:
            import refinitiv.data as rd
            rd.close_session()
        except Exception:
            pass

    # Restore stdout and write the ONLY line of real output
    sys.stdout = _real_stdout
    print(json.dumps(result, default=_json_safe))
