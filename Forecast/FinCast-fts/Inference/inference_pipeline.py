"""
FinCast Inference Pipeline with Hyperparameter Optimization
============================================================
This script provides:
1. Data preprocessing for FinCast
2. Walk-forward backtesting for configuration evaluation
3. Optuna-based hyperparameter optimization to find the best config
4. Final inference with the optimal configuration

Usage:
    python inference_pipeline.py                    # Run with default config
    python inference_pipeline.py --optimize         # Find optimal config first
    python inference_pipeline.py --optimize --trials 50  # More optimization trials
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
from types import SimpleNamespace
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# Add src to path for FinCast imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_tools.TSdataset import TimeSeriesDataset_MultiCSV_train_Production
from tools.inference_utils import FinCast_Inference, plot_last_outputs, _save_outputs_to_csv, _slice_to_horizon, _pick_last_window_indices


# =============================================================================
# GPU CHECK & LIVE MONITOR
# =============================================================================

def check_gpu() -> bool:
    """
    Check if CUDA/GPU is available and display GPU information.
    Returns True if GPU is available and will be used.
    """
    print(f"\n{'='*60}")
    print("GPU STATUS CHECK")
    print(f"{'='*60}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available")
        print("   PyTorch cannot detect your GPU.")
        print("   Possible causes:")
        print("   - No NVIDIA GPU installed")
        print("   - CUDA drivers not installed")
        print("   - PyTorch installed without CUDA support")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"{'='*60}\n")
        return False
    
    # CUDA is available
    print("✅ CUDA is AVAILABLE")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA version: {torch.version.cuda}")
    
    # GPU details
    gpu_count = torch.cuda.device_count()
    print(f"   GPU count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024**3)  # Convert to GB
        print(f"\n   GPU {i}: {props.name}")
        print(f"      Memory: {total_mem:.1f} GB")
        print(f"      Compute capability: {props.major}.{props.minor}")
        print(f"      Multi-processors: {props.multi_processor_count}")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\n   Active GPU: {torch.cuda.get_device_name(current_device)}")
    
    # Memory usage
    allocated = torch.cuda.memory_allocated(current_device) / (1024**3)
    reserved = torch.cuda.memory_reserved(current_device) / (1024**3)
    print(f"   Memory allocated: {allocated:.2f} GB")
    print(f"   Memory reserved: {reserved:.2f} GB")
    
    print(f"{'='*60}\n")
    return True


class GPUMonitor:
    """
    Live GPU usage monitor that runs nvidia-smi in a background thread.
    Prints GPU utilization, memory usage, and temperature at a set interval.
    
    Usage:
        monitor = GPUMonitor(interval=3)
        monitor.start()
        # ... do GPU work ...
        monitor.stop()
    
    Or as a context manager:
        with GPUMonitor(interval=3):
            # ... do GPU work ...
    """
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._peak_gpu_util = 0
        self._peak_mem_used = 0
        self._samples = 0
        self._total_gpu_util = 0
    
    def _query_gpu(self) -> Optional[dict]:
        """Query nvidia-smi for current GPU stats."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) < 4:
                return None
            
            return {
                "gpu_util": int(parts[0]),
                "mem_used": int(parts[1]),
                "mem_total": int(parts[2]),
                "temp": int(parts[3]),
            }
        except Exception:
            return None
    
    def _monitor_loop(self):
        """Background thread loop."""
        while not self._stop_event.is_set():
            stats = self._query_gpu()
            if stats:
                self._samples += 1
                self._total_gpu_util += stats["gpu_util"]
                self._peak_gpu_util = max(self._peak_gpu_util, stats["gpu_util"])
                self._peak_mem_used = max(self._peak_mem_used, stats["mem_used"])
                
                mem_pct = (stats["mem_used"] / stats["mem_total"]) * 100
                bar_len = 20
                filled = int(bar_len * stats["gpu_util"] / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                print(
                    f"  [GPU] {bar} {stats['gpu_util']:3d}% | "
                    f"VRAM: {stats['mem_used']}/{stats['mem_total']} MiB ({mem_pct:.0f}%) | "
                    f"Temp: {stats['temp']}°C",
                    flush=True,
                )
            
            self._stop_event.wait(self.interval)
    
    def start(self):
        """Start the monitor thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._peak_gpu_util = 0
        self._peak_mem_used = 0
        self._samples = 0
        self._total_gpu_util = 0
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[GPU MONITOR] Started (polling every {self.interval}s)")
    
    def stop(self):
        """Stop the monitor thread and print summary."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 1)
        self._thread = None
        
        if self._samples > 0:
            avg_util = self._total_gpu_util / self._samples
            print(f"\n[GPU MONITOR] Summary:")
            print(f"  Avg GPU utilization: {avg_util:.0f}%")
            print(f"  Peak GPU utilization: {self._peak_gpu_util}%")
            print(f"  Peak VRAM usage: {self._peak_mem_used} MiB")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths - UPDATE THESE FOR YOUR SETUP
_INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(_INFERENCE_DIR, "output", "data.csv")
PROCESSED_DATA_PATH = os.path.join(_INFERENCE_DIR, "output", "data.csv")
MODEL_PATH = os.path.join(_INFERENCE_DIR, "v1.pth")
OUTPUT_PATH = os.path.join(_INFERENCE_DIR, "output")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "backend": "gpu",                # "cpu" or "gpu"
    "model_version": "v1",
    "data_frequency": "d",           # s/t/h/d/w/m/y
    "context_len": 128,              # 32-1024
    "horizon_len": 32,               # 1-256
    "all_data": False,
    "columns_target": ["Close"],
    "series_norm": False,
    "batch_size": 64,
    "forecast_mode": "mean",         # "mean" or "median"
    "quantile_outputs": [],
    "save_output": True,
    "plt_outputs": True,
    "plt_quantiles": [1, 3, 7, 9],
}

# Hyperparameter search space (min, max)
HP_SEARCH_SPACE = {
    "context_len": (16, 512),     # FinCast window (up to 800-1024)
    "horizon_len": (8, 128),     # Forecast steps
}


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def preprocess_data(raw_path: str, output_path: str) -> pd.DataFrame:
    """
    Preprocess raw CSV data for FinCast inference.
    
    Handles:
    - BOM encoding issues
    - European number format (semicolon separator, comma decimal)
    - Unnamed/empty columns
    - Missing dates (resampling to daily)
    - NaN interpolation
    """
    print(f"[PREPROCESS] Loading raw data from: {raw_path}")
    
    # Read with proper encoding to handle BOM
    df = pd.read_csv(raw_path, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"[PREPROCESS] Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Pre-processing to handle columns with units/symbols like '%'
    for col in df.columns:
        if col != 'Exchange Date' and df[col].dtype == 'object':
            try:
                df[col] = df[col].str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # Drop empty/unnamed columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    print(f"[PREPROCESS] After removing unnamed columns: {df.columns.tolist()}")
    
    # Parse and sort by date
    df['Exchange Date'] = pd.to_datetime(df['Exchange Date'], dayfirst=True)
    df = df.sort_values('Exchange Date')
    df = df.set_index('Exchange Date')
    
    # Resample to daily frequency
    df = df.resample('D').mean(numeric_only=True)
    
    # Interpolate missing values
    if 'Close' in df.columns:
        df['Close'] = df['Close'].interpolate(method='linear')
    df = df.interpolate(method='linear').ffill().bfill()
    
    df = df.reset_index()
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"[PREPROCESS] Saved {len(df)} rows to: {output_path}")
    
    return df


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================

def create_config(
    data_path: str,
    model_path: str,
    output_path: str,
    context_len: int = 128,
    horizon_len: int = 32,
    **kwargs
) -> SimpleNamespace:
    """Create a FinCast configuration object."""
    config = SimpleNamespace()
    
    # Required paths
    config.data_path = data_path
    config.model_path = model_path
    config.save_output_path = output_path
    
    # Merge defaults with overrides
    params = {**DEFAULT_CONFIG, **kwargs}
    params["context_len"] = context_len
    params["horizon_len"] = horizon_len
    
    for key, value in params.items():
        setattr(config, key, value)
    
    return config


def run_inference(config: SimpleNamespace) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Run FinCast inference with the given configuration."""
    print(f"\n[INFERENCE] Running with context_len={config.context_len}, horizon_len={config.horizon_len}")
    
    fincast = FinCast_Inference(config)
    preds, mapping, full_outputs = fincast.run_inference()
    
    print(f"[INFERENCE] Generated forecasts: {preds.shape}")
    
    return preds, mapping, full_outputs, fincast


# =============================================================================
# WALK-FORWARD BACKTESTING
# =============================================================================

def walk_forward_backtest(
    data: pd.DataFrame,
    model_path: str,
    context_len: int,
    horizon_len: int,
    n_folds: int = 3,
    target_col: str = "Close",
    temp_dir: str = None,
) -> Dict[str, float]:
    """
    Perform walk-forward validation to evaluate a configuration.
    
    This splits the data into training/test folds:
    - Fold 1: train on [0:T-2H], test on [T-2H:T-H]
    - Fold 2: train on [0:T-H], test on [T-H:T]
    - etc.
    
    Returns metrics: MAE, RMSE, MAPE
    """
    if temp_dir is None:
        temp_dir = os.path.dirname(PROCESSED_DATA_PATH)
    
    close_values = data[target_col].values
    total_len = len(close_values)
    
    all_maes = []
    all_rmses = []
    all_mapes = []
    
    for fold in range(n_folds):
        # Calculate split points
        test_end = total_len - fold * horizon_len
        test_start = test_end - horizon_len
        train_end = test_start
        
        if train_end < context_len + horizon_len:
            print(f"[BACKTEST] Fold {fold+1}: Not enough data, skipping")
            continue
        
        # Create temporary CSV with training data only
        train_df = data.iloc[:train_end].copy()
        temp_csv = os.path.join(temp_dir, f"_backtest_temp_{fold}.csv")
        train_df.to_csv(temp_csv, index=False)
        
        try:
            # Run inference on training data
            config = create_config(
                data_path=temp_csv,
                model_path=model_path,
                output_path=temp_dir,
                context_len=context_len,
                horizon_len=horizon_len,
                save_output=False,
                plt_outputs=False,
            )
            
            preds, _, _, _ = run_inference(config)
            
            # Get the forecast (last row, all horizons)
            forecast = preds[0, :horizon_len]
            
            # Get actual values
            actual = close_values[test_start:test_end]
            
            # Ensure same length
            min_len = min(len(forecast), len(actual))
            forecast = forecast[:min_len]
            actual = actual[:min_len]
            
            # Calculate metrics
            mae = np.mean(np.abs(forecast - actual))
            rmse = np.sqrt(np.mean((forecast - actual) ** 2))
            mape = np.mean(np.abs((actual - forecast) / actual)) * 100
            
            all_maes.append(mae)
            all_rmses.append(rmse)
            all_mapes.append(mape)
            
            print(f"[BACKTEST] Fold {fold+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            
        except Exception as e:
            print(f"[BACKTEST] Fold {fold+1} failed: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
    
    if not all_maes:
        return {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}
    
    return {
        "mae": np.mean(all_maes),
        "rmse": np.mean(all_rmses),
        "mape": np.mean(all_mapes),
    }


# =============================================================================
# HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# =============================================================================

def optimize_config(
    data: pd.DataFrame,
    model_path: str,
    n_trials: int = 20,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """
    Use Optuna to find the optimal (context_len, horizon_len) configuration.
    
    Optuna uses Bayesian optimization (TPE sampler) to efficiently search
    the hyperparameter space.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("[ERROR] Optuna not installed. Install with: pip install optuna")
        print("[FALLBACK] Running grid search instead...")
        return grid_search_config(data, model_path, n_folds)
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print(f"{'='*60}")
    print(f"Search space:")
    print(f"  context_len: {HP_SEARCH_SPACE['context_len']}")
    print(f"  horizon_len: {HP_SEARCH_SPACE['horizon_len']}")
    print(f"Trials: {n_trials}, Folds per trial: {n_folds}")
    print(f"{'='*60}\n")
    
    def objective(trial: optuna.Trial) -> float:
        # Hardcore search: use integer ranges with steps for better discovery
        context_len = trial.suggest_int("context_len", HP_SEARCH_SPACE["context_len"][0], HP_SEARCH_SPACE["context_len"][1], step=16)
        horizon_len = trial.suggest_int("horizon_len", HP_SEARCH_SPACE["horizon_len"][0], HP_SEARCH_SPACE["horizon_len"][1], step=4)
        
        # Validate we have enough data
        if len(data) < context_len + horizon_len * (n_folds + 1):
            return float("inf")
        
        metrics = walk_forward_backtest(
            data=data,
            model_path=model_path,
            context_len=context_len,
            horizon_len=horizon_len,
            n_folds=n_folds,
        )
        
        # Optimize for MAE
        return metrics["mae"]
    
    # Create study with TPE sampler (Bayesian optimization)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="fincast_config_optimization"
    )
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Results
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Best MAE: {best_value:.4f}")
    print(f"Best config:")
    print(f"  context_len: {best_params['context_len']}")
    print(f"  horizon_len: {best_params['horizon_len']}")
    print(f"{'='*60}\n")
    
    # Show all trials
    print("All trials (sorted by MAE):")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value")
    for _, row in trials_df.head(10).iterrows():
        print(f"  context={int(row['params_context_len']):4d}, "
              f"horizon={int(row['params_horizon_len']):3d} → MAE={row['value']:.4f}")
    
    return best_params


def grid_search_config(
    data: pd.DataFrame,
    model_path: str,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """
    Fallback grid search if Optuna is not available.
    """
    print(f"\n{'='*60}")
    print("GRID SEARCH HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*60}")
    
    results = []
    
    for context_len in HP_SEARCH_SPACE["context_len"]:
        for horizon_len in HP_SEARCH_SPACE["horizon_len"]:
            if len(data) < context_len + horizon_len * (n_folds + 1):
                print(f"[SKIP] context={context_len}, horizon={horizon_len} - not enough data")
                continue
            
            print(f"\n[EVAL] context_len={context_len}, horizon_len={horizon_len}")
            metrics = walk_forward_backtest(
                data=data,
                model_path=model_path,
                context_len=context_len,
                horizon_len=horizon_len,
                n_folds=n_folds,
            )
            
            results.append({
                "context_len": context_len,
                "horizon_len": horizon_len,
                **metrics
            })
    
    # Find best
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    best = results_df.iloc[0]
    best_params = {
        "context_len": int(best["context_len"]),
        "horizon_len": int(best["horizon_len"]),
    }
    
    print(f"\nBest config: context_len={best_params['context_len']}, horizon_len={best_params['horizon_len']}")
    print(f"Best MAE: {best['mae']:.4f}")
    
    return best_params


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FinCast Inference Pipeline")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization first")
    parser.add_argument("--trials", type=int, default=20, help="Number of optimization trials (default: 20)")
    parser.add_argument("--folds", type=int, default=3, help="Number of backtesting folds (default: 3)")
    parser.add_argument("--context-len", type=int, default=None, help="Override context length")
    parser.add_argument("--horizon-len", type=int, default=None, help="Override horizon length")
    parser.add_argument("--freq", type=str, default=None, help="Data frequency (s, t, h, d, w, m, y)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for all outputs")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FINCAST INFERENCE PIPELINE")
    print("=" * 60)
    
    # Override paths if output-dir is provided
    global OUTPUT_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
    if args.output_dir:
        OUTPUT_PATH = args.output_dir
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        RAW_DATA_PATH = os.path.join(OUTPUT_PATH, "data.csv")
        PROCESSED_DATA_PATH = os.path.join(OUTPUT_PATH, "data.csv")
    
    # Check GPU status
    gpu_available = check_gpu()
    if args.cpu:
        print("[INFO] Running on CPU (--cpu flag specified)")
    elif not gpu_available:
        print("[WARNING] GPU not available, falling back to CPU")
    
    # Create GPU monitor (polls every 5s)
    gpu_monitor = GPUMonitor(interval=5.0) if (gpu_available and not args.cpu) else None
    
    # Step 1: Preprocess data
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(RAW_DATA_PATH):
        if os.path.exists(RAW_DATA_PATH):
            data = preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
        else:
            print(f"[ERROR] Raw data not found: {RAW_DATA_PATH}")
            sys.exit(1)
    else:
        print(f"[INFO] Using existing processed data: {PROCESSED_DATA_PATH}")
        data = pd.read_csv(PROCESSED_DATA_PATH)
    
    print(f"[INFO] Data shape: {data.shape}")
    print(f"[INFO] Date range: {data['Exchange Date'].min()} to {data['Exchange Date'].max()}")
    
    # Step 2: Determine configuration
    context_len = args.context_len or DEFAULT_CONFIG["context_len"]
    horizon_len = args.horizon_len or DEFAULT_CONFIG["horizon_len"]
    
    if args.optimize:
        print("\n[OPTIMIZE] Finding optimal configuration...")
        if gpu_monitor:
            gpu_monitor.start()
        best_params = optimize_config(
            data=data,
            model_path=MODEL_PATH,
            n_trials=args.trials,
            n_folds=args.folds,
        )
        if gpu_monitor:
            gpu_monitor.stop()
        context_len = best_params["context_len"]
        horizon_len = best_params["horizon_len"]
        
        # Save best config
        config_file = os.path.join(OUTPUT_PATH, "best_config.json")
        with open(config_file, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"[OPTIMIZE] Saved best config to: {config_file}")
    else:
        print("\n[INFO] Skipping optimization (use --optimize to enable)")
    
    # Step 3: Run final inference
    print(f"\n[INFERENCE] Running with config:")
    print(f"  context_len: {context_len}")
    print(f"  horizon_len: {horizon_len}")
    print(f"  frequency:   {args.freq or DEFAULT_CONFIG['data_frequency']}")
    
    config = create_config(
        data_path=PROCESSED_DATA_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        context_len=context_len,
        horizon_len=horizon_len,
        data_frequency=args.freq or DEFAULT_CONFIG["data_frequency"],
        backend="cpu" if args.cpu else "gpu",
        plt_outputs=False, # Handled by pretty_show_forecast.py
    )
    
    try:
        preds, mapping, full_outputs, fincast = run_inference(config)
    except ValueError as e:
        if "too short" in str(e):
            print(f"\n❌ DATA ERROR: {e}")
            print(f"   The fetched history ({data.shape[0]} points) is shorter than the context length ({context_len}).")
            print("   👉 Solution: Fetch more data! (e.g. increase --days or --years in run.py)")
            sys.exit(1)
        else:
            raise e
    
    # Step 4: Save forecast CSVs
    H = horizon_len
    mean_sliced = _slice_to_horizon(preds, H)
    full_sliced = _slice_to_horizon(full_outputs, H)
    pick_idx = _pick_last_window_indices(mapping)
    mean_sel = mean_sliced[pick_idx, :]
    full_sel = None if full_sliced is None else full_sliced[pick_idx, :, :]
    mean_csv, full_csv = _save_outputs_to_csv(
        mean_sel, full_sel, mapping.iloc[pick_idx], OUTPUT_PATH, prefix="fincast"
    )
    print(f"[SAVED] Mean forecast → {mean_csv}")
    if full_csv:
        print(f"[SAVED] Full forecast → {full_csv}")
    
    # Step 5: Display forecast summary
    print(f"\n{'='*60}")
    print("FORECAST SUMMARY")
    print(f"{'='*60}")
    print(f"Target: {config.columns_target}")
    print(f"Horizon: {horizon_len} days")
    print(f"\nForecast values:")
    
    forecast = preds[0, :]
    current_price = data['Close'].iloc[-1]
    
    for i in [0, 4, 9, 14, 19, 29, min(31, horizon_len-1)]:
        if i < len(forecast):
            change = (forecast[i] / current_price - 1) * 100
            print(f"  Day {i+1:2d}: {forecast[i]:.4f} ({change:+.2f}%)")
    
    print(f"\nCurrent price: {current_price:.4f}")
    print(f"End forecast:  {forecast[-1]:.4f} ({(forecast[-1]/current_price-1)*100:+.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
