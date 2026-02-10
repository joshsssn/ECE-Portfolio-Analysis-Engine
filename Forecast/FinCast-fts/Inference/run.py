"""
FinCast Full Pipeline — Fetch data, run inference, show results.

Usage:
    python run.py AAPL.O                                      # daily, 5y, default config
    python run.py AAPL.O --freq 1min --days 30 -L 128 -H 16   # 1-min, 30 days
    python run.py MSFT.O --freq d --years 10 -L 256 -H 32     # daily, 10 years
    python run.py EUR=   --freq 1h --days 30 -L 128 -H 16     # hourly FX
    python run.py AAPL.O --optimize --trials 12                # with optimization
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FETCH_SCRIPT = os.path.join(SCRIPT_DIR, "fetch_data.py")
INFERENCE_SCRIPT = os.path.join(SCRIPT_DIR, "inference_pipeline.py")
PRETTY_SCRIPT = os.path.join(SCRIPT_DIR, "pretty_show_forecast.py")

# Map fetch_data freq codes to FinCast freq codes
FREQ_TO_FINCAST = {
    "tick": "s",
    "1min": "t", "5min": "t", "10min": "t", "15min": "t", "30min": "t",
    "1h": "h",
    "d": "d", "daily": "d",
    "w": "w", "weekly": "w",
    "m": "m", "monthly": "m",
    "q": "q", "quarterly": "q",
    "y": "y", "yearly": "y",
}


def run_step(label: str, cmd: list[str]) -> int:
    """Run a subprocess with live output."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(SCRIPT_DIR))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="FinCast Full Pipeline: Fetch → Inference → Visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py AAPL.O                                      # daily, default
  python run.py AAPL.O --freq 1min --days 30 -L 128 -H 16   # 1-min bars
  python run.py MSFT.O --freq d --years 10 -L 256 -H 32     # daily, 10y
  python run.py EUR=   --freq 1h --days 30 -L 128 -H 16     # hourly FX
  python run.py AAPL.O --optimize --trials 12                # with optimization
""",
    )

    # --- Data Fetching args ---
    parser.add_argument("ric", type=str, help="Refinitiv Instrument Code (e.g., AAPL.O, VOD.L, EUR=)")
    parser.add_argument("--freq", type=str, default="d", help="Data frequency (1min, 5min, 1h, d, w, m) [default: d]")
    parser.add_argument("--days", type=int, default=None, help="Number of days of history")
    parser.add_argument("--years", type=int, default=None, help="Number of years of history")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")

    # --- Inference args ---
    parser.add_argument("-L", "--context-len", type=int, default=128, help="Context length [default: 128]")
    parser.add_argument("-H", "--horizon-len", type=int, default=16, help="Horizon length [default: 16]")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=20, help="Number of optimization trials [default: 20]")
    parser.add_argument("--folds", type=int, default=3, help="Number of backtesting folds [default: 3]")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of GPU")

    # --- Pipeline control ---
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching (use existing data.csv)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference (just show latest results)")

    args = parser.parse_args()

    # Resolve Python executable (prefer venv)
    python = sys.executable

    # Derive FinCast frequency from fetch frequency
    fincast_freq = FREQ_TO_FINCAST.get(args.freq, "d")

    print("=" * 60)
    print("  FINCAST FULL PIPELINE")
    print("=" * 60)
    print(f"  RIC:          {args.ric}")
    print(f"  Frequency:    {args.freq} → FinCast freq: {fincast_freq}")
    print(f"  Context (L):  {args.context_len}")
    print(f"  Horizon (H):  {args.horizon_len}")
    print(f"  Optimize:     {'Yes' if args.optimize else 'No'}")
    print(f"  Device:       {'CPU' if args.cpu else 'GPU'}")
    print("=" * 60)

    # ─── STEP 0: Prepare Run Context & Config ────────────────────
    # Create unique folder: Inference/output/RIC_FREQ_TIMESTAMP
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ric = args.ric.replace(".", "_").replace("=", "_").replace("/", "_")
    run_folder_name = f"{safe_ric}_{args.freq}_{timestamp}"
    
    # Base output dir (Inference/output)
    base_output = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(base_output, exist_ok=True)
    
    # Specific run dir
    run_dir = os.path.join(base_output, run_folder_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"📂 Run Directory: {run_dir}")
    
    # Save Config
    config_file = os.path.join(run_dir, "run_config.txt")
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  FINCAST RUN CONFIGURATION ({timestamp})\n")
        f.write("=" * 60 + "\n\n")
        for arg, value in vars(args).items():
            f.write(f"{arg:<20}: {value}\n")
        f.write("\nDerived Values:\n")
        f.write(f"{'fincast_freq':<20}: {fincast_freq}\n")
        f.write(f"{'python_exe':<20}: {python}\n")
        f.write(f"{'run_dir':<20}: {run_dir}\n")
    print(f"📄 Config saved to: {config_file}\n")

    # ─── STEP 1: Fetch Data ──────────────────────────────────────
    if not args.skip_fetch:
        # Pass --output run_dir so data.csv lands there
        fetch_cmd = [python, FETCH_SCRIPT, args.ric, "--freq", args.freq, "--output", run_dir]
        if args.days:
            fetch_cmd += ["--days", str(args.days)]
        if args.years:
            fetch_cmd += ["--years", str(args.years)]
        if args.start:
            fetch_cmd += ["--start", args.start]
        if args.end:
            fetch_cmd += ["--end", args.end]

        rc = run_step("STEP 1/3: FETCHING DATA FROM REFINITIV", fetch_cmd)
        if rc != 0:
            print("\n❌ Data fetch failed. Aborting.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping data fetch (--skip-fetch)")
        # Note: If skipping fetch, we assume data.csv is already IN the run_dir? 
        # Actually, if the user skips fetch, they probably want to reuse an OLD run's data.
        # But for now, --skip-fetch implies "I manually put data.csv there" or "I am re-running in same folder".
        # To make it robust, if skipping fetch, we should probably COPY the latest data.csv from somewhere, 
        # but let's keep it simple: assume data.csv exists in run_dir OR user manually copied it.
        # Wait, usually users use --skip-fetch to re-run inference on the SAME data. 
        # But we just created a NEW empty folder. So data.csv won't be there.
        # FIX: If skipping fetch, try to find the LATEST data.csv in `output/` and copy it to `run_dir`.
        
        latest_data = None
        # walk output dir to find latest data.csv
        all_data_files = [] 
        for root, dirs, files in os.walk(base_output):
            if "data.csv" in files:
                all_data_files.append(os.path.join(root, "data.csv"))
        
        if all_data_files:
            latest_data = max(all_data_files, key=os.path.getmtime)
            import shutil
            shutil.copy(latest_data, os.path.join(run_dir, "data.csv"))
            print(f"   (Copied data.csv from latest run: {os.path.dirname(latest_data)})")
        else:
            print("⚠️  Warning: --skip-fetch used but no previous data.csv found to copy.")

    # ─── STEP 2: Run Inference ───────────────────────────────────
    if not args.skip_inference:
        # Pass --output-dir run_dir
        inference_cmd = [
            python, INFERENCE_SCRIPT,
            "--context-len", str(args.context_len),
            "--horizon-len", str(args.horizon_len),
            "--freq", fincast_freq,
            "--output-dir", run_dir
        ]
        if args.optimize:
            inference_cmd += ["--optimize", "--trials", str(args.trials), "--folds", str(args.folds)]
        if args.cpu:
            inference_cmd.append("--cpu")

        rc = run_step("STEP 2/3: RUNNING FINCAST INFERENCE", inference_cmd)
        if rc != 0:
            print("\n❌ Inference failed. Aborting.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping inference (--skip-inference)")

    # ─── STEP 3: Pretty Show Results ─────────────────────────────
    # Pass run_dir so it finds the csv inside it
    pretty_cmd = [python, PRETTY_SCRIPT, run_dir]
    rc = run_step("STEP 3/3: VISUALIZING RESULTS", pretty_cmd)
    if rc != 0:
        print("\n⚠️  Visualization had issues, but results may still be saved.")

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print(f"  👉 Results at: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
