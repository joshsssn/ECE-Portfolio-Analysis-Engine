"""
FinOracle Wrapper
=================
Singleton wrapper for the FinCast forecasting engine.
Runs FinCast via subprocess using the FinCast .venv (which has torch, etc.).
"""

import os
import sys
import json
import subprocess
from typing import Optional, Dict, Any
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────
FINCAST_DIR = os.path.join(os.path.dirname(__file__), "Forecast", "FinCast-fts", "Inference")
FINCAST_VENV_PYTHON = os.path.join(
    os.path.dirname(__file__), "Forecast", "FinCast-fts", ".venv", "Scripts", "python.exe"
)
BRIDGE_SCRIPT = os.path.join(FINCAST_DIR, "finoracle_bridge.py")


class FinOracleWrapper:
    _instance = None
    _model_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FinOracleWrapper, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the wrapper — validate that the bridge can run."""
        self.model_path = os.path.join(FINCAST_DIR, "v1.pth")
        
        # Check venv Python exists
        if not os.path.exists(FINCAST_VENV_PYTHON):
            print(f"[FinOracle] Error: FinCast venv Python not found at {FINCAST_VENV_PYTHON}")
            self.engine_available = False
            self._unavail_reason = f"FinCast venv not found at: {FINCAST_VENV_PYTHON}"
            return
        
        # Check bridge script exists
        if not os.path.exists(BRIDGE_SCRIPT):
            print(f"[FinOracle] Error: Bridge script not found at {BRIDGE_SCRIPT}")
            self.engine_available = False
            self._unavail_reason = f"Bridge script not found at: {BRIDGE_SCRIPT}"
            return
        
        # Check model weights
        if not os.path.exists(self.model_path):
            print(f"[FinOracle] Error: Model not found at {self.model_path}")
            self.engine_available = False
            self._unavail_reason = f"Model weights not found at: {self.model_path}"
            return
        
        self.engine_available = True
        self._unavail_reason = None
        print(f"[FinOracle] Engine ready (subprocess mode)")
        print(f"   Python : {FINCAST_VENV_PYTHON}")
        print(f"   Bridge : {BRIDGE_SCRIPT}")
        print(f"   Model  : {self.model_path}")
            
    def run_forecast(self, 
                     ticker: str, 
                     ric: str,
                     output_dir: Path, 
                     config: Any) -> Dict[str, Any]:
        """
        Run the full FinOracle forecasting pipeline for a ticker.
        
        Delegates to finoracle_bridge.py running in the FinCast .venv
        via subprocess, passing config as JSON and reading results back.
        """
        if not self.engine_available:
            print("=" * 60)
            print("[FinOracle] ERROR: Engine not available!")
            print(f"   Reason: {self._unavail_reason}")
            print("=" * 60)
            return {'error': 'FinCast engine not available'}

        print(f"\n" + "="*60)
        print(f"[FinOracle] Forecasting for {ticker} ({ric})")
        print("="*60)

        # Force absolute path for output_dir so the bridge (running in a different CWD)
        # saves to the correct location relative to ECE root.
        abs_output_dir = os.path.abspath(str(output_dir))

        # Build the config dict to send to the bridge
        bridge_cfg = {
            "ticker": ticker,
            "ric": ric,
            "output_dir": abs_output_dir,
            "model_path": self.model_path,
            # Map config.finoracle_* fields to bridge keys
            "freq": getattr(config, "finoracle_freq", "d"),
            "days": getattr(config, "finoracle_days", None),
            "years": getattr(config, "finoracle_years", 5),
            "start": getattr(config, "finoracle_start", None),
            "end": getattr(config, "finoracle_end", None),
            "skip_fetch": getattr(config, "finoracle_skip_fetch", False),
            "skip_inference": getattr(config, "finoracle_skip_inference", False),
            "context_len": getattr(config, "finoracle_context_len", 128),
            "horizon_len": getattr(config, "finoracle_horizon_len", 32),
            "optimize": getattr(config, "finoracle_optimize", False),
            "trials": getattr(config, "finoracle_trials", 20),
            "folds": getattr(config, "finoracle_folds", 3),
            "use_gpu": getattr(config, "finoracle_use_gpu", True),
        }

        try:
            # Run bridge script in the FinCast venv
            # Force UTF-8 so Unicode chars (arrows, emojis) in FinCast prints
            # don't crash on Windows cp1252
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            proc = subprocess.run(
                [FINCAST_VENV_PYTHON, BRIDGE_SCRIPT],
                input=json.dumps(bridge_cfg),
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=FINCAST_DIR,  # So relative imports in FinCast work
                timeout=600,      # 10 min timeout for heavy inference
                env=env,
            )

            # Print stderr (log messages from bridge) to our console
            if proc.stderr:
                for line in proc.stderr.strip().splitlines():
                    print(line)

            if proc.returncode != 0:
                print(f"   [Error] Bridge process exited with code {proc.returncode}")
                return {'error': f'Bridge process failed (exit code {proc.returncode})'}

            # Parse JSON result from stdout
            if not proc.stdout.strip():
                return {'error': 'Bridge returned empty output'}
            
            result = json.loads(proc.stdout.strip())
            
            if 'error' in result:
                print(f"   [Error] {result['error']}")
            else:
                status = result.get('status', 'unknown')
                if status == 'success':
                    ret = result.get('expected_return', 0)
                    price = result.get('forecast_price', 0)
                    print(f"   [OK] Forecast: ${price:.2f} ({ret:+.2%})")
                    print(f"   [OK] Horizon: {result.get('horizon_days')}d, Context: {result.get('context_used')}")
                    
            return result

        except subprocess.TimeoutExpired:
            print("   [Error] FinOracle timed out (600s limit)")
            return {'error': 'FinCast inference timed out'}
        except json.JSONDecodeError as e:
            print(f"   [Error] Could not parse bridge output: {e}")
            print(f"   [Debug] Raw stdout: {proc.stdout[:500]}")
            return {'error': f'Invalid JSON from bridge: {e}'}
        except Exception as e:
            print(f"   [Error] FinOracle run failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
