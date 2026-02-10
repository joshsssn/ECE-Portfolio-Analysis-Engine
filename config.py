"""
ECE Portfolio Analysis Engine
=============================
Configuration Module
=============================
Centralized configuration management for the analysis pipeline.
"""

from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    """
    Configuration parameters for the analysis pipeline.
    """
    # 1. Optimal Allocation Parameters
    risk_aversion: float = 2.0          # Lambda (λ)
    concentration_penalty: float = 0.5  # Gamma (γ)
    min_recommended_allocation: float = 0.03
    min_allocation: float = 0.00
    max_allocation: float = 0.25
    
    # 2. General Market Parameters
    benchmark_ticker: str = 'ACWI'
    risk_free_rate: float = 0.04
    lookback_years: int = 10
    resample_freq: str = 'W'  # 'W' for Weekly, 'D' for Daily
    
    # 3. Model Specifics
    tech_etf_ticker: str = 'IXN'  # Used for correlation checks
    
    # 4. Valuation Parameters
    n_simulations: int = 10000    # Monte Carlo count
    projection_years: int = 10
    terminal_growth_base: float = 0.025
    max_growth_rate: float = 0.10
    
    # 5. Drawdown Protection Parameters
    drawdown_reduction_threshold: float = 0.10  # -10% drawdown triggers reduction
    drawdown_reduction_factor: float = 0.50      # Reduce exposure to 50%
    drawdown_recovery_threshold: float = 0.05   # -5% drawdown, start recovering
    
    # 6. Advanced Statistics
    use_ledoit_wolf: bool = True  # Use Covariance Shrinkage

    # 7. FinOracle (Forecasting) Parameters
    enable_finoracle: bool = False
    # Data Fetching
    finoracle_freq: str = 'd'           # tick, 1min, 5min, 1h, d, w, m
    finoracle_days: int = None          # Fetch last N days (overrides years)
    finoracle_years: int = 5            # Fetch last N years
    finoracle_start: str = None         # Specific start date YYYY-MM-DD
    finoracle_end: str = None           # Specific end date YYYY-MM-DD (default: today)
    finoracle_skip_fetch: bool = False  # Reuse existing data.csv
    # Model Configuration
    finoracle_context_len: int = 128    # L: 32-1024
    finoracle_horizon_len: int = 16     # H: 1-256
    finoracle_optimize: bool = False    # AutoML hyperopt
    finoracle_trials: int = 20          # Optuna trials
    finoracle_folds: int = 3            # CV folds
    finoracle_use_gpu: bool = True      # Use GPU if available
    finoracle_skip_inference: bool = False  # Skip model run (re-visualize)

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

