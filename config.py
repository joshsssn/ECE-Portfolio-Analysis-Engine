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
