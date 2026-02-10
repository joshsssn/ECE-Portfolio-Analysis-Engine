"""
ECE Portfolio Analysis Engine
=============================
Covariance Estimation Module
============================
Implements Ledoit-Wolf shrinkage for stable covariance matrices.
Standard technique used by quant funds to stabilize mean-variance optimization.

Author: Josh E. SOUSSAN
"""

import numpy as np
import pandas as pd
from typing import Literal


def estimate_covariance(
    returns: pd.DataFrame, 
    method: Literal['sample', 'ledoit_wolf'] = 'ledoit_wolf'
) -> np.ndarray:
    """
    Estimate covariance matrix with optional shrinkage.
    
    Ledoit-Wolf shrinkage provides a regularized covariance matrix that is
    more stable than the sample covariance, especially when T (observations)
    is not much larger than N (assets).
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns matrix (T observations x N assets)
    method : str
        'sample' - Raw sample covariance (unstable with few observations)
        'ledoit_wolf' - Shrunk covariance (more stable, recommended)
    
    Returns
    -------
    np.ndarray
        Covariance matrix (N x N)
    
    Notes
    -----
    Ledoit-Wolf shrinkage combines the sample covariance with a structured
    estimator (identity matrix scaled by average variance) to reduce estimation
    error, especially for the off-diagonal elements.
    
    Reference:
        Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for
        large-dimensional covariance matrices." Journal of Multivariate Analysis.
    """
    if method == 'ledoit_wolf':
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(returns.values)
            return lw.covariance_
        except ImportError:
            print("Warning: sklearn not installed. Falling back to sample covariance.")
            return returns.cov().values
    
    return returns.cov().values


def estimate_correlation(
    returns: pd.DataFrame,
    method: Literal['sample', 'ledoit_wolf'] = 'ledoit_wolf'
) -> np.ndarray:
    """
    Estimate correlation matrix with optional shrinkage.
    
    Converts shrunk covariance to correlation matrix.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns matrix (T observations x N assets)
    method : str
        'sample' or 'ledoit_wolf'
    
    Returns
    -------
    np.ndarray
        Correlation matrix (N x N)
    """
    cov = estimate_covariance(returns, method)
    
    # Convert covariance to correlation
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    
    # Ensure diagonal is exactly 1 (numerical precision)
    np.fill_diagonal(corr, 1.0)
    
    return corr


def get_shrinkage_intensity(returns: pd.DataFrame) -> float:
    """
    Get the shrinkage intensity (alpha) used by Ledoit-Wolf.
    
    Higher alpha = more shrinkage toward structured estimator.
    Alpha ≈ 0 means sample covariance is reliable.
    Alpha ≈ 1 means sample covariance is very noisy.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns matrix
    
    Returns
    -------
    float
        Shrinkage intensity (0 to 1)
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.values)
        return lw.shrinkage_
    except ImportError:
        return 0.0


if __name__ == "__main__":
    # Quick test
    import yfinance as yf
    from datetime import datetime, timedelta
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    end = datetime.now()
    start = end - timedelta(days=365 * 2)
    
    prices = yf.download(tickers, start=start, end=end)['Close']
    returns = prices.pct_change().dropna()
    
    print("="*60)
    print("COVARIANCE ESTIMATION TEST")
    print("="*60)
    
    sample_cov = estimate_covariance(returns, 'sample')
    lw_cov = estimate_covariance(returns, 'ledoit_wolf')
    shrinkage = get_shrinkage_intensity(returns)
    
    print(f"\nNumber of observations: {len(returns)}")
    print(f"Number of assets: {len(tickers)}")
    print(f"Shrinkage intensity: {shrinkage:.4f}")
    print(f"\nSample covariance condition number: {np.linalg.cond(sample_cov):.2f}")
    print(f"Ledoit-Wolf covariance condition number: {np.linalg.cond(lw_cov):.2f}")
    print("\n✅ Lower condition number = more stable matrix inversion")
