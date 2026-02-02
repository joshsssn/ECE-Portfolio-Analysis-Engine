"""
ECE Portfolio Analysis Engine
=============================
Portfolio Loader
=============================
Handles loading portfolio reconstruction data from CSV files.
"""

import pandas as pd
from typing import Dict, Tuple

import os
from pathlib import Path

# Defaults Paths
DEFAULT_DATA_DIR = Path('default')
DEFAULT_HOLDINGS_PATH = DEFAULT_DATA_DIR / 'default_holdings.csv'
DEFAULT_SECTORS_PATH = DEFAULT_DATA_DIR / 'default_sectors.csv'

# Hardcoded Fallbacks (used if files are missing)
FALLBACK_TOP_HOLDINGS = {
    'AAPL': {'name': 'APPLE INC.', 'weight': 7.0, 'sector': 'Information Technology', 'country': 'United States'},
    'MSFT': {'name': 'MICROSOFT CORP.', 'weight': 6.0, 'sector': 'Information Technology', 'country': 'United States'},
    'NVDA': {'name': 'NVIDIA CORP.', 'weight': 5.0, 'sector': 'Information Technology', 'country': 'United States'},
    'ASML': {'name': 'ASML HOLDING NV', 'weight': 4.0, 'sector': 'Information Technology', 'country': 'Netherlands'},
    'SAP': {'name': 'SAP SE', 'weight': 2.5, 'sector': 'Information Technology', 'country': 'Germany'},
    'REY.MI': {'name': 'REPLY SPA', 'weight': 2.0, 'sector': 'Information Technology', 'country': 'Italy'},
    'IDR.MC': {'name': 'INDRA SISTEMAS SA', 'weight': 2.0, 'sector': 'Industrials', 'country': 'Spain'},
    'JPM': {'name': 'JPMORGAN CHASE & CO.', 'weight': 3.0, 'sector': 'Financials', 'country': 'United States'},
    'GS': {'name': 'GOLDMAN SACHS GROUP INC.', 'weight': 2.5, 'sector': 'Financials', 'country': 'United States'},
    'HSBC': {'name': 'HSBC HOLDINGS PLC', 'weight': 2.0, 'sector': 'Financials', 'country': 'United Kingdom'},
}

FALLBACK_SECTOR_TARGETS = {
    'Information Technology': 26.5,
    'Financials': 12.5,
    'Industrials': 8.0,
    'Health Care': 9.5,
    'Consumer Discretionary': 5.0,
    'Communication Services': 6.5,
    'Real Estate': 8.7,
    'Consumer Staples': 5.0,
    'Utilities': 2.3,
    'Energy': 3.9,
    'Commodities': 12.1,
}

# Mapping of Sector to ETF Ticker
SECTOR_ETF_MAP = {
    'Information Technology': 'IXN',
    'Financials': 'IXG',
    'Health Care': 'IXJ',
    'Industrials': 'EXI',
    'Energy': 'IXC',
    'Commodities': 'MXI',
    'Consumer Staples': 'KXI',
    'Consumer Discretionary': 'RXI',
    'Utilities': 'JXI',
    'Communication Services': 'IXP',
    'Real Estate': 'REET',
}

def load_holdings_csv(csv_path: str) -> Dict[str, Dict]:
    """
    Load loadings from a CSV file.
    Expected columns: Ticker, Weight, Sector, [Name, Country]
    
    Returns standard dict format:
    { 'TICKER': {'name': '...', 'weight': 5.0, 'sector': '...', 'country': '...'} }
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        required = ['ticker', 'weight', 'sector']
        for req in required:
            if req not in df.columns:
                print(f"Error: Missing required column '{req}' in holdings CSV")
                return FALLBACK_TOP_HOLDINGS  # Use fallback on error

        holdings = {}
        for _, row in df.iterrows():
            ticker = str(row['ticker']).strip().upper()
            holdings[ticker] = {
                'name': row.get('name', ticker),
                'weight': float(row['weight']),
                'sector': row['sector'],
                'country': row.get('country', 'Unknown')
            }
        return holdings
        
    except Exception as e:
        print(f"Error loading holdings CSV: {e}")
        return FALLBACK_TOP_HOLDINGS

def load_sector_targets_csv(csv_path: str) -> Dict[str, float]:
    """
    Load sector targets from CSV.
    Expected columns: Sector, Weight
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        
        if 'sector' not in df.columns or 'weight' not in df.columns:
             print("Error: Sector targets CSV must have 'Sector' and 'Weight' columns")
             return FALLBACK_SECTOR_TARGETS
             
        targets = {}
        for _, row in df.iterrows():
            sector = row['sector']
            weight = float(row['weight'])
            targets[sector] = weight
            
        return targets
        
    except Exception as e:
        print(f"Error loading sector CSV: {e}")
        return FALLBACK_SECTOR_TARGETS

# Initialize Defaults (Dynamic Loading)
def get_default_holdings():
    if DEFAULT_HOLDINGS_PATH.exists():
        return load_holdings_csv(str(DEFAULT_HOLDINGS_PATH))
    return FALLBACK_TOP_HOLDINGS

def get_default_sectors():
    if DEFAULT_SECTORS_PATH.exists():
        return load_sector_targets_csv(str(DEFAULT_SECTORS_PATH))
    return FALLBACK_SECTOR_TARGETS

DEFAULT_TOP_HOLDINGS = get_default_holdings()
DEFAULT_SECTOR_TARGETS = get_default_sectors()
