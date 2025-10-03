"""
Configuration file for ML Model v1
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v1\\config.py
"""

import os
from pathlib import Path

# Base paths
BASE_PATH = Path(r"C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP")
DATA_PATH = BASE_PATH / "Data"
MODEL_PATH = BASE_PATH / "MODELS" / "v1"

# Data file paths
MASTER_DATASET = DATA_PATH / "master_dataset.csv"
QQQ_DAILY_FILLED = DATA_PATH / "QQQ" / "qqq_daily_dataset_filled.csv"
QQQ_DAILY_REDUCED = DATA_PATH / "QQQ" / "qqq_daily_dataset_reduced.csv"

# Intraday data
IEX_FILLED = DATA_PATH / "QQQ" / "iex_filled.csv"
IEX_CLEANED = DATA_PATH / "QQQ" / "iex_cleaned.csv"
OHLCV_PROCESSED = DATA_PATH / "QQQ" / "ohlcv_processed.csv"

# External data
CBOE_INDEX = DATA_PATH / "CBOE_index_processed.csv"
UMCSI = DATA_PATH / "UMCSI_processed.csv"
POLICY_UNCERTAINTY = DATA_PATH / "US_Policy_Uncertainty_Data_processed.csv"
COMP_NASDAQ = DATA_PATH / "COMP_NASDAQ_processed.csv"

# Calendar data
NASDAQ_CALENDAR = DATA_PATH / "NASDAQ_calendar_processed.csv"
OPTIONS_EXPIRATIONS = DATA_PATH / "us_options_expirations(2020)_processed.csv"

# Model parameters
CRASH_BOOM_THRESHOLD = 2.0  # γ for standardized returns
EWMA_LAMBDA = 0.94  # For EWMA calculations
LOOKBACK_WINDOW = 252  # Trading days for volatility estimation

# Features to calculate
VOLATILITY_FEATURES = [
    'realized_variance',
    'parkinson_estimator',
    'garman_klass',
    'rogers_satchell',
    'bipower_variation',
    'jump_variation',
    'realized_skewness',
    'realized_kurtosis',
    'ewma_rv',
    'volatility_ratio'
]

TREND_FEATURES = [
    'gap',
    'intraday_range_ratio',
    'last_hour_return'
]

LIQUIDITY_FEATURES = [
    'dollar_volume',
    'volume_volatility'
]

# Output targets
TARGETS = {
    'crash_boom': ['crash', 'boom', 'normal'],
    'volatility': 'log_rv_next',
    'quantiles': [0.05, 0.50, 0.95]
}

# Monitoring parameters
MONITOR_DAYS = list(range(0, 1674, 100))  # First day and every 100th day
VALIDATION_SPLIT = 0.2  # Last 20% for validation

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (15, 8)
COLORS = {
    'actual': 'blue',
    'predicted': 'red',
    'crash': 'darkred',
    'boom': 'darkgreen',
    'normal': 'gray'
}