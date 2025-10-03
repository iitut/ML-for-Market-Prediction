"""
Configuration file for ML Model v2 - Enhanced Version
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v2\\config.py
"""

import os
from pathlib import Path

# Base paths
BASE_PATH = Path(r"C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP")
DATA_PATH = BASE_PATH / "Data"
MODEL_PATH = BASE_PATH / "MODELS" / "v2"

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

# ===== ENHANCED MODEL PARAMETERS =====

# Crash/Boom Thresholds - MORE SENSITIVE
THRESHOLDS = {
    'extreme_crash': -0.03,    # -3% daily return
    'crash': -0.015,          # -1.5% daily return (was -2σ)
    'mild_down': -0.005,      # -0.5% daily return
    'normal': (-0.005, 0.005), # -0.5% to +0.5%
    'mild_up': 0.005,         # +0.5% daily return
    'boom': 0.015,            # +1.5% daily return (was +2σ)
    'extreme_boom': 0.03      # +3% daily return
}

# Time horizons for predictions
TIME_HORIZONS = {
    'intraday_1h': 60,        # 1 hour after open (minutes)
    'daily': 1,               # Next day
    'weekly': 5,              # Next week (trading days)
    'monthly': 21             # Next month (trading days)
}

# Model ensemble configuration
ENSEMBLE_MODELS = [
    'xgboost',
    'lightgbm',
    'catboost',
    'random_forest',
    'neural_network',
    'gradient_boosting'
]

# Enhanced feature categories
FEATURE_CATEGORIES = {
    'volatility': [
        'realized_variance', 'parkinson_estimator', 'garman_klass',
        'rogers_satchell', 'bipower_variation', 'jump_variation',
        'realized_skewness', 'realized_kurtosis', 'ewma_rv',
        'volatility_ratio', 'yang_zhang', 'close_to_close_vol'
    ],
    'trend': [
        'gap', 'intraday_range_ratio', 'last_hour_return',
        'rsi', 'macd', 'bollinger_position', 'momentum',
        'rate_of_change', 'williams_r', 'stochastic_k'
    ],
    'liquidity': [
        'dollar_volume', 'volume_volatility', 'amihud_illiquidity',
        'volume_ratio', 'trade_intensity', 'kyle_lambda'
    ],
    'microstructure': [
        'bid_ask_spread', 'effective_spread', 'realized_spread',
        'price_impact', 'order_imbalance', 'quote_slope'
    ],
    'technical': [
        'sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'atr', 'adx', 'cci', 'mfi', 'obv_ratio'
    ],
    'sentiment': [
        'vix_level', 'vix_change', 'put_call_ratio',
        'term_structure', 'skew_index'
    ]
}

# Output predictions configuration
PREDICTION_OUTPUTS = {
    'classification': {
        'crash_boom_7class': ['extreme_crash', 'crash', 'mild_down', 
                              'normal', 'mild_up', 'boom', 'extreme_boom'],
        'direction_3class': ['down', 'neutral', 'up'],
        'week_outlook': ['week_crash', 'week_normal', 'week_boom']
    },
    'regression': {
        'next_day_return': 'continuous',
        'next_week_return': 'continuous',
        'next_month_return': 'continuous',
        'hour_after_open_price': 'continuous',
        'volatility_forecast': 'continuous'
    },
    'probabilistic': {
        'crash_probability_1d': [0, 1],
        'crash_probability_1w': [0, 1],
        'boom_probability_1d': [0, 1],
        'boom_probability_1w': [0, 1]
    }
}

# Training parameters
TRAINING_PARAMS = {
    'initial_train_size': 252,
    'validation_split': 0.2,
    'test_split': 0.15,
    'cv_folds': 5,
    'random_state': 42
}

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'random_state': 42
}

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.01,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42
}

# CatBoost parameters
CATBOOST_PARAMS = {
    'iterations': 300,
    'depth': 8,
    'learning_rate': 0.01,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_state': 42,
    'verbose': False
}

# Neural Network architecture
NN_ARCHITECTURE = {
    'hidden_layers': [256, 128, 64, 32],
    'dropout_rates': [0.3, 0.3, 0.2, 0.2],
    'activation': 'relu',
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

# Monitoring parameters
MONITOR_DAYS = list(range(0, 2000, 100))  # Monitor every 100 days
VERBOSE_LEVEL = 2  # 0: silent, 1: progress, 2: detailed

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE = (20, 12)
COLORS = {
    'extreme_crash': '#8B0000',  # Dark red
    'crash': '#FF0000',          # Red
    'mild_down': '#FFA500',      # Orange
    'normal': '#808080',         # Gray
    'mild_up': '#90EE90',        # Light green
    'boom': '#00FF00',           # Green
    'extreme_boom': '#006400'    # Dark green
}