"""
Volatility feature engineering including RV, BV, JV and their transformations.
All features use only information known at day-t close.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityFeatures:
    """Generate volatility-based features with proper time alignment."""
    
    PREFIX = 'vr_'  # Volatility/Regime prefix
    
    def __init__(self,
                 lookback_windows: List[int] = [5, 10, 20, 60],
                 ewma_lambda: float = 0.94):
        """
        Initialize volatility feature generator.
        
        Args:
            lookback_windows: Rolling window sizes in days
            ewma_lambda: EWMA decay parameter
        """
        self.lookback_windows = lookback_windows
        self.ewma_lambda = ewma_lambda
        self.ewma_alpha = 1 - ewma_lambda
        
    def generate(self, daily_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate volatility features from daily data.
        
        Args:
            daily_df: DataFrame with RV, BV, JV measures
            
        Returns:
            DataFrame with volatility features
        """
        logger.info("Generating volatility features")
        
        # Sort by date
        df = daily_df.sort('session_date')
        
        # Core volatility transformations
        df = self._add_vol_transformations(df)
        
        # EWMA and volatility ratios
        df = self._add_ewma_features(df)
        
        # Rolling statistics
        df = self._add_rolling_features(df)
        
        # Volatility regimes
        df = self._add_regime_features(df)
        
        # Volatility term structure
        df = self._add_term_structure(df)
        
        # Validate features
        self._validate_features(df)
        
        return df
    
    def _add_vol_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add core volatility transformations."""
        return df.with_columns([
            # Log transformations (for modeling)
            pl.col('RV_daily').log().alias(f'{self.PREFIX}log_rv'),
            pl.col('BV_daily').log().alias(f'{self.PREFIX}log_bv'),
            (pl.col('JV_daily') + 1e-8).log().alias(f'{self.PREFIX}log_jv'),
            
            # Square root (volatility scale)
            pl.col('RV_daily').sqrt().alias(f'{self.PREFIX}rvol'),
            pl.col('BV_daily').sqrt().alias(f'{self.PREFIX}bvol'),
            pl.col('JV_daily').sqrt().alias(f'{self.PREFIX}jvol'),
            
            # Jump indicators
            (pl.col('JV_daily') / (pl.col('RV_daily') + 1e-8)).alias(f'{self.PREFIX}jump_ratio'),
            (pl.col('JV_daily') > pl.col('RV_daily').quantile(0.95)).alias(f'{self.PREFIX}jump_flag'),
            
            # Continuous vs jump components
            ((pl.col('BV_daily') / (pl.col('RV_daily') + 1e-8))).alias(f'{self.PREFIX}continuous_ratio'),
            
            # Variance ratios
            (pl.col('upside_var') / (pl.col('downside_var') + 1e-8)).alias(f'{self.PREFIX}updown_ratio'),
            
            # Realized moments (standardized)
            pl.col('realized_skew').alias(f'{self.PREFIX}skew'),
            pl.col('realized_kurt').alias(f'{self.PREFIX}kurt')
        ])
    
    def _add_ewma_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add EWMA features."""
        return df.with_columns([
            # EWMA of RV
            pl.col('RV_daily').ewm_mean(alpha=self.ewma_alpha).alias(f'{self.PREFIX}ewma_rv'),
            
            # EWMA of log RV
            pl.col('log_RV_daily').ewm_mean(alpha=self.ewma_alpha).alias(f'{self.PREFIX}ewma_log_rv'),
            
            # Volatility ratio (current / EWMA)
            (pl.col('RV_daily') / pl.col('RV_daily').ewm_mean(alpha=self.ewma_alpha).shift(1))
            .alias(f'{self.PREFIX}vr'),
            
            # EWMA of BV
            pl.col('BV_daily').ewm_mean(alpha=self.ewma_alpha).alias(f'{self.PREFIX}ewma_bv'),
            
            # EWMA of JV
            pl.col('JV_daily').ewm_mean(alpha=self.ewma_alpha).alias(f'{self.PREFIX}ewma_jv')
        ])
    
    def _add_rolling_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling window features."""
        for window in self.lookback_windows:
            df = df.with_columns([
                # Rolling mean
                pl.col('RV_daily')
                .rolling_mean(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}rv_ma_{window}d'),
                
                # Rolling std
                pl.col('RV_daily')
                .rolling_std(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}rv_std_{window}d'),
                
                # Rolling max
                pl.col('RV_daily')
                .rolling_max(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}rv_max_{window}d'),
                
                # Rolling min
                pl.col('RV_daily')
                .rolling_min(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}rv_min_{window}d'),
                
                # Z-score
                ((pl.col('RV_daily') - pl.col('RV_daily').rolling_mean(window_size=window, min_periods=max(1, window//2))) /
                 (pl.col('RV_daily').rolling_std(window_size=window, min_periods=max(1, window//2)) + 1e-8))
                .alias(f'{self.PREFIX}rv_zscore_{window}d')
            ])
            
        return df
    
    def _add_regime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add regime classification features."""
        return df.with_columns([
            # Percentile rank
            pl.col('RV_daily')
            .rank()
            .over(pl.col('session_date').rolling_index(window_size='252d'))
            .alias(f'{self.PREFIX}rv_percentile'),
            
            # Quantile bins
            pl.col('RV_daily')
            .qcut(5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
            .alias(f'{self.PREFIX}rv_quintile'),
            
            # High volatility flag
            (pl.col('RV_daily') > pl.col('RV_daily').rolling_quantile(0.75, window_size=252, min_periods=126))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}high_vol_flag'),
            
            # Low volatility flag
            (pl.col('RV_daily') < pl.col('RV_daily').rolling_quantile(0.25, window_size=252, min_periods=126))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}low_vol_flag')
        ])
    
    def _add_term_structure(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility term structure features."""
        return df.with_columns([
            # Short vs long term
            (pl.col(f'{self.PREFIX}rv_ma_5d') / pl.col(f'{self.PREFIX}rv_ma_20d'))
            .alias(f'{self.PREFIX}term_structure_5_20'),
            
            (pl.col(f'{self.PREFIX}rv_ma_20d') / pl.col(f'{self.PREFIX}rv_ma_60d'))
            .alias(f'{self.PREFIX}term_structure_20_60'),
            
            # Slope of term structure (simplified)
            (pl.col(f'{self.PREFIX}rv_ma_5d') - pl.col(f'{self.PREFIX}rv_ma_60d'))
            .alias(f'{self.PREFIX}term_slope')
        ])
    
    def _validate_features(self, df: pl.DataFrame):
        """Validate generated features."""
        feature_cols = [c for c in df.columns if c.startswith(self.PREFIX)]
        
        # Check for infinite values
        for col in feature_cols:
            if df[col].is_infinite().any():
                logger.warning(f"Infinite values found in {col}")
                
        # Check for extreme values
        for col in feature_cols:
            if df[col].abs().max() > 1e10:
                logger.warning(f"Extreme values found in {col}")
                
        logger.info(f"Generated {len(feature_cols)} volatility features")