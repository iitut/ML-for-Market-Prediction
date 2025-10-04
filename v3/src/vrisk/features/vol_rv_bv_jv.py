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
            (pl.col('JV_daily') > pl.col('RV_daily').quantile(0.95)).cast(pl.Int32, strict=False).alias(f'{self.PREFIX}jump_flag'),
            
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
        alpha = self.ewma_alpha
        
        return df.with_columns([
            # EWMA of RV
            pl.col('RV_daily').ewm_mean(alpha=alpha).alias(f'{self.PREFIX}ewma_rv'),
            
            # EWMA of log RV
            pl.col(f'{self.PREFIX}log_rv').ewm_mean(alpha=alpha).alias(f'{self.PREFIX}ewma_log_rv'),
            
            # EWMA std
            pl.col('RV_daily').ewm_std(alpha=alpha).alias(f'{self.PREFIX}ewma_std'),
        ])
    
    def _add_rolling_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling statistics."""
        for window in self.lookback_windows:
            df = df.with_columns([
                # Rolling mean
                pl.col('RV_daily').rolling_mean(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}rv_ma{window}'),
                
                # Rolling std
                pl.col('RV_daily').rolling_std(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}rv_std{window}'),
                
                # Rolling max/min
                pl.col('RV_daily').rolling_max(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}rv_max{window}'),
                
                pl.col('RV_daily').rolling_min(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}rv_min{window}'),
            ])
        
        return df
    
    def _add_regime_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility regime indicators."""
        return df.with_columns([
            # High vol regime
            (pl.col('RV_daily') > pl.col('RV_daily').rolling_quantile(0.75, window_size=60, min_periods=30))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}high_vol_regime'),
            
            # Low vol regime
            (pl.col('RV_daily') < pl.col('RV_daily').rolling_quantile(0.25, window_size=60, min_periods=30))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}low_vol_regime'),
        ])
    
    def _add_term_structure(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add term structure features."""
        return df.with_columns([
            # Short vs long term
            (pl.col(f'{self.PREFIX}rv_ma5') / pl.col(f'{self.PREFIX}rv_ma20'))
            .alias(f'{self.PREFIX}term_structure_5_20'),
            
            (pl.col(f'{self.PREFIX}rv_ma10') / pl.col(f'{self.PREFIX}rv_ma60'))
            .alias(f'{self.PREFIX}term_structure_10_60'),
        ])
    
    def _validate_features(self, df: pl.DataFrame) -> None:
        """Validate generated features."""
        feature_cols = [c for c in df.columns if c.startswith(self.PREFIX)]
        logger.info(f"Generated {len(feature_cols)} volatility features")