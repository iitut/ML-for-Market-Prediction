"""
Calculate daily realized variance (RV), bipower variation (BV), and jump variation (JV).
These are core volatility measures for the volatility prediction head.
"""

import polars as pl
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RealizedVarianceCalculator:
    """Calculate RV, BV, JV and other volatility estimators."""
    
    def __init__(self, return_col: str = 'minute_return'):
        """
        Initialize RV calculator.
        
        Args:
            return_col: Column containing minute returns
        """
        self.return_col = return_col
        
    def calculate_daily_measures(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate daily volatility measures from minute returns.
        
        Args:
            df: DataFrame with minute returns and session_date
            
        Returns:
            DataFrame with daily volatility measures
        """
        logger.info("Calculating daily volatility measures")
        
        # Calculate RV (sum of squared returns)
        rv = (
            df.group_by('session_date')
            .agg([
                # Realized Variance
                (pl.col(self.return_col).pow(2).sum())
                .alias('RV_daily'),
                
                # Number of observations
                pl.col(self.return_col).count().alias('n_minutes'),
                
                # For annualization
                pl.lit(252).alias('trading_days_year')
            ])
        )
        
        # Calculate BV (Bipower Variation)
        bv = self._calculate_bipower_variation(df)
        rv = rv.join(bv, on='session_date', how='left')
        
        # Calculate JV (Jump Variation)
        rv = rv.with_columns([
            # JV = max(RV - BV, 0)
            pl.max_horizontal([
                pl.col('RV_daily') - pl.col('BV_daily'),
                pl.lit(0)
            ]).alias('JV_daily')
        ])
        
        # Add additional estimators
        rv = self._add_range_estimators(rv, df)
        
        # Calculate realized volatility (square root of RV)
        rv = rv.with_columns([
            pl.col('RV_daily').sqrt().alias('RVol_daily'),
            pl.col('BV_daily').sqrt().alias('BVol_daily'),
            pl.col('JV_daily').sqrt().alias('JVol_daily'),
            
            # Annualized versions
            (pl.col('RV_daily') * pl.col('trading_days_year')).sqrt()
            .alias('RVol_annual'),
            
            # Log versions for modeling
            pl.col('RV_daily').log().alias('log_RV_daily'),
            
            # Jump ratio
            (pl.col('JV_daily') / pl.col('RV_daily')).alias('jump_ratio')
        ])
        
        return rv
    
    def _calculate_bipower_variation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Bipower Variation (robust to jumps).
        BV = (π/2) * Σ|r_i||r_{i-1}|
        """
        # Add lagged absolute returns
        df_bv = df.with_columns([
            pl.col(self.return_col).abs().alias('abs_return'),
            pl.col(self.return_col).abs().shift(1).over('session_date')
            .alias('abs_return_lag')
        ])
        
        # Calculate BV
        bv = (
            df_bv.group_by('session_date')
            .agg([
                # Sum of products of consecutive absolute returns
                (pl.col('abs_return') * pl.col('abs_return_lag'))
                .sum()
                .alias('bv_sum')
            ])
            .with_columns([
                # Multiply by π/2 for BV
                (pl.col('bv_sum') * (np.pi / 2)).alias('BV_daily')
            ])
            .select(['session_date', 'BV_daily'])
        )
        
        return bv
    
    def _add_range_estimators(self, 
                              rv_df: pl.DataFrame, 
                              minute_df: pl.DataFrame) -> pl.DataFrame:
        """Add Parkinson, Garman-Klass, and Rogers-Satchell estimators."""
        
        # Get daily OHLC
        daily_ohlc = (
            minute_df.group_by('session_date')
            .agg([
                pl.col('ohlcv_open').first().alias('daily_open'),
                pl.col('ohlcv_high').max().alias('daily_high'),
                pl.col('ohlcv_low').min().alias('daily_low'),
                pl.col('ohlcv_close').last().alias('daily_close')
            ])
        )
        
        # Join with RV data
        rv_df = rv_df.join(daily_ohlc, on='session_date', how='left')
        
        # Calculate range-based estimators
        rv_df = rv_df.with_columns([
            # Parkinson estimator: (1/4ln2) * (ln(H/L))^2
            ((pl.col('daily_high') / pl.col('daily_low')).log().pow(2) / (4 * np.log(2)))
            .alias('parkinson_var'),
            
            # Garman-Klass estimator
            ((0.5 * (pl.col('daily_high') / pl.col('daily_low')).log().pow(2) -
              (2 * np.log(2) - 1) * 
              (pl.col('daily_close') / pl.col('daily_open')).log().pow(2)))
            .alias('garman_klass_var'),
            
            # Rogers-Satchell estimator
            ((pl.col('daily_high') / pl.col('daily_close')).log() *
             (pl.col('daily_high') / pl.col('daily_open')).log() +
             (pl.col('daily_low') / pl.col('daily_close')).log() *
             (pl.col('daily_low') / pl.col('daily_open')).log())
            .alias('rogers_satchell_var'),
            
            # Range (normalized)
            ((pl.col('daily_high') - pl.col('daily_low')) / pl.col('daily_close'))
            .alias('daily_range'),
            
            # Gap (overnight return)
            (pl.col('daily_open') / pl.col('daily_close').shift(1)).log()
            .alias('overnight_gap')
        ])
        
        return rv_df
    
    def calculate_realized_moments(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate higher moments: realized skewness and kurtosis."""
        
        moments = (
            df.group_by('session_date')
            .agg([
                # Realized Skewness
                ((pl.col(self.return_col).pow(3).sum()) /
                 (pl.col(self.return_col).pow(2).sum().pow(1.5)))
                .alias('realized_skew'),
                
                # Realized Kurtosis
                ((pl.col(self.return_col).pow(4).sum()) /
                 (pl.col(self.return_col).pow(2).sum().pow(2)))
                .alias('realized_kurt'),
                
                # Downside variance
                pl.when(pl.col(self.return_col) < 0)
                .then(pl.col(self.return_col).pow(2))
                .otherwise(0)
                .sum()
                .alias('downside_var'),
                
                # Upside variance
                pl.when(pl.col(self.return_col) > 0)
                .then(pl.col(self.return_col).pow(2))
                .otherwise(0)
                .sum()
                .alias('upside_var')
            ])
        )
        
        # Add variance ratio
        moments = moments.with_columns([
            (pl.col('downside_var') / pl.col('upside_var')).alias('variance_ratio')
        ])
        
        return moments


def calculate_daily_rv(
    df: pl.DataFrame,
    return_col: str = 'minute_return'
) -> pl.DataFrame:
    """
    Convenience function to calculate daily RV measures.
    
    Args:
        df: DataFrame with minute returns
        return_col: Column containing returns
        
    Returns:
        DataFrame with daily volatility measures
    """
    calculator = RealizedVarianceCalculator(return_col)
    rv_df = calculator.calculate_daily_measures(df)
    moments_df = calculator.calculate_realized_moments(df)
    
    # Combine RV and moments
    return rv_df.join(moments_df, on='session_date', how='left')