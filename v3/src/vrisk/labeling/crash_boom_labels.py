"""
Create crash/boom/normal classification labels based on standardized returns.
Implements multiple gamma thresholds for sensitivity analysis.
"""

import polars as pl
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CrashBoomLabeler:
    """Create 3-class labels for extreme market movements."""
    
    def __init__(self, 
                 gammas: List[float] = [1.5, 2.0, 2.5, 3.0],
                 volatility_estimator: str = 'RV'):
        """
        Initialize labeler.
        
        Args:
            gammas: List of threshold values for crash/boom
            volatility_estimator: Method for volatility ('RV', 'EWMA', 'model')
        """
        self.gammas = gammas
        self.volatility_estimator = volatility_estimator
        
    def create_labels(self, 
                     df: pl.DataFrame,
                     rv_df: pl.DataFrame) -> pl.DataFrame:
        """
        Create crash/boom/normal labels for next-day returns.
        
        Args:
            df: Minute data with returns
            rv_df: Daily RV measures
            
        Returns:
            DataFrame with labels for each gamma
        """
        logger.info(f"Creating crash/boom labels for gammas: {self.gammas}")
        
        # Calculate daily returns
        daily_returns = self._calculate_daily_returns(df)
        
        # Merge with RV data
        daily_data = daily_returns.join(rv_df, on='session_date', how='left')
        
        # Calculate standardized next-day returns
        daily_data = self._calculate_standardized_returns(daily_data)
        
        # Create labels for each gamma
        for gamma in self.gammas:
            daily_data = self._add_labels_for_gamma(daily_data, gamma)
            
        # Add label statistics
        daily_data = self._add_label_stats(daily_data)
        
        # Validate labels
        self._validate_labels(daily_data)
        
        return daily_data
    
    def _calculate_daily_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate daily returns from minute data."""
        daily_returns = (
            df.group_by('session_date')
            .agg([
                # Daily return (close to close)
                (pl.col('ohlcv_close').last() / pl.col('ohlcv_close').first() - 1)
                .alias('daily_return'),
                
                # Alternative: sum of minute returns
                pl.col('minute_return').sum().alias('daily_return_sum'),
                
                # OHLC for the day
                pl.col('ohlcv_open').first().alias('daily_open'),
                pl.col('ohlcv_high').max().alias('daily_high'),
                pl.col('ohlcv_low').min().alias('daily_low'),
                pl.col('ohlcv_close').last().alias('daily_close'),
                
                # Volume
                pl.col('ohlcv_volume').sum().alias('daily_volume')
            ])
            .sort('session_date')
        )
        
        # Add next-day return (target)
        daily_returns = daily_returns.with_columns([
            pl.col('daily_return').shift(-1).alias('next_day_return'),
            pl.col('daily_return_sum').shift(-1).alias('next_day_return_sum')
        ])
        
        return daily_returns
    
    def _calculate_standardized_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate standardized returns z = r_{t+1} / σ_t.
        """
        # Choose volatility estimator
        if self.volatility_estimator == 'RV':
            vol_col = 'RVol_daily'
        elif self.volatility_estimator == 'BV':
            vol_col = 'BVol_daily'
        elif self.volatility_estimator == 'EWMA':
            # Calculate EWMA volatility
            df = df.with_columns([
                pl.col('RV_daily')
                .ewm_mean(alpha=0.06)  # λ = 0.94 => α = 1 - 0.94 = 0.06
                .sqrt()
                .alias('ewma_vol')
            ])
            vol_col = 'ewma_vol'
        else:
            vol_col = 'RVol_daily'
            
        # Standardize next-day return
        df = df.with_columns([
            (pl.col('next_day_return') / pl.col(vol_col))
            .alias('z_score_next'),
            
            # Also store the volatility used
            pl.col(vol_col).alias('volatility_for_z')
        ])
        
        # Handle edge cases
        df = df.with_columns([
            # Cap extreme z-scores
            pl.col('z_score_next').clip(-10, 10).alias('z_score_next_clipped')
        ])
        
        return df
    
    def _add_labels_for_gamma(self, 
                              df: pl.DataFrame, 
                              gamma: float) -> pl.DataFrame:
        """Add crash/boom/normal labels for a specific gamma."""
        label_col = f'label_gamma_{gamma:.1f}'
        prob_cols = [f'prob_crash_{gamma:.1f}', 
                    f'prob_normal_{gamma:.1f}', 
                    f'prob_boom_{gamma:.1f}']
        
        # Create labels
        df = df.with_columns([
            pl.when(pl.col('z_score_next') <= -gamma)
            .then(pl.lit('crash'))
            .when(pl.col('z_score_next') >= gamma)
            .then(pl.lit('boom'))
            .otherwise(pl.lit('normal'))
            .alias(label_col),
            
            # Numeric version (for models)
            pl.when(pl.col('z_score_next') <= -gamma)
            .then(0)  # Crash = 0
            .when(pl.col('z_score_next') >= gamma)
            .then(2)  # Boom = 2
            .otherwise(1)  # Normal = 1
            .alias(f'{label_col}_numeric')
        ])
        
        # Add distance to thresholds (useful features)
        df = df.with_columns([
            # Distance to crash threshold
            (pl.col('z_score_next') - (-gamma)).alias(f'dist_to_crash_{gamma:.1f}'),
            
            # Distance to boom threshold
            (pl.col('z_score_next') - gamma).alias(f'dist_to_boom_{gamma:.1f}'),
            
            # Min distance to any threshold
            pl.min_horizontal([
                (pl.col('z_score_next') - (-gamma)).abs(),
                (pl.col('z_score_next') - gamma).abs()
            ]).alias(f'min_dist_threshold_{gamma:.1f}')
        ])
        
        return df
    
    def _add_label_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add label distribution statistics."""
        for gamma in self.gammas:
            label_col = f'label_gamma_{gamma:.1f}'
            
            # Calculate class counts
            class_counts = df[label_col].value_counts()
            total = len(df)
            
            # Log class distribution
            for row in class_counts.to_dicts():
                pct = row['count'] / total * 100
                logger.info(f"Gamma {gamma}: {row[label_col]} = {row['count']} ({pct:.2f}%)")
                
        return df
    
    def _validate_labels(self, df: pl.DataFrame) -> None:
        """Validate created labels."""
        # Check for null labels
        for gamma in self.gammas:
            label_col = f'label_gamma_{gamma:.1f}'
            null_count = df[label_col].null_count()
            
            if null_count > 0:
                logger.warning(f"Found {null_count} null labels for gamma={gamma}")
                
        # Check z-score distribution
        z_stats = df.select([
            pl.col('z_score_next').mean().alias('mean'),
            pl.col('z_score_next').std().alias('std'),
            pl.col('z_score_next').min().alias('min'),
            pl.col('z_score_next').max().alias('max')
        ]).to_dicts()[0]
        
        logger.info(f"Z-score statistics: {z_stats}")
        
        # Verify threshold logic
        for gamma in self.gammas:
            label_col = f'label_gamma_{gamma:.1f}'
            
            # Check crash labels
            crash_check = df.filter(
                (pl.col(label_col) == 'crash') & (pl.col('z_score_next') > -gamma)
            )
            if len(crash_check) > 0:
                logger.error(f"Invalid crash labels for gamma={gamma}")
                
            # Check boom labels  
            boom_check = df.filter(
                (pl.col(label_col) == 'boom') & (pl.col('z_score_next') < gamma)
            )
            if len(boom_check) > 0:
                logger.error(f"Invalid boom labels for gamma={gamma}")


def create_crash_boom_labels(
    minute_df: pl.DataFrame,
    rv_df: pl.DataFrame,
    gammas: List[float] = [1.5, 2.0, 2.5, 3.0],
    volatility_estimator: str = 'RV'
) -> pl.DataFrame:
    """
    Convenience function to create crash/boom labels.
    
    Args:
        minute_df: Minute-level data
        rv_df: Daily RV measures
        gammas: Threshold values
        volatility_estimator: Method for volatility
        
    Returns:
        DataFrame with daily labels
    """
    labeler = CrashBoomLabeler(gammas, volatility_estimator)
    return labeler.create_labels(minute_df, rv_df)