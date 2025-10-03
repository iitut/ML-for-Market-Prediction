"""
Calculate minute-level returns with proper session handling.
Ensures no overnight gaps or pre/post-market contamination.
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MinuteReturnsCalculator:
    """Calculate intraday minute returns with session awareness."""
    
    def __init__(self, 
                 price_col: str = 'ohlcv_close',
                 log_returns: bool = True):
        """
        Initialize returns calculator.
        
        Args:
            price_col: Column name for prices
            log_returns: Use log returns (True) or simple returns
        """
        self.price_col = price_col
        self.log_returns = log_returns
        
    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate minute returns with proper session handling.
        
        Args:
            df: DataFrame with price column and session_date
            
        Returns:
            DataFrame with added returns column
        """
        logger.info(f"Calculating minute returns from {self.price_col}")
        
        # Ensure data is sorted
        df = df.sort('timestamp')
        
        # Calculate returns within each session
        if self.log_returns:
            df = df.with_columns([
                # Log returns within session
                (pl.col(self.price_col).log() - 
                 pl.col(self.price_col).log().shift(1))
                .over('session_date')
                .alias('minute_return')
            ])
        else:
            # Simple returns
            df = df.with_columns([
                ((pl.col(self.price_col) - pl.col(self.price_col).shift(1)) / 
                 pl.col(self.price_col).shift(1))
                .over('session_date')
                .alias('minute_return')
            ])
            
        # Set first return of each session to null (no overnight return)
        df = df.with_columns([
            pl.when(
                pl.col('minute_of_session') == 1
            ).then(None)
            .otherwise(pl.col('minute_return'))
            .alias('minute_return')
        ])
        
        # Add return statistics
        df = self._add_return_stats(df)
        
        # Validate returns
        self._validate_returns(df)
        
        return df
    
    def _add_return_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling return statistics."""
        return df.with_columns([
            # Squared returns (for RV calculation)
            pl.col('minute_return').pow(2).alias('minute_return_sq'),
            
            # Absolute returns
            pl.col('minute_return').abs().alias('minute_return_abs'),
            
            # Sign of returns
            pl.when(pl.col('minute_return') > 0).then(1)
            .when(pl.col('minute_return') < 0).then(-1)
            .otherwise(0)
            .alias('minute_return_sign'),
            
            # Rolling statistics (last 30 minutes)
            pl.col('minute_return')
            .rolling_mean(window_size=30, min_periods=10)
            .over('session_date')
            .alias('minute_return_ma30'),
            
            pl.col('minute_return')
            .rolling_std(window_size=30, min_periods=10)
            .over('session_date')
            .alias('minute_return_std30'),
            
            # Cumulative return within session
            pl.col('minute_return')
            .cum_sum()
            .over('session_date')
            .alias('session_cumulative_return')
        ])
    
    def _validate_returns(self, df: pl.DataFrame) -> None:
        """Validate calculated returns."""
        # Check for extreme returns
        extreme_returns = df.filter(
            pl.col('minute_return').abs() > 0.1  # 10% minute return
        )
        
        if len(extreme_returns) > 0:
            logger.warning(f"Found {len(extreme_returns)} extreme minute returns (>10%)")
            
        # Check for NaN values (excluding first minute of each session)
        non_first = df.filter(pl.col('minute_of_session') > 1)
        nan_count = non_first['minute_return'].null_count()
        
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN returns (excluding session starts)")
            
        # Log summary statistics
        stats = df.select([
            pl.col('minute_return').mean().alias('mean'),
            pl.col('minute_return').std().alias('std'),
            pl.col('minute_return').min().alias('min'),
            pl.col('minute_return').max().alias('max'),
            pl.col('minute_return').quantile(0.01).alias('q01'),
            pl.col('minute_return').quantile(0.99).alias('q99')
        ]).to_dicts()[0]
        
        logger.info(f"Return statistics: {stats}")


def calculate_minute_returns(
    df: pl.DataFrame,
    price_col: str = 'ohlcv_close',
    log_returns: bool = True
) -> pl.DataFrame:
    """
    Convenience function to calculate minute returns.
    
    Args:
        df: Input DataFrame
        price_col: Price column to use
        log_returns: Use log returns
        
    Returns:
        DataFrame with minute returns
    """
    calculator = MinuteReturnsCalculator(price_col, log_returns)
    return calculator.calculate(df)