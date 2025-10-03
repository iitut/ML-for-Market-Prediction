"""
Liquidity features from volume and dollar volume.
Captures market depth and trading activity.
"""

import polars as pl
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class LiquidityFeatures:
    """Generate liquidity and volume-based features."""
    
    PREFIX = 'liq_'
    
    def __init__(self,
                 lookback_windows: List[int] = [5, 10, 20, 60],
                 volume_buckets: int = 10):
        """
        Initialize liquidity feature generator.
        
        Args:
            lookback_windows: Rolling window sizes in days
            volume_buckets: Number of volume profile buckets
        """
        self.lookback_windows = lookback_windows
        self.volume_buckets = volume_buckets
        
    def generate(self, minute_df: pl.DataFrame, daily_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate liquidity features.
        
        Args:
            minute_df: Minute-level data
            daily_df: Daily aggregated data
            
        Returns:
            DataFrame with liquidity features
        """
        logger.info("Generating liquidity features")
        
        # Calculate minute-level liquidity metrics
        minute_liq = self._calculate_minute_liquidity(minute_df)
        
        # Aggregate to daily
        daily_liq = self._aggregate_liquidity_daily(minute_liq)
        
        # Merge with daily data
        df = daily_df.join(daily_liq, on='session_date', how='left')
        
        # Add rolling liquidity statistics
        df = self._add_rolling_liquidity(df)
        
        # Add volume profile features
        df = self._add_volume_profile(df, minute_df)
        
        # Add relative liquidity measures
        df = self._add_relative_liquidity(df)
        
        return df
    
    def _calculate_minute_liquidity(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate minute-level liquidity metrics."""
        return df.with_columns([
            # Dollar volume
            (pl.col('ohlcv_volume') * pl.col('ohlcv_vwap'))
            .alias(f'{self.PREFIX}dollar_volume'),
            
            # Log dollar volume
            ((pl.col('ohlcv_volume') * pl.col('ohlcv_vwap')) + 1).log()
            .alias(f'{self.PREFIX}log_dollar_volume'),
            
            # Amihud illiquidity (|return| / dollar_volume)
            (pl.col('minute_return').abs() / 
             (pl.col('ohlcv_volume') * pl.col('ohlcv_vwap') + 1))
            .alias(f'{self.PREFIX}amihud'),
            
            # Volume rate (volume / time)
            pl.col('ohlcv_volume').alias(f'{self.PREFIX}volume_rate'),
            
            # Trade intensity (approximated by volume volatility)
            pl.col('ohlcv_volume')
            .rolling_std(window_size=30, min_periods=10)
            .over('session_date')
            .alias(f'{self.PREFIX}trade_intensity'),
            
            # Kyle's lambda approximation (price impact)
            (pl.col('minute_return').abs() / (pl.col('ohlcv_volume').sqrt() + 1))
            .alias(f'{self.PREFIX}kyle_lambda')
        ])
    
    def _aggregate_liquidity_daily(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate liquidity metrics to daily level."""
        return (
            df.group_by('session_date')
            .agg([
                # Total volume and dollar volume
                pl.col('ohlcv_volume').sum().alias(f'{self.PREFIX}daily_volume'),
                pl.col(f'{self.PREFIX}dollar_volume').sum()
                .alias(f'{self.PREFIX}daily_dollar_volume'),
                
                # Log transformations
                (pl.col('ohlcv_volume').sum() + 1).log()
                .alias(f'{self.PREFIX}log_daily_volume'),
                
                # Average liquidity measures
                pl.col(f'{self.PREFIX}amihud').mean()
                .alias(f'{self.PREFIX}daily_amihud'),
                
                pl.col(f'{self.PREFIX}kyle_lambda').mean()
                .alias(f'{self.PREFIX}daily_kyle_lambda'),
                
                # Volume concentration (Herfindahl)
                (pl.col('ohlcv_volume').pow(2).sum() / 
                 pl.col('ohlcv_volume').sum().pow(2))
                .alias(f'{self.PREFIX}volume_herfindahl'),
                
                # Intraday volume volatility
                pl.col('ohlcv_volume').std()
                .alias(f'{self.PREFIX}volume_volatility'),
                
                # Volume skewness
                ((pl.col('ohlcv_volume') - pl.col('ohlcv_volume').mean()).pow(3).sum() /
                 (pl.len() * pl.col('ohlcv_volume').std().pow(3)))
                .alias(f'{self.PREFIX}volume_skew'),
                
                # Max volume spike
                (pl.col('ohlcv_volume').max() / pl.col('ohlcv_volume').mean())
                .alias(f'{self.PREFIX}max_volume_spike')
            ])
        )
    
    def _add_rolling_liquidity(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling liquidity statistics."""
        for window in self.lookback_windows:
            df = df.with_columns([
                # Rolling average volume
                pl.col(f'{self.PREFIX}daily_volume')
                .rolling_mean(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}volume_ma_{window}d'),
                
                # Rolling volume ratio (current / MA)
                (pl.col(f'{self.PREFIX}daily_volume') / 
                 pl.col(f'{self.PREFIX}daily_volume')
                 .rolling_mean(window_size=window, min_periods=max(1, window//2)))
                .alias(f'{self.PREFIX}volume_ratio_{window}d'),
                
                # Rolling Amihud illiquidity
                pl.col(f'{self.PREFIX}daily_amihud')
                .rolling_mean(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}amihud_ma_{window}d'),
                
                # Rolling dollar volume
                pl.col(f'{self.PREFIX}daily_dollar_volume')
                .rolling_mean(window_size=window, min_periods=max(1, window//2))
                .alias(f'{self.PREFIX}dollar_volume_ma_{window}d'),
                
                # Volume trend (linear regression slope simplified)
                pl.col(f'{self.PREFIX}log_daily_volume')
                .rolling_map(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    window_size=window,
                    min_periods=max(2, window//2)
                )
                .alias(f'{self.PREFIX}volume_trend_{window}d')
            ])
            
        return df
    
    def _add_volume_profile(self, df: pl.DataFrame, minute_df: pl.DataFrame) -> pl.DataFrame:
        """Add volume profile features (distribution across day)."""
        
        # Calculate volume profile statistics
        profile_stats = (
            minute_df.with_columns([
                # Assign time buckets (e.g., 10 buckets across day)
                ((pl.col('minute_of_session') - 1) * self.volume_buckets // 390)
                .alias('time_bucket')
            ])
            .group_by(['session_date', 'time_bucket'])
            .agg([
                pl.col('ohlcv_volume').sum().alias('bucket_volume')
            ])
            .group_by('session_date')
            .agg([
                # Volume concentration in specific periods
                pl.when(pl.col('time_bucket') == 0)  # First bucket (open)
                .then(pl.col('bucket_volume'))
                .otherwise(0)
                .sum()
                .alias(f'{self.PREFIX}open_volume'),
                
                pl.when(pl.col('time_bucket') == self.volume_buckets - 1)  # Last bucket (close)
                .then(pl.col('bucket_volume'))
                .otherwise(0)
                .sum()
                .alias(f'{self.PREFIX}close_volume'),
                
                # Volume distribution entropy
                (-(pl.col('bucket_volume') / pl.col('bucket_volume').sum()) * 
                 (pl.col('bucket_volume') / pl.col('bucket_volume').sum()).log())
                .sum()
                .alias(f'{self.PREFIX}volume_entropy')
            ])
        )
        
        # Calculate VWAP deviation
        vwap_stats = (
            minute_df.group_by('session_date')
            .agg([
                # VWAP for the day
                ((pl.col('ohlcv_close') * pl.col('ohlcv_volume')).sum() / 
                 pl.col('ohlcv_volume').sum())
                .alias(f'{self.PREFIX}daily_vwap'),
                
                # VWAP deviation
                ((pl.col('ohlcv_close') - 
                  (pl.col('ohlcv_close') * pl.col('ohlcv_volume')).sum() / 
                  pl.col('ohlcv_volume').sum()).abs().mean())
                .alias(f'{self.PREFIX}vwap_deviation')
            ])
        )
        
        # Merge profile statistics
        df = df.join(profile_stats, on='session_date', how='left')
        df = df.join(vwap_stats, on='session_date', how='left')
        
        # Add open/close volume ratios
        df = df.with_columns([
            (pl.col(f'{self.PREFIX}open_volume') / 
             pl.col(f'{self.PREFIX}daily_volume'))
            .alias(f'{self.PREFIX}open_volume_ratio'),
            
            (pl.col(f'{self.PREFIX}close_volume') / 
             pl.col(f'{self.PREFIX}daily_volume'))
            .alias(f'{self.PREFIX}close_volume_ratio')
        ])
        
        return df
    
    def _add_relative_liquidity(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add relative liquidity measures."""
        return df.with_columns([
            # Relative volume (percentile rank in last 60 days)
            pl.col(f'{self.PREFIX}daily_volume')
            .rank()
            .over(pl.col('session_date').rolling_index(window_size='60d'))
            .alias(f'{self.PREFIX}volume_percentile'),
            
            # Abnormal volume flag
            (pl.col(f'{self.PREFIX}daily_volume') > 
             pl.col(f'{self.PREFIX}daily_volume')
             .rolling_mean(window_size=20, min_periods=10)
             .rolling_quantile(0.95, window_size=60, min_periods=30))
            .alias(f'{self.PREFIX}abnormal_volume'),
            
            # Liquidity regime (high/medium/low based on rolling quantiles)
            pl.when(pl.col(f'{self.PREFIX}daily_dollar_volume') > 
                   pl.col(f'{self.PREFIX}daily_dollar_volume')
                   .rolling_quantile(0.67, window_size=60, min_periods=30))
            .then(2)  # High liquidity
            .when(pl.col(f'{self.PREFIX}daily_dollar_volume') > 
                 pl.col(f'{self.PREFIX}daily_dollar_volume')
                 .rolling_quantile(0.33, window_size=60, min_periods=30))
            .then(1)  # Medium liquidity
            .otherwise(0)  # Low liquidity
            .alias(f'{self.PREFIX}regime')
        ])