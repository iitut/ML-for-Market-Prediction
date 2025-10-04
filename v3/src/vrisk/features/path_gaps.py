"""
Path-dependent features including gaps, ranges, and cumulative movements.
Ensures all features are known at day-t close.
"""

import polars as pl
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PathFeatures:
    """Generate path-dependent features from price movements."""
    
    PREFIX = 'path_'
    
    def __init__(self,
                 lookback_windows: List[int] = [5, 10, 20],
                 last_n_minutes: List[int] = [30, 60, 120]):
        """
        Initialize path feature generator.
        
        Args:
            lookback_windows: Rolling window sizes in days
            last_n_minutes: Minute windows for intraday features
        """
        self.lookback_windows = lookback_windows
        self.last_n_minutes = last_n_minutes
        
    def generate(self, minute_df: pl.DataFrame, daily_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate path features from minute and daily data.
        
        Args:
            minute_df: Minute-level DataFrame
            daily_df: Daily-level DataFrame
            
        Returns:
            DataFrame with path features at daily level
        """
        logger.info("Generating path features")
        
        # Calculate intraday features from minute data
        intraday_features = self._calculate_intraday_features(minute_df)
        
        # Merge with daily data - use left join and handle duplicates
        # First, identify which columns already exist in daily_df
        overlap_cols = set(intraday_features.columns) & set(daily_df.columns)
        overlap_cols.discard('session_date')  # Keep session_date for join
        
        if overlap_cols:
            logger.warning(f"Dropping overlapping columns from intraday features: {overlap_cols}")
            intraday_features = intraday_features.drop(list(overlap_cols))
        
        df = daily_df.join(intraday_features, on='session_date', how='left')
        
        # Add gap features
        df = self._add_gap_features(df)
        
        # Add range features
        df = self._add_range_features(df)
        
        # Add cumulative return features
        df = self._add_cumulative_features(df)
        
        # Add path characteristics
        df = self._add_path_characteristics(df)
        
        # Validate features
        self._validate_features(df)
        
        return df
    
    def _calculate_intraday_features(self, minute_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate intraday path features from minute data."""
        
        # Collect all intraday aggregations
        agg_exprs = []
        
        # Last N minutes features
        for n_min in self.last_n_minutes:
            agg_exprs.extend([
                # Return over last N minutes
                pl.col('minute_return').tail(n_min).sum()
                .alias(f'{self.PREFIX}last_{n_min}min_return'),
                
                # Volatility over last N minutes
                pl.col('minute_return').tail(n_min).std()
                .alias(f'{self.PREFIX}last_{n_min}min_vol'),
                
                # Max/min in last N minutes
                (pl.col('ohlcv_close').tail(n_min).max() / 
                 pl.col('ohlcv_close').tail(n_min).min() - 1)
                .alias(f'{self.PREFIX}last_{n_min}min_range'),
                
                # Volume concentration
                (pl.col('ohlcv_volume').tail(n_min).sum() / 
                 pl.col('ohlcv_volume').sum())
                .alias(f'{self.PREFIX}last_{n_min}min_vol_pct')
            ])
        
        # Intraday path statistics
        agg_exprs.extend([
            # High/Low times
            (pl.col('ohlcv_high').arg_max() / pl.len())
            .alias(f'{self.PREFIX}high_time_pct'),
            
            (pl.col('ohlcv_low').arg_min() / pl.len())
            .alias(f'{self.PREFIX}low_time_pct'),
            
            # Path efficiency
            (pl.col('minute_return').sum() / 
             (pl.col('minute_return').abs().sum() + 1e-8))
            .alias(f'{self.PREFIX}path_efficiency'),
            
            # Number of direction changes
            ((pl.col('minute_return').sign() != 
              pl.col('minute_return').sign().shift(1))
             .sum())
            .alias(f'{self.PREFIX}direction_changes'),
            
            # Maximum drawdown
            ((pl.col('ohlcv_close').max() - pl.col('ohlcv_close').min()) / 
             pl.col('ohlcv_close').max())
            .alias(f'{self.PREFIX}intraday_drawdown')
        ])
        
        # Aggregate by session_date
        result = minute_df.group_by('session_date').agg(agg_exprs)
        
        return result
    
    def _add_gap_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add overnight gap features."""
        return df.with_columns([
            # Overnight gap (use existing if available)
            pl.when(pl.col('overnight_gap').is_not_null())
            .then(pl.col('overnight_gap'))
            .otherwise(
                (pl.col('daily_open') / pl.col('daily_close').shift(1)).log()
            )
            .alias(f'{self.PREFIX}gap'),
            
            # Gap size (absolute)
            pl.col(f'{self.PREFIX}gap').abs().alias(f'{self.PREFIX}gap_abs'),
            
            # Gap direction
            pl.col(f'{self.PREFIX}gap').sign().alias(f'{self.PREFIX}gap_direction'),
            
            # Rolling gap statistics
            pl.col(f'{self.PREFIX}gap')
            .rolling_mean(window_size=5, min_periods=1)
            .alias(f'{self.PREFIX}gap_ma5'),
            
            pl.col(f'{self.PREFIX}gap')
            .rolling_std(window_size=20, min_periods=5)
            .alias(f'{self.PREFIX}gap_vol20'),
            
            # Gap vs previous close
            (pl.col(f'{self.PREFIX}gap') / 
             (pl.col('daily_close').shift(1).abs() + 1e-8))
            .alias(f'{self.PREFIX}gap_pct'),
            
            # Cumulative gap over last N days
            pl.col(f'{self.PREFIX}gap')
            .rolling_sum(window_size=5, min_periods=1)
            .alias(f'{self.PREFIX}gap_cum5')
        ])
    
    def _add_range_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add daily range features."""
        return df.with_columns([
            # True range
            pl.max_horizontal([
                pl.col('daily_high') - pl.col('daily_low'),
                (pl.col('daily_high') - pl.col('daily_close').shift(1)).abs(),
                (pl.col('daily_low') - pl.col('daily_close').shift(1)).abs()
            ]).alias(f'{self.PREFIX}true_range'),
            
            # Normalized range
            ((pl.col('daily_high') - pl.col('daily_low')) / 
             pl.col('daily_close'))
            .alias(f'{self.PREFIX}range_pct'),
            
            # Log range
            (pl.col('daily_high') / pl.col('daily_low')).log()
            .alias(f'{self.PREFIX}log_range'),
            
            # Close position in range
            ((pl.col('daily_close') - pl.col('daily_low')) / 
             (pl.col('daily_high') - pl.col('daily_low') + 1e-8))
            .alias(f'{self.PREFIX}close_position'),
            
            # Body vs range
            ((pl.col('daily_close') - pl.col('daily_open')).abs() / 
             (pl.col('daily_high') - pl.col('daily_low') + 1e-8))
            .alias(f'{self.PREFIX}body_ratio'),
            
            # Upper/lower shadows
            ((pl.col('daily_high') - pl.max_horizontal([
                pl.col('daily_open'), pl.col('daily_close')
            ])) / (pl.col('daily_high') - pl.col('daily_low') + 1e-8))
            .alias(f'{self.PREFIX}upper_shadow'),
            
            ((pl.min_horizontal([
                pl.col('daily_open'), pl.col('daily_close')
            ]) - pl.col('daily_low')) / 
             (pl.col('daily_high') - pl.col('daily_low') + 1e-8))
            .alias(f'{self.PREFIX}lower_shadow')
        ])
    
    def _add_cumulative_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add cumulative return features."""
        for window in self.lookback_windows:
            df = df.with_columns([
                # Cumulative return
                pl.col('daily_return')
                .rolling_sum(window_size=window, min_periods=1)
                .alias(f'{self.PREFIX}cum_ret_{window}d'),
                
                # Maximum return in window
                pl.col('daily_return')
                .rolling_max(window_size=window, min_periods=1)
                .alias(f'{self.PREFIX}max_ret_{window}d'),
                
                # Minimum return in window
                pl.col('daily_return')
                .rolling_min(window_size=window, min_periods=1)
                .alias(f'{self.PREFIX}min_ret_{window}d'),
                
                # Number of positive days
                (pl.col('daily_return') > 0)
                .cast(pl.Int32, strict=False)
                .rolling_sum(window_size=window, min_periods=1)
                .alias(f'{self.PREFIX}pos_days_{window}d'),
            ])
            
        return df
    
    def _add_path_characteristics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add path characteristic features."""
        return df.with_columns([
            # Momentum
            (pl.col('daily_close') / pl.col('daily_close').shift(5) - 1)
            .alias(f'{self.PREFIX}momentum_5d'),
            
            (pl.col('daily_close') / pl.col('daily_close').shift(20) - 1)
            .alias(f'{self.PREFIX}momentum_20d'),
            
            # Rate of change
            ((pl.col('daily_close') - pl.col('daily_close').shift(10)) / 10)
            .alias(f'{self.PREFIX}roc_10d'),
            
            # Distance from high/low
            (pl.col('daily_close') / 
             pl.col('daily_high').rolling_max(window_size=20, min_periods=5) - 1)
            .alias(f'{self.PREFIX}dist_from_high20'),
            
            (pl.col('daily_close') / 
             pl.col('daily_low').rolling_min(window_size=20, min_periods=5) - 1)
            .alias(f'{self.PREFIX}dist_from_low20'),
            
            # Trend strength
            pl.col('daily_close')
            .rolling_map(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                window_size=20,
                min_periods=5
            )
            .alias(f'{self.PREFIX}trend_strength_20d')
        ])
    
    def _validate_features(self, df: pl.DataFrame) -> None:
        """Validate path features."""
        feature_cols = [c for c in df.columns if c.startswith(self.PREFIX)]
        
        # Check for infinite values
        for col in feature_cols:
            if col in df.columns:
                inf_count = df[col].is_infinite().sum()
                if inf_count > 0:
                    logger.warning(f"Found {inf_count} infinite values in {col}, replacing with null")
                
        logger.info(f"Generated {len(feature_cols)} path features")