"""
IEX microstructure features from top-of-book data.
Handles NA values before 2020-08-28 appropriately.
"""

import polars as pl
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MicrostructureFeatures:
    """Generate microstructure features from IEX top-of-book data."""
    
    PREFIX = 'iex_'
    
    def __init__(self,
                 lookback_minutes: List[int] = [30, 60, 120],
                 handle_na: str = 'forward_fill'):
        """
        Initialize microstructure feature generator.
        
        Args:
            lookback_minutes: Window sizes for rolling stats
            handle_na: How to handle NA values ('forward_fill', 'zero', 'mean')
        """
        self.lookback_minutes = lookback_minutes
        self.handle_na = handle_na
        
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate microstructure features from minute data.
        
        Args:
            df: DataFrame with IEX bid/ask data
            
        Returns:
            DataFrame with microstructure features
        """
        logger.info("Generating microstructure features")
        
        # Calculate basic microstructure metrics
        df = self._add_spread_metrics(df)
        
        # Calculate depth imbalance
        df = self._add_depth_metrics(df)
        
        # Mid-price dynamics
        df = self._add_midprice_features(df)
        
        # Rolling microstructure stats
        df = self._add_rolling_micro_stats(df)
        
        # Aggregate to daily level
        daily_df = self._aggregate_to_daily(df)
        
        # Handle NA values appropriately
        daily_df = self._handle_na_values(daily_df)
        
        # Validate features
        self._validate_features(daily_df)
        
        return daily_df
    
    def _add_spread_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate bid-ask spread metrics."""
        return df.with_columns([
            # Mid price
            ((pl.col('iex_bid_price') + pl.col('iex_ask_price')) / 2)
            .alias(f'{self.PREFIX}mid_price'),
            
            # Absolute spread
            (pl.col('iex_ask_price') - pl.col('iex_bid_price'))
            .alias(f'{self.PREFIX}spread_abs'),
            
            # Relative spread (percentage)
            ((pl.col('iex_ask_price') - pl.col('iex_bid_price')) /
             ((pl.col('iex_bid_price') + pl.col('iex_ask_price')) / 2))
            .alias(f'{self.PREFIX}spread_rel'),
            
            # Effective spread (using trade price as proxy)
            (2 * (pl.col('ohlcv_close') - 
                  ((pl.col('iex_bid_price') + pl.col('iex_ask_price')) / 2)).abs())
            .alias(f'{self.PREFIX}effective_spread'),
            
            # Log spread (for modeling)
            ((pl.col('iex_ask_price') - pl.col('iex_bid_price')) + 0.0001).log()
            .alias(f'{self.PREFIX}log_spread')
        ])
    
    def _add_depth_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate order book depth metrics."""
        return df.with_columns([
            # Total depth
            (pl.col('iex_bid_size') + pl.col('iex_ask_size'))
            .alias(f'{self.PREFIX}total_depth'),
            
            # Depth imbalance
            ((pl.col('iex_ask_size') - pl.col('iex_bid_size')) /
             (pl.col('iex_ask_size') + pl.col('iex_bid_size') + 1))
            .alias(f'{self.PREFIX}depth_imbalance'),
            
            # Log depth
            (pl.col('iex_bid_size') + 1).log().alias(f'{self.PREFIX}log_bid_size'),
            (pl.col('iex_ask_size') + 1).log().alias(f'{self.PREFIX}log_ask_size'),
            
            # Depth ratio
            (pl.col('iex_bid_size') / (pl.col('iex_ask_size') + 1))
            .alias(f'{self.PREFIX}depth_ratio'),
            
            # Dollar depth
            (pl.col('iex_bid_size') * pl.col('iex_bid_price'))
            .alias(f'{self.PREFIX}bid_dollar_depth'),
            (pl.col('iex_ask_size') * pl.col('iex_ask_price'))
            .alias(f'{self.PREFIX}ask_dollar_depth')
        ])
    
    def _add_midprice_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate mid-price dynamics."""
        return df.with_columns([
            # Mid-price returns
            ((pl.col(f'{self.PREFIX}mid_price') / 
              pl.col(f'{self.PREFIX}mid_price').shift(1)) - 1)
            .over('session_date')
            .alias(f'{self.PREFIX}mid_return'),
            
            # Mid-price volatility (rolling)
            pl.col(f'{self.PREFIX}mid_price')
            .rolling_std(window_size=30, min_periods=10)
            .over('session_date')
            .alias(f'{self.PREFIX}mid_vol_30m'),
            
            # Price pressure (trade price vs mid)
            ((pl.col('ohlcv_close') - pl.col(f'{self.PREFIX}mid_price')) /
             pl.col(f'{self.PREFIX}mid_price'))
            .alias(f'{self.PREFIX}price_pressure'),
            
            # Quote stability (changes in spread)
            pl.col(f'{self.PREFIX}spread_rel').diff()
            .over('session_date')
            .alias(f'{self.PREFIX}spread_change')
        ])
    
    def _add_rolling_micro_stats(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add rolling microstructure statistics."""
        for window in self.lookback_minutes:
            df = df.with_columns([
                # Rolling average spread
                pl.col(f'{self.PREFIX}spread_rel')
                .rolling_mean(window_size=window, min_periods=max(1, window//3))
                .over('session_date')
                .alias(f'{self.PREFIX}spread_ma_{window}m'),
                
                # Rolling spread volatility
                pl.col(f'{self.PREFIX}spread_rel')
                .rolling_std(window_size=window, min_periods=max(1, window//3))
                .over('session_date')
                .alias(f'{self.PREFIX}spread_vol_{window}m'),
                
                # Rolling depth imbalance
                pl.col(f'{self.PREFIX}depth_imbalance')
                .rolling_mean(window_size=window, min_periods=max(1, window//3))
                .over('session_date')
                .alias(f'{self.PREFIX}imbalance_ma_{window}m'),
                
                # Rolling quote intensity (changes)
                pl.col(f'{self.PREFIX}mid_price').diff()
                .abs()
                .rolling_sum(window_size=window, min_periods=max(1, window//3))
                .over('session_date')
                .alias(f'{self.PREFIX}quote_intensity_{window}m')
            ])
            
        return df
    
    def _aggregate_to_daily(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate minute-level microstructure to daily."""
        micro_cols = [c for c in df.columns if c.startswith(self.PREFIX)]
        
        # Define aggregation rules
        agg_exprs = []
        for col in micro_cols:
            if 'spread' in col or 'imbalance' in col or 'vol' in col:
                # Mean and std for spreads and imbalances
                agg_exprs.extend([
                    pl.col(col).mean().alias(f'{col}_mean'),
                    pl.col(col).std().alias(f'{col}_std'),
                    pl.col(col).quantile(0.75).alias(f'{col}_q75')
                ])
            elif 'depth' in col or 'size' in col:
                # Sum and mean for depths
                agg_exprs.extend([
                    pl.col(col).sum().alias(f'{col}_sum'),
                    pl.col(col).mean().alias(f'{col}_mean')
                ])
            else:
                # Default to mean
                agg_exprs.append(pl.col(col).mean().alias(f'{col}_mean'))
                
        # Add special aggregations
        agg_exprs.extend([
            # Time-weighted average spread (using last hour)
            pl.when(pl.col('is_last_hour'))
            .then(pl.col(f'{self.PREFIX}spread_rel'))
            .otherwise(None)
            .mean()
            .alias(f'{self.PREFIX}spread_last_hour'),
            
            # Close spread (last value of day)
            pl.col(f'{self.PREFIX}spread_rel').last()
            .alias(f'{self.PREFIX}spread_close'),
            
            # Maximum depth imbalance
            pl.col(f'{self.PREFIX}depth_imbalance').abs().max()
            .alias(f'{self.PREFIX}max_imbalance'),
            
            # Microstructure noise proxy
            pl.col(f'{self.PREFIX}mid_return').std()
            .alias(f'{self.PREFIX}noise_proxy')
        ])
        
        # Aggregate
        daily_df = df.group_by('session_date').agg(agg_exprs)
        
        return daily_df
    
    def _handle_na_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle NA values in microstructure features."""
        micro_cols = [c for c in df.columns if c.startswith(self.PREFIX)]
        
        if self.handle_na == 'forward_fill':
            # Forward fill NAs
            for col in micro_cols:
                df = df.with_columns(pl.col(col).forward_fill())
                
        elif self.handle_na == 'zero':
            # Fill with zeros
            for col in micro_cols:
                df = df.with_columns(pl.col(col).fill_null(0))
                
        elif self.handle_na == 'mean':
            # Fill with rolling mean
            for col in micro_cols:
                df = df.with_columns(
                    pl.col(col).fill_null(
                        pl.col(col).rolling_mean(window_size=20, min_periods=1)
                    )
                )
                
        # Add NA indicator features
        df = df.with_columns([
            # Indicator for pre-IEX period
            (pl.col('session_date') < pl.lit('2020-08-28').str.to_date())
            .alias(f'{self.PREFIX}pre_data_flag')
        ])
        
        return df
    
    def _validate_features(self, df: pl.DataFrame) -> None:
        """Validate microstructure features."""
        feature_cols = [c for c in df.columns if c.startswith(self.PREFIX)]
        
        # Check spread bounds
        spread_cols = [c for c in feature_cols if 'spread_rel' in c]
        for col in spread_cols:
            max_spread = df[col].max()
            if max_spread is not None and max_spread > 0.1:  # 10% spread
                logger.warning(f"Large spread detected in {col}: {max_spread:.4f}")
                
        # Check imbalance bounds
        imb_cols = [c for c in feature_cols if 'imbalance' in c]
        for col in imb_cols:
            if df[col].abs().max() > 1:
                logger.warning(f"Imbalance out of bounds in {col}")
                
        logger.info(f"Generated {len(feature_cols)} microstructure features")