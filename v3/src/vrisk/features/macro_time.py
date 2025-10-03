"""
Macro features with proper lag and time/seasonality features.
Ensures no lookahead bias in macro data usage.
"""

import polars as pl
import numpy as np
from typing import List, Optional, Dict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MacroTimeFeatures:
    """Generate macro and time-based features with proper lag."""
    
    MACRO_PREFIX = 'macro_'
    TIME_PREFIX = 'time_'
    
    # Lag for macro data (business days)
    MACRO_LAG_DAYS = 3
    
    def __init__(self):
        """Initialize macro and time feature generator."""
        pass
        
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate macro and time features.
        
        Args:
            df: DataFrame with macro data and dates
            
        Returns:
            DataFrame with macro and time features
        """
        logger.info("Generating macro and time features")
        
        # Add properly lagged macro features
        df = self._add_lagged_macro(df)
        
        # Add macro transformations
        df = self._add_macro_transformations(df)
        
        # Add time features
        df = self._add_time_features(df)
        
        # Add seasonality features
        df = self._add_seasonality_features(df)
        
        # Add calendar anomalies
        df = self._add_calendar_anomalies(df)
        
        return df
    
    def _add_lagged_macro(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add macro features with proper lag to prevent lookahead."""
        
        # Macro columns that need lagging
        macro_cols = ['umcsi', 'us_policy_uncertainty', 'cboe_VIXCLS']
        
        for col in macro_cols:
            if col in df.columns:
                # Apply business day lag
                df = df.with_columns([
                    # Lag by MACRO_LAG_DAYS business days
                    pl.col(col).shift(self.MACRO_LAG_DAYS)
                    .alias(f'{self.MACRO_PREFIX}{col}_lagged'),
                    
                    # Also create change features
                    (pl.col(col).shift(self.MACRO_LAG_DAYS) - 
                     pl.col(col).shift(self.MACRO_LAG_DAYS + 20))
                    .alias(f'{self.MACRO_PREFIX}{col}_change_20d'),
                    
                    # Percentage change
                    ((pl.col(col).shift(self.MACRO_LAG_DAYS) / 
                      pl.col(col).shift(self.MACRO_LAG_DAYS + 20)) - 1)
                    .alias(f'{self.MACRO_PREFIX}{col}_pct_change_20d')
                ])
                
        return df
    
    def _add_macro_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add macro feature transformations and interactions."""
        
        # VIX features (if available)
        if f'{self.MACRO_PREFIX}cboe_VIXCLS_lagged' in df.columns:
            df = df.with_columns([
                # VIX term structure proxy (current vs MA)
                (pl.col(f'{self.MACRO_PREFIX}cboe_VIXCLS_lagged') / 
                 pl.col(f'{self.MACRO_PREFIX}cboe_VIXCLS_lagged')
                 .rolling_mean(window_size=20, min_periods=10))
                .alias(f'{self.MACRO_PREFIX}vix_term_structure'),
                
                # VIX regime (high/low)
                (pl.col(f'{self.MACRO_PREFIX}cboe_VIXCLS_lagged') > 20)
                .cast(pl.Int32)
                .alias(f'{self.MACRO_PREFIX}high_vix_regime'),
                
                # VIX percentile
                pl.col(f'{self.MACRO_PREFIX}cboe_VIXCLS_lagged')
                .rank()
                .over(pl.col('session_date').rolling_index(window_size='252d'))
                .alias(f'{self.MACRO_PREFIX}vix_percentile')
            ])
            
        # Consumer sentiment features
        if f'{self.MACRO_PREFIX}umcsi_lagged' in df.columns:
            df = df.with_columns([
                # Sentiment vs trend
                (pl.col(f'{self.MACRO_PREFIX}umcsi_lagged') - 
                 pl.col(f'{self.MACRO_PREFIX}umcsi_lagged')
                 .rolling_mean(window_size=120, min_periods=60))
                .alias(f'{self.MACRO_PREFIX}sentiment_vs_trend'),
                
                # Sentiment momentum
                pl.col(f'{self.MACRO_PREFIX}umcsi_change_20d')
                .rolling_mean(window_size=60, min_periods=30)
                .alias(f'{self.MACRO_PREFIX}sentiment_momentum')
            ])
            
        # Policy uncertainty features
        if f'{self.MACRO_PREFIX}us_policy_uncertainty_lagged' in df.columns:
            df = df.with_columns([
                # Uncertainty spike
                (pl.col(f'{self.MACRO_PREFIX}us_policy_uncertainty_lagged') > 
                 pl.col(f'{self.MACRO_PREFIX}us_policy_uncertainty_lagged')
                 .rolling_quantile(0.8, window_size=252, min_periods=126))
                .cast(pl.Int32)
                .alias(f'{self.MACRO_PREFIX}uncertainty_spike'),
                
                # Uncertainty trend
                pl.col(f'{self.MACRO_PREFIX}us_policy_uncertainty_lagged')
                .rolling_map(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    window_size=60,
                    min_periods=30
                )
                .alias(f'{self.MACRO_PREFIX}uncertainty_trend')
            ])
            
        return df
    
    def _add_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add time-based features."""
        return df.with_columns([
            # Day of week (0 = Monday)
            pl.col('session_date').dt.weekday().alias(f'{self.TIME_PREFIX}day_of_week'),
            
            # Day of month
            pl.col('session_date').dt.day().alias(f'{self.TIME_PREFIX}day_of_month'),
            
            # Week of year
            pl.col('session_date').dt.week().alias(f'{self.TIME_PREFIX}week_of_year'),
            
            # Month
            pl.col('session_date').dt.month().alias(f'{self.TIME_PREFIX}month'),
            
            # Quarter
            pl.col('session_date').dt.quarter().alias(f'{self.TIME_PREFIX}quarter'),
            
            # Year
            pl.col('session_date').dt.year().alias(f'{self.TIME_PREFIX}year'),
            
            # Trading day of month (business day count)
            pl.col('session_date').rank()
            .over(pl.col('session_date').dt.strftime('%Y-%m'))
            .alias(f'{self.TIME_PREFIX}trading_day_of_month'),
            
            # Trading day of year
            pl.col('session_date').rank()
            .over(pl.col('session_date').dt.year())
            .alias(f'{self.TIME_PREFIX}trading_day_of_year'),
            
            # Days since last holiday
            # (Would need holiday calendar for exact calculation)
            pl.when(pl.col('is_holiday').shift(1))
            .then(1)
            .otherwise(pl.lit(None))
            .forward_fill()
            .alias(f'{self.TIME_PREFIX}days_since_holiday'),
            
            # Days until next holiday (simplified)
            pl.when(pl.col('is_holiday').shift(-1))
            .then(1)
            .otherwise(pl.lit(None))
            .backward_fill()
            .alias(f'{self.TIME_PREFIX}days_until_holiday')
        ])
    
    def _add_seasonality_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add seasonality encoding features."""
        return df.with_columns([
            # Sine/cosine encoding for month
            (np.sin(2 * np.pi * pl.col(f'{self.TIME_PREFIX}month') / 12))
            .alias(f'{self.TIME_PREFIX}month_sin'),
            
            (np.cos(2 * np.pi * pl.col(f'{self.TIME_PREFIX}month') / 12))
            .alias(f'{self.TIME_PREFIX}month_cos'),
            
            # Sine/cosine encoding for day of week
            (np.sin(2 * np.pi * pl.col(f'{self.TIME_PREFIX}day_of_week') / 5))
            .alias(f'{self.TIME_PREFIX}dow_sin'),
            
            (np.cos(2 * np.pi * pl.col(f'{self.TIME_PREFIX}day_of_week') / 5))
            .alias(f'{self.TIME_PREFIX}dow_cos'),
            
            # Sine/cosine encoding for day of month
            (np.sin(2 * np.pi * pl.col(f'{self.TIME_PREFIX}day_of_month') / 31))
            .alias(f'{self.TIME_PREFIX}dom_sin'),
            
            (np.cos(2 * np.pi * pl.col(f'{self.TIME_PREFIX}day_of_month') / 31))
            .alias(f'{self.TIME_PREFIX}dom_cos'),
            
            # Quarter indicators
            (pl.col(f'{self.TIME_PREFIX}quarter') == 1).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}is_q1'),
            
            (pl.col(f'{self.TIME_PREFIX}quarter') == 4).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}is_q4'),
            
            # Month-end effects
            (pl.col(f'{self.TIME_PREFIX}trading_day_of_month') >= 
             pl.col(f'{self.TIME_PREFIX}trading_day_of_month').max().over(
                 pl.col('session_date').dt.strftime('%Y-%m')) - 3)
            .cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}is_month_end'),
            
            # January effect
            (pl.col(f'{self.TIME_PREFIX}month') == 1).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}is_january'),
            
            # December effect  
            (pl.col(f'{self.TIME_PREFIX}month') == 12).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}is_december')
        ])
    
    def _add_calendar_anomalies(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add known calendar anomaly features."""
        return df.with_columns([
            # Monday effect
            (pl.col(f'{self.TIME_PREFIX}day_of_week') == 0).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}is_monday'),
            
            # Friday effect
            (pl.col(f'{self.TIME_PREFIX}day_of_week') == 4).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}is_friday'),
            
            # Turn of month (last 2 and first 3 days)
            ((pl.col(f'{self.TIME_PREFIX}trading_day_of_month') <= 3) | 
             (pl.col(f'{self.TIME_PREFIX}trading_day_of_month') >= 
              pl.col(f'{self.TIME_PREFIX}trading_day_of_month').max().over(
                  pl.col('session_date').dt.strftime('%Y-%m')) - 2))
            .cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}turn_of_month'),
            
            # Options expiration week (simplified - 3rd Friday)
            ((pl.col(f'{self.TIME_PREFIX}week_of_year') % 4 == 3) & 
             (pl.col(f'{self.TIME_PREFIX}day_of_week') >= 3))
            .cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}opex_week'),
            
            # Pre-holiday
            pl.col('is_holiday').shift(-1).fill_null(False).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}pre_holiday'),
            
            # Post-holiday
            pl.col('is_holiday').shift(1).fill_null(False).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}post_holiday'),
            
            # Summer months (May-August)
            pl.col(f'{self.TIME_PREFIX}month').is_between(5, 8).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}is_summer'),
            
            # Tax loss selling season (November-December)
            pl.col(f'{self.TIME_PREFIX}month').is_between(11, 12).cast(pl.Int32)
            .alias(f'{self.TIME_PREFIX}tax_loss_season')
        ])