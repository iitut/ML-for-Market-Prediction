"""
Event flag features for known market events and conditions.
"""

import polars as pl
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EventFlagFeatures:
    """Generate event-based binary flag features."""
    
    PREFIX = 'event_'
    
    def __init__(self):
        """Initialize event flag generator."""
        pass
        
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate event flag features.
        
        Args:
            df: DataFrame with calendar and market data
            
        Returns:
            DataFrame with event flags
        """
        logger.info("Generating event flag features")
        
        # Add options expiration flags
        df = self._add_opex_flags(df)
        
        # Add earnings season flags
        df = self._add_earnings_flags(df)
        
        # Add FOMC and economic events
        df = self._add_fomc_flags(df)
        
        # Add rebalancing events
        df = self._add_rebalancing_flags(df)
        
        # Add market stress indicators
        df = self._add_stress_flags(df)
        
        # Add combination flags
        df = self._add_combination_flags(df)
        
        return df
    
    def _add_opex_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add options expiration related flags."""
        
        if 'is_opx' not in df.columns:
            # Create OPX flag if not present
            df = df.with_columns([
                # Simplified: 3rd Friday detection
                ((pl.col('session_date').dt.weekday() == 4) &  # Friday
                 (pl.col('session_date').dt.day().is_between(15, 21)))  # 3rd week
                .cast(pl.Int32)
                .alias('is_opx')
            ])
            
        return df.with_columns([
            # Days before OPX
            pl.when(pl.col('is_opx').shift(-1))
            .then(1)
            .when(pl.col('is_opx').shift(-2))
            .then(2)
            .when(pl.col('is_opx').shift(-3))
            .then(3)
            .when(pl.col('is_opx').shift(-4))
            .then(4)
            .when(pl.col('is_opx').shift(-5))
            .then(5)
            .otherwise(0)
            .alias(f'{self.PREFIX}days_to_opx'),
            
            # Days after OPX
            pl.when(pl.col('is_opx').shift(1))
            .then(1)
            .when(pl.col('is_opx').shift(2))
            .then(2)
            .when(pl.col('is_opx').shift(3))
            .then(3)
            .otherwise(0)
            .alias(f'{self.PREFIX}days_from_opx'),
            
            # OPX week flag
            ((pl.col(f'{self.PREFIX}days_to_opx') > 0) | 
             (pl.col(f'{self.PREFIX}days_to_opx') <= 5))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}opx_week'),
            
            # Quad witching (March, June, Sept, Dec)
            (pl.col('is_opx') & 
             pl.col('session_date').dt.month().is_in([3, 6, 9, 12]))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}quad_witch'),
            
            # VIX expiration (Wednesday before OPX)
            (pl.col('is_opx').shift(-7) & 
             (pl.col('session_date').dt.weekday() == 2))  # Wednesday
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}vix_expiry')
        ])
    
    def _add_earnings_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add earnings season flags."""
        
        return df.with_columns([
            # Traditional earnings months (Jan, Apr, Jul, Oct)
            pl.col('session_date').dt.month().is_in([1, 4, 7, 10])
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}earnings_month'),
            
            # Peak earnings weeks (3rd and 4th week of earnings months)
            (pl.col('session_date').dt.month().is_in([1, 4, 7, 10]) &
             pl.col('session_date').dt.day().is_between(15, 31))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}peak_earnings'),
            
            # Tech earnings concentration (late Jan, late Apr, late Jul, late Oct)
            (pl.col('session_date').dt.month().is_in([1, 4, 7, 10]) &
             pl.col('session_date').dt.day() > 20)
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}tech_earnings'),
            
            # Pre-earnings drift period
            (pl.col('session_date').dt.month().is_in([1, 4, 7, 10]) &
             pl.col('session_date').dt.day() < 15)
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}pre_earnings')
        ])
    
    def _add_fomc_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add FOMC meeting flags (simplified without exact calendar)."""
        
        # FOMC meetings typically occur every 6 weeks
        # Simplified: mark typical FOMC weeks
        return df.with_columns([
            # FOMC months (simplified - every 6 weeks pattern)
            ((pl.col('session_date').dt.month().is_in([1, 3, 5, 6, 7, 9, 11, 12])) &
             (pl.col('session_date').dt.week() % 6 == 3))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}fomc_week'),
            
            # Jackson Hole (late August)
            ((pl.col('session_date').dt.month() == 8) &
             (pl.col('session_date').dt.day().is_between(20, 31)))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}jackson_hole'),
            
            # Non-farm payrolls (first Friday of month)
            ((pl.col('session_date').dt.weekday() == 4) &  # Friday
             (pl.col('session_date').dt.day() <= 7))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}nfp_day'),
            
            # CPI/PPI release (mid-month)
            ((pl.col('session_date').dt.day().is_between(10, 15)))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}inflation_data')
        ])
    
    def _add_rebalancing_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add index rebalancing flags."""
        
        return df.with_columns([
            # Quarter-end rebalancing
            ((pl.col('session_date').dt.month().is_in([3, 6, 9, 12])) &
             (pl.col('session_date').dt.day() >= 25))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}quarter_end_rebal'),
            
            # Month-end rebalancing (last 3 days)
            (pl.col('time_trading_day_of_month') >= 
             pl.col('time_trading_day_of_month').max().over(
                 pl.col('session_date').dt.strftime('%Y-%m')) - 3)
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}month_end_rebal'),
            
            # Russell rebalancing (June)
            ((pl.col('session_date').dt.month() == 6) &
             (pl.col('session_date').dt.day() >= 20))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}russell_rebal'),
            
            # S&P rebalancing (3rd Friday of Mar, Jun, Sep, Dec)
            (pl.col('is_opx') & 
             pl.col('session_date').dt.month().is_in([3, 6, 9, 12]))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}sp_rebal')
        ])
    
    def _add_stress_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add market stress indicator flags."""
        
        # These require actual calculation from data
        return df.with_columns([
            # Consecutive down days
            (pl.col('daily_return') < 0)
            .cast(pl.Int32)
            .rolling_sum(window_size=5, min_periods=1)
            .alias(f'{self.PREFIX}consecutive_down'),
            
            # Extreme volume day (check if column exists first)
            (pl.col('daily_volume') >
             pl.col('daily_volume').rolling_quantile(0.95, window_size=60, min_periods=30))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}extreme_volume'),
            
            # Gap day
            (pl.col('path_gap').abs() > 0.01)  # 1% gap
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}gap_day'),
            
            # Reversal day (outside reversal pattern)
            ((pl.col('daily_high') > pl.col('daily_high').shift(1)) &
             (pl.col('daily_low') < pl.col('daily_low').shift(1)) &
             (pl.col('daily_close') < pl.col('daily_open')))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}reversal_pattern'),
            
            # Wide range day
            (pl.col('path_range') > 
             pl.col('path_range').rolling_quantile(0.9, window_size=20, min_periods=10))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}wide_range')
        ])
    
    def _add_combination_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add combination of event flags."""
        
        return df.with_columns([
            # High stress + OPX
            (pl.col(f'{self.PREFIX}extreme_volume') & pl.col('is_opx'))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}stressed_opx'),
            
            # Earnings + FOMC overlap
            (pl.col(f'{self.PREFIX}earnings_month') & pl.col(f'{self.PREFIX}fomc_week'))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}earnings_fomc'),
            
            # Month-end + OPX
            (pl.col(f'{self.PREFIX}month_end_rebal') & pl.col(f'{self.PREFIX}opx_week'))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}monthend_opx'),
            
            # Multiple events
            (pl.col(f'{self.PREFIX}earnings_month').cast(pl.Int32) +
             pl.col(f'{self.PREFIX}fomc_week').cast(pl.Int32) +
             pl.col(f'{self.PREFIX}month_end_rebal').cast(pl.Int32) +
             pl.col('is_opx').cast(pl.Int32))
            .alias(f'{self.PREFIX}event_count'),
            
            # High event concentration
            ((pl.col(f'{self.PREFIX}event_count') >= 2))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}multi_event')
        ])