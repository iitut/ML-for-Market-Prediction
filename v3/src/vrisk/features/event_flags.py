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
                .cast(pl.Int32, strict=False)
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
             (pl.col(f'{self.PREFIX}days_from_opx') > 0))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}opx_week'),
            
            # Quad witching (March, June, Sept, Dec)
            (pl.col('is_opx') & 
             pl.col('session_date').dt.month().is_in([3, 6, 9, 12]))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}quad_witch'),
            
            # VIX expiration (Wednesday before OPX)
            (pl.col('is_opx').shift(-2) & 
             (pl.col('session_date').dt.weekday() == 2))  # Wednesday
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}vix_expiry')
        ])
    
    def _add_earnings_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add earnings season flags."""
        
        return df.with_columns([
            # Traditional earnings months (Jan, Apr, Jul, Oct)
            pl.col('session_date').dt.month().is_in([1, 4, 7, 10])
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}earnings_month'),
            
            # Peak earnings weeks (3rd and 4th week of earnings months)
            (pl.col('session_date').dt.month().is_in([1, 4, 7, 10]) &
             pl.col('session_date').dt.day().is_between(15, 31))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}peak_earnings'),
            
            # Tech earnings concentration (late Jan, late Apr, late Jul, late Oct)
            (pl.col('session_date').dt.month().is_in([1, 4, 7, 10]) &
             pl.col('session_date').dt.day() > 20)
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}tech_earnings'),
            
            # Pre-earnings drift period
            (pl.col('session_date').dt.month().is_in([1, 4, 7, 10]) &
             pl.col('session_date').dt.day() < 15)
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}pre_earnings')
        ])
    
    def _add_fomc_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add FOMC meeting flags (simplified without exact calendar)."""
        
        return df.with_columns([
            # FOMC months (simplified - every 6 weeks pattern)
            ((pl.col('session_date').dt.month().is_in([1, 3, 5, 6, 7, 9, 11, 12])) &
             (pl.col('session_date').dt.week() % 6 == 3))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}fomc_week'),
            
            # Jackson Hole (late August)
            ((pl.col('session_date').dt.month() == 8) &
             (pl.col('session_date').dt.day().is_between(20, 31)))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}jackson_hole'),
            
            # Non-farm payrolls (first Friday of month)
            ((pl.col('session_date').dt.weekday() == 4) &  # Friday
             (pl.col('session_date').dt.day() <= 7))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}nfp_day'),
            
            # CPI/PPI release (mid-month)
            ((pl.col('session_date').dt.day().is_between(10, 15)))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}inflation_data')
        ])
    
    def _add_rebalancing_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add index rebalancing flags."""
        
        # Check if we have time_trading_day_of_month
        if 'time_trading_day_of_month' not in df.columns:
            # Create a simple proxy using day of month
            df = df.with_columns([
                pl.col('session_date').dt.day().alias('time_trading_day_of_month')
            ])
        
        return df.with_columns([
            # Quarter-end rebalancing
            ((pl.col('session_date').dt.month().is_in([3, 6, 9, 12])) &
             (pl.col('session_date').dt.day() >= 25))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}quarter_end_rebal'),
            
            # Month-end rebalancing (last 3 days)
            (pl.col('time_trading_day_of_month') >= 
             (pl.col('session_date').dt.days_in_month() - 3))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}month_end_rebal'),
            
            # Russell rebalancing (June)
            ((pl.col('session_date').dt.month() == 6) &
             (pl.col('session_date').dt.day() >= 20))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}russell_rebal'),
            
            # S&P rebalancing (3rd Friday of Mar, Jun, Sep, Dec)
            (pl.col('is_opx') & 
             pl.col('session_date').dt.month().is_in([3, 6, 9, 12]))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}sp_rebal')
        ])
    
    def _add_stress_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add market stress indicator flags."""
        
        exprs = [
            # Consecutive down days
            (pl.col('daily_return') < 0)
            .cast(pl.Int32, strict=False)
            .rolling_sum(window_size=5, min_periods=1)
            .alias(f'{self.PREFIX}consecutive_down'),
        ]
        
        # Only add if column exists
        if 'liq_daily_volume' in df.columns:
            exprs.append(
                (pl.col('liq_daily_volume') > 
                 pl.col('liq_daily_volume').rolling_quantile(0.95, window_size=60, min_periods=30))
                .cast(pl.Int32, strict=False)
                .alias(f'{self.PREFIX}extreme_volume')
            )
        
        # Gap day - create from overnight_gap if available, else compute
        if 'overnight_gap' in df.columns:
            exprs.append(
                (pl.col('overnight_gap').abs() > 0.01)
                .cast(pl.Int32, strict=False)
                .alias(f'{self.PREFIX}gap_day')
            )
        else:
            # Create from daily data
            exprs.append(
                ((pl.col('daily_open') / pl.col('daily_close').shift(1)).log().abs() > 0.01)
                .cast(pl.Int32, strict=False)
                .alias(f'{self.PREFIX}gap_day')
            )
        
        # Reversal day
        exprs.append(
            ((pl.col('daily_high') > pl.col('daily_high').shift(1)) &
             (pl.col('daily_low') < pl.col('daily_low').shift(1)) &
             (pl.col('daily_close') < pl.col('daily_open')))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}reversal_pattern')
        )
        
        # Wide range day - use daily_range if available
        if 'daily_range' in df.columns:
            exprs.append(
                (pl.col('daily_range') > 
                 pl.col('daily_range').rolling_quantile(0.9, window_size=20, min_periods=10))
                .cast(pl.Int32, strict=False)
                .alias(f'{self.PREFIX}wide_range')
            )
        else:
            # Calculate on the fly
            exprs.append(
                (((pl.col('daily_high') - pl.col('daily_low')) / pl.col('daily_close')) > 
                 ((pl.col('daily_high') - pl.col('daily_low')) / pl.col('daily_close'))
                 .rolling_quantile(0.9, window_size=20, min_periods=10))
                .cast(pl.Int32, strict=False)
                .alias(f'{self.PREFIX}wide_range')
            )
        
        return df.with_columns(exprs)
    
    def _add_combination_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add combination of event flags."""
        
        exprs = []
        
        # High stress + OPX (only if extreme_volume exists)
        if f'{self.PREFIX}extreme_volume' in df.columns:
            exprs.append(
                (pl.col(f'{self.PREFIX}extreme_volume') & pl.col('is_opx'))
                .cast(pl.Int32, strict=False)
                .alias(f'{self.PREFIX}stressed_opx')
            )
        
        # Earnings + FOMC overlap
        exprs.extend([
            (pl.col(f'{self.PREFIX}earnings_month') & pl.col(f'{self.PREFIX}fomc_week'))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}earnings_fomc'),
            
            # Month-end + OPX
            (pl.col(f'{self.PREFIX}month_end_rebal') & pl.col(f'{self.PREFIX}opx_week'))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}monthend_opx'),
            
            # Multiple events
            (pl.col(f'{self.PREFIX}earnings_month').cast(pl.Int32, strict=False) +
             pl.col(f'{self.PREFIX}fomc_week').cast(pl.Int32, strict=False) +
             pl.col(f'{self.PREFIX}month_end_rebal').cast(pl.Int32, strict=False) +
             pl.col('is_opx').cast(pl.Int32, strict=False))
            .alias(f'{self.PREFIX}event_count'),
            
            # High event concentration
            ((pl.col(f'{self.PREFIX}event_count') >= 2))
            .cast(pl.Int32, strict=False)
            .alias(f'{self.PREFIX}multi_event')
        ])
        
        return df.with_columns(exprs)