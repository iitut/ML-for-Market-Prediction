"""
Trading session and calendar utilities.
Handles regular/early close sessions, timezone conversions, and minute counting.
"""

import polars as pl
import pandas as pd
from datetime import datetime, time, timedelta
from typing import Optional, Tuple, List
import pytz
import logging

logger = logging.getLogger(__name__)


class TradingCalendar:
    """NYSE/NASDAQ trading calendar with session management."""
    
    # Regular session times (ET)
    REGULAR_OPEN = time(9, 30)
    REGULAR_CLOSE = time(16, 0)
    REGULAR_MINUTES = 390  # 9:30 AM - 4:00 PM
    
    # Early close time
    EARLY_CLOSE = time(13, 0)
    EARLY_CLOSE_MINUTES = 210  # 9:30 AM - 1:00 PM
    
    def __init__(self, timezone: str = 'America/New_York'):
        """
        Initialize trading calendar.
        
        Args:
            timezone: Trading timezone (default NYSE)
        """
        self.tz = pytz.timezone(timezone)
        self.utc = pytz.UTC
        
    def get_session_minutes(self, 
                           date: datetime,
                           is_early_close: bool = False) -> int:
        """
        Get number of trading minutes for a session.
        
        Args:
            date: Session date
            is_early_close: Whether it's an early close day
            
        Returns:
            Number of trading minutes
        """
        return self.EARLY_CLOSE_MINUTES if is_early_close else self.REGULAR_MINUTES
    
    def get_session_bounds(self,
                          date: datetime,
                          is_early_close: bool = False) -> Tuple[datetime, datetime]:
        """
        Get session open and close times in UTC.
        
        Args:
            date: Session date
            is_early_close: Whether it's an early close day
            
        Returns:
            Tuple of (open_time_utc, close_time_utc)
        """
        # Create ET datetime objects
        open_et = self.tz.localize(
            datetime.combine(date.date(), self.REGULAR_OPEN)
        )
        
        close_time = self.EARLY_CLOSE if is_early_close else self.REGULAR_CLOSE
        close_et = self.tz.localize(
            datetime.combine(date.date(), close_time)
        )
        
        # Convert to UTC
        open_utc = open_et.astimezone(self.utc)
        close_utc = close_et.astimezone(self.utc)
        
        return open_utc, close_utc
    
    def is_regular_session_time(self, 
                               timestamp: datetime,
                               is_early_close: bool = False) -> bool:
        """
        Check if timestamp is within regular trading session.
        
        Args:
            timestamp: UTC timestamp to check
            is_early_close: Whether it's an early close day
            
        Returns:
            True if within regular session
        """
        # Convert to ET
        et_time = timestamp.astimezone(self.tz)
        time_only = et_time.time()
        
        close_time = self.EARLY_CLOSE if is_early_close else self.REGULAR_CLOSE
        
        return self.REGULAR_OPEN <= time_only <= close_time
    
    def get_minute_of_session(self,
                             timestamp: datetime,
                             session_open: datetime) -> int:
        """
        Get minute number within session (1-based).
        
        Args:
            timestamp: Current timestamp
            session_open: Session open timestamp
            
        Returns:
            Minute number (1 for first minute, etc.)
        """
        diff = timestamp - session_open
        minutes = int(diff.total_seconds() / 60)
        return minutes + 1
    
    def get_last_n_minutes_mask(self,
                               df: pl.DataFrame,
                               n_minutes: int = 60) -> pl.Series:
        """
        Create mask for last N minutes of each session.
        
        Args:
            df: DataFrame with timestamp and is_early_close columns
            n_minutes: Number of minutes to include
            
        Returns:
            Boolean mask Series
        """
        # Group by session_date
        return (
            df.lazy()
            .with_columns([
                # Get session close time
                pl.when(pl.col('is_early_close'))
                .then(pl.lit(self.EARLY_CLOSE_MINUTES))
                .otherwise(pl.lit(self.REGULAR_MINUTES))
                .alias('session_minutes'),
                
                # Calculate minute of day
                pl.col('timestamp')
                .dt.convert_time_zone('America/New_York')
                .dt.time()
                .alias('time_et')
            ])
            .with_columns([
                # Calculate minutes from open
                ((pl.col('time_et').dt.total_seconds() - 
                  (self.REGULAR_OPEN.hour * 3600 + self.REGULAR_OPEN.minute * 60)) / 60)
                .cast(pl.Int32)
                .alias('minute_from_open')
            ])
            .with_columns([
                # Check if in last N minutes
                (pl.col('minute_from_open') >= 
                 (pl.col('session_minutes') - pl.lit(n_minutes)))
                .alias('is_last_n_minutes')
            ])
            .select('is_last_n_minutes')
            .collect()
            .to_series()
        )
    
    def add_session_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add session-related features to DataFrame.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with added session features
        """
        return df.with_columns([
            # Minute of session (1-based)
            ((pl.col('timestamp')
              .dt.convert_time_zone('America/New_York')
              .dt.time()
              .dt.total_seconds() - 
              (self.REGULAR_OPEN.hour * 3600 + self.REGULAR_OPEN.minute * 60)) / 60 + 1)
            .cast(pl.Int32)
            .alias('minute_of_session'),
            
            # Hour of session
            ((pl.col('minute_of_session') - 1) // 60 + 1)
            .cast(pl.Int32)
            .alias('hour_of_session'),
            
            # Is first hour
            (pl.col('minute_of_session') <= 60)
            .alias('is_first_hour'),
            
            # Is last hour (accounting for early close)
            pl.when(pl.col('is_early_close'))
            .then(pl.col('minute_of_session') > (self.EARLY_CLOSE_MINUTES - 60))
            .otherwise(pl.col('minute_of_session') > (self.REGULAR_MINUTES - 60))
            .alias('is_last_hour'),
            
            # Minutes until close
            pl.when(pl.col('is_early_close'))
            .then(self.EARLY_CLOSE_MINUTES - pl.col('minute_of_session') + 1)
            .otherwise(self.REGULAR_MINUTES - pl.col('minute_of_session') + 1)
            .alias('minutes_until_close'),
            
            # Session progress (0 to 1)
            pl.when(pl.col('is_early_close'))
            .then(pl.col('minute_of_session') / self.EARLY_CLOSE_MINUTES)
            .otherwise(pl.col('minute_of_session') / self.REGULAR_MINUTES)
            .alias('session_progress')
        ])