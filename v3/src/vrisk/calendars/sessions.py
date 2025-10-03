"""
Trading session and calendar utilities.
Handles regular/early close sessions, timezone conversions, and minute counting.
"""

import polars as pl
import pandas as pd  # kept for parity
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

    def __init__(self, timezone: str = "America/New_York"):
        """
        Initialize trading calendar.

        Args:
            timezone: Trading timezone (default NYSE)
        """
        self.tz = pytz.timezone(timezone)
        self.utc = pytz.UTC

    def get_session_minutes(self, date: datetime, is_early_close: bool = False) -> int:
        """Return number of trading minutes for a session."""
        return self.EARLY_CLOSE_MINUTES if is_early_close else self.REGULAR_MINUTES

    def get_session_bounds(
        self, date: datetime, is_early_close: bool = False
    ) -> Tuple[datetime, datetime]:
        """
        Get session open and close times in UTC.
        """
        open_et = self.tz.localize(datetime.combine(date.date(), self.REGULAR_OPEN))
        close_time = self.EARLY_CLOSE if is_early_close else self.REGULAR_CLOSE
        close_et = self.tz.localize(datetime.combine(date.date(), close_time))
        return open_et.astimezone(self.utc), close_et.astimezone(self.utc)

    def is_regular_session_time(
        self, timestamp: datetime, is_early_close: bool = False
    ) -> bool:
        """
        Check if timestamp (UTC) is within the regular/early trading session in local time.
        """
        et_time = timestamp.astimezone(self.tz)
        t = et_time.time()
        close_time = self.EARLY_CLOSE if is_early_close else self.REGULAR_CLOSE
        return self.REGULAR_OPEN <= t <= close_time

    def get_minute_of_session(self, timestamp: datetime, session_open: datetime) -> int:
        """Minute number (1-based) within the session."""
        return int((timestamp - session_open).total_seconds() // 60) + 1

    # ---------- internal helpers ----------

    def _session_minutes_expr(self) -> pl.Expr:
        """
        Polars expression yielding session length (minutes) per row:
        - if early close and 'early_close_minutes_local' exists → use it
        - else 210 for early close, 390 otherwise
        """
        return (
            pl.when(
                (pl.col("is_early_close").fill_null(False))
                & (pl.col("early_close_minutes_local").is_not_null())
            )
            .then(pl.col("early_close_minutes_local").cast(pl.Int32, strict=False))
            .otherwise(
                pl.when(pl.col("is_early_close").fill_null(False))
                .then(pl.lit(self.EARLY_CLOSE_MINUTES, dtype=pl.Int32))
                .otherwise(pl.lit(self.REGULAR_MINUTES, dtype=pl.Int32))
            )
            .alias("session_minutes")
        )

    def _ensure_minute_and_session_cols(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Ensure the DataFrame has:
          - 'session_minutes' (int)
          - 'minute_of_session' (int, 1..session_minutes, else null)
          - 'session_date' (date in America/New_York)
        Does not overwrite if already present.
        """
        if "timestamp" not in df.columns:
            raise ValueError("timestamp column is required to derive session features")

        # Local NY timestamp once (used for minute_of_session and session_date)
        ts_ny = pl.col("timestamp").dt.convert_time_zone("America/New_York")

        # Add session_minutes if missing
        if "session_minutes" not in df.columns:
            df = df.with_columns(self._session_minutes_expr())

        # Add session_date if missing (NY local date)
        if "session_date" not in df.columns:
            df = df.with_columns(ts_ny.dt.date().alias("session_date"))

        # Add minute_of_session if missing
        if "minute_of_session" not in df.columns:
            # minutes since local midnight
            mins_since_midnight = ts_ny.dt.hour() * 60 + ts_ny.dt.minute()
            open_minute = self.REGULAR_OPEN.hour * 60 + self.REGULAR_OPEN.minute  # 570
            raw_mins = (mins_since_midnight - pl.lit(open_minute)).alias("raw_mins")

            minute_of_session = (
                pl.when((raw_mins >= 0) & (raw_mins < pl.col("session_minutes")))
                .then((raw_mins + 1).cast(pl.Int32, strict=False))
                .otherwise(pl.lit(None, dtype=pl.Int32))
                .alias("minute_of_session")
            )

            df = df.with_columns(minute_of_session)

        return df

    # ---------- public API ----------

    def get_last_n_minutes_mask(self, df: pl.DataFrame, n_minutes: int = 60) -> pl.Series:
        """
        Boolean mask for the last N minutes of each session.
        """
        df = self._ensure_minute_and_session_cols(df)
        mask = (
            (pl.col("minute_of_session").is_not_null())
            & (pl.col("minute_of_session") > (pl.col("session_minutes") - pl.lit(n_minutes)))
        ).alias("is_last_n_minutes")
        return df.select(mask).to_series()

    def add_session_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add session-related features to DataFrame:
          - session_date (NY local date)
          - session_minutes (per-row session length)
          - minute_of_session (1..session_minutes, else null)
          - hour_of_session (1-based hour bucket)
          - is_first_hour, is_last_hour
          - minutes_until_close
          - session_progress (0..1)
        """
        # Stage 1: ensure base columns exist
        df = self._ensure_minute_and_session_cols(df)

        # Stage 2: derive features using already-existing columns
        df = df.with_columns(
            [
                # 1..6 (regular) or 1..(early close)
                (((pl.col("minute_of_session") - 1) // 60) + 1)
                .cast(pl.Int32, strict=False)
                .alias("hour_of_session"),

                # boolean flags (explicit aliases to avoid any default names)
                (
                    (pl.col("minute_of_session").is_not_null())
                    & (pl.col("minute_of_session") <= 60)
                ).alias("is_first_hour"),
                (
                    (pl.col("minute_of_session").is_not_null())
                    & (pl.col("minute_of_session") > (pl.col("session_minutes") - 60))
                ).alias("is_last_hour"),

                # inclusive countdown to close
                (pl.col("session_minutes") - pl.col("minute_of_session") + 1).alias(
                    "minutes_until_close"
                ),

                # 0..1 progress through the session
                (pl.col("minute_of_session") / pl.col("session_minutes")).alias(
                    "session_progress"
                ),
            ]
        )

        return df
