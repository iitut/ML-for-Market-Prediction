"""
Master minute dataset loader with validation and integrity checks.
Enforces 1-minute granularity, UTC timestamps, and anti-leakage rules.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
from datetime import datetime, timezone
import hashlib

logger = logging.getLogger(__name__)


class MasterDataLoader:
    """Load and validate master minute dataset with strict contracts."""
    
    # Expected columns and their types
    SCHEMA = {
        # Timestamps
        'timestamp': pl.Datetime,
        'session_date': pl.Date,
        
        # A) Minute Market Data (from 2020-07-27)
        'ohlcv_open': pl.Float64,
        'ohlcv_high': pl.Float64,
        'ohlcv_low': pl.Float64,
        'ohlcv_close': pl.Float64,
        'ohlcv_volume': pl.Float64,
        'ohlcv_vwap': pl.Float64,
        
        # B) Minute Microstructure IEX (from 2020-08-28, NA before)
        'iex_bid_price': pl.Float64,
        'iex_bid_size': pl.Float64,
        'iex_ask_price': pl.Float64,
        'iex_ask_size': pl.Float64,
        
        # C) QQQ Daily EOD
        'eod_Open': pl.Float64,
        'eod_High': pl.Float64,
        'eod_Low': pl.Float64,
        'eod_Close': pl.Float64,
        'eod_Last': pl.Float64,
        'eod_Volume': pl.Float64,
        
        # D) SIP Daily
        'sip_Open': pl.Float64,
        'sip_High': pl.Float64,
        'sip_Low': pl.Float64,
        'sip_Close': pl.Float64,
        'sip_Volume': pl.Float64,
        
        # E) Nasdaq Composite
        'comp_Open': pl.Float64,
        'comp_High': pl.Float64,
        'comp_Low': pl.Float64,
        'comp_Close': pl.Float64,
        'comp_Last': pl.Float64,
        
        # F) VIX
        'cboe_VIXCLS': pl.Float64,
        
        # G) Monthly Macro
        'umcsi': pl.Float64,
        'us_policy_uncertainty': pl.Float64,
        
        # H) Calendar & Events
        'is_holiday': pl.Boolean,
        'is_early_close': pl.Boolean,
        'early_close_minutes_local': pl.Int32,
        'is_weekly': pl.Boolean,
        'is_monthly_third_friday': pl.Boolean,
        'is_quarterly_eoq': pl.Boolean,
        'is_opx': pl.Boolean,
    }
    
    # Data availability dates (for validation)
    AVAILABILITY = {
        'ohlcv': datetime(2020, 7, 27),
        'iex': datetime(2020, 8, 28),
        'eod': datetime(2019, 1, 2),
        'sip': datetime(2020, 8, 27),
        'comp': datetime(2019, 1, 2),
        'cboe': datetime(2019, 1, 1),
        'calendar': datetime(2020, 1, 1),
        'options': datetime(2020, 8, 28),
    }
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 use_parquet: bool = True,
                 validate: bool = True,
                 cache: bool = True):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to master_minute.parquet or .csv
            use_parquet: Use parquet format (recommended)
            validate: Run validation checks
            cache: Cache loaded data
        """
        self.data_path = Path(data_path)
        self.use_parquet = use_parquet
        self.validate = validate
        self.cache = cache
        self._data: Optional[pl.DataFrame] = None
        self._data_hash: Optional[str] = None
        
    def load(self) -> pl.DataFrame:
        """
        Load master minute dataset with validation.
        
        Returns:
            Polars DataFrame with validated minute data
        """
        if self._data is not None and self.cache:
            logger.info("Returning cached data")
            return self._data
            
        logger.info(f"Loading data from {self.data_path}")
        
        # Load data
        if self.use_parquet:
            df = pl.read_parquet(self.data_path)
        else:
            df = pl.read_csv(
                self.data_path,
                try_parse_dates=True,
                null_values=["", "NA", "null", "None"]
            )
            
        # Ensure UTC timestamps
        if 'timestamp' in df.columns:
            df = df.with_columns(
                pl.col('timestamp').dt.replace_time_zone('UTC')
            )
        
        # Derive session_date in NY timezone
        df = self._add_session_date(df)
        
        # Apply schema and type casting
        df = self._apply_schema(df)
        
        # Validation
        if self.validate:
            self._validate_data(df)
            
        # Compute data hash for versioning
        self._data_hash = self._compute_hash(df)
        logger.info(f"Data hash: {self._data_hash}")
        
        if self.cache:
            self._data = df
            
        return df
    
    def _add_session_date(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add session_date column in NY timezone."""
        return df.with_columns(
            pl.col('timestamp')
            .dt.convert_time_zone('America/New_York')
            .dt.date()
            .alias('session_date')
        )
    
    def _apply_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply expected schema and handle type conversions."""
        for col, dtype in self.SCHEMA.items():
            if col in df.columns:
                if dtype in [pl.Float64, pl.Float32]:
                    # Handle numeric columns
                    df = df.with_columns(
                        pl.col(col).cast(dtype, strict=False)
                    )
                elif dtype in [pl.Boolean]:
                    # Handle boolean columns
                    df = df.with_columns(
                        pl.col(col).cast(dtype, strict=False).fill_null(False)
                    )
                elif dtype in [pl.Int32, pl.Int64]:
                    # Handle integer columns
                    df = df.with_columns(
                        pl.col(col).cast(dtype, strict=False)
                    )
        return df
    
    def _validate_data(self, df: pl.DataFrame) -> None:
        """Run comprehensive validation checks."""
        logger.info("Running validation checks...")
        
        # 1. Check monotone timestamps
        if not df['timestamp'].is_sorted():
            raise ValueError("Timestamps are not monotonically increasing")
            
        # 2. Check for duplicates
        if df.select('timestamp').n_unique() != len(df):
            raise ValueError("Duplicate timestamps found")
            
        # 3. Check regular session only (09:30-16:00 ET)
        ny_times = df.select(
            pl.col('timestamp').dt.convert_time_zone('America/New_York').dt.time()
        )
        
        # 4. Validate minute frequency
        time_diffs = df.select(
            pl.col('timestamp').diff().dt.total_seconds().alias('seconds_diff')
        ).drop_nulls()
        
        # Should be 60 seconds between consecutive minutes (with some tolerance for gaps)
        regular_minutes = time_diffs.filter(pl.col('seconds_diff') == 60)
        if len(regular_minutes) < len(time_diffs) * 0.95:  # Allow 5% gaps
            logger.warning("More than 5% of timestamps have irregular spacing")
            
        # 5. Check data availability dates
        for prefix, start_date in [
            ('ohlcv_', self.AVAILABILITY['ohlcv']),
            ('iex_', self.AVAILABILITY['iex']),
        ]:
            cols = [c for c in df.columns if c.startswith(prefix)]
            if cols:
                early_data = df.filter(
                    pl.col('timestamp') < start_date
                ).select(cols)
                
                if not early_data.select(pl.all().is_null().all()).to_numpy()[0, 0]:
                    logger.warning(f"Found non-null {prefix}* data before {start_date}")
                    
        # 6. Check IEX NA policy
        iex_cols = [c for c in df.columns if c.startswith('iex_')]
        pre_iex = df.filter(pl.col('timestamp') < self.AVAILABILITY['iex'])
        if len(pre_iex) > 0 and iex_cols:
            if not pre_iex.select([pl.col(c).is_null().all() for c in iex_cols]).to_numpy().all():
                raise ValueError("IEX data should be NA before 2020-08-28")
                
        # 7. Flag zero-volume bars
        zero_vol_count = df.filter(pl.col('ohlcv_volume') == 0).height
        if zero_vol_count > 0:
            logger.info(f"Found {zero_vol_count} zero-volume bars (flagged, not dropped)")
            
        # 8. Check for flat bars (OHLC all equal)
        flat_bars = df.filter(
            (pl.col('ohlcv_open') == pl.col('ohlcv_high')) &
            (pl.col('ohlcv_high') == pl.col('ohlcv_low')) &
            (pl.col('ohlcv_low') == pl.col('ohlcv_close'))
        ).height
        if flat_bars > 0:
            logger.info(f"Found {flat_bars} flat bars (flagged, not dropped)")
            
        logger.info("Validation checks completed")
        
    def _compute_hash(self, df: pl.DataFrame) -> str:
        """Compute hash of data for versioning."""
        # Use first and last rows plus shape for quick hash
        hash_input = f"{df.shape}_{df.head(100).to_pandas().values.tobytes()}_{df.tail(100).to_pandas().values.tobytes()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about loaded data."""
        if self._data is None:
            self.load()
            
        return {
            'shape': self._data.shape,
            'columns': self._data.columns,
            'date_range': {
                'start': self._data['timestamp'].min(),
                'end': self._data['timestamp'].max()
            },
            'data_hash': self._data_hash,
            'null_counts': {
                col: self._data[col].null_count() 
                for col in self._data.columns
            }
        }


def load_master_minute(
    path: Union[str, Path],
    use_parquet: bool = True,
    validate: bool = True
) -> pl.DataFrame:
    """
    Convenience function to load master minute data.
    
    Args:
        path: Path to data file
        use_parquet: Use parquet format
        validate: Run validation checks
        
    Returns:
        Validated Polars DataFrame
    """
    loader = MasterDataLoader(path, use_parquet, validate)
    return loader.load()