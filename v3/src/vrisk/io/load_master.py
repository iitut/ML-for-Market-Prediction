"""
Master data loader for minute-level market data.
Handles data validation, type casting, and basic preprocessing.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any
import hashlib
import logging

logger = logging.getLogger(__name__)


class MasterDataLoader:
    """Load and validate master minute dataset."""
    
    # Expected schema
    SCHEMA = {
        # Timestamp
        'timestamp': pl.Datetime('us', 'UTC'),
        
        # OHLCV data
        'ohlcv_open': pl.Float64,
        'ohlcv_high': pl.Float64,
        'ohlcv_low': pl.Float64,
        'ohlcv_close': pl.Float64,
        'ohlcv_volume': pl.Float64,
        'ohlcv_vwap': pl.Float64,
        
        # IEX microstructure
        'iex_bid_price': pl.Float64,
        'iex_bid_size': pl.Float64,
        'iex_ask_price': pl.Float64,
        'iex_ask_size': pl.Float64,
        
        # SIP data
        'sip_Open': pl.Float64,
        'sip_High': pl.Float64,
        'sip_Low': pl.Float64,
        'sip_Close': pl.Float64,
        'sip_Volume': pl.Float64,
        
        # Daily EOD
        'eod_Open': pl.Float64,
        'eod_High': pl.Float64,
        'eod_Low': pl.Float64,
        'eod_Close': pl.Float64,
        'eod_Volume': pl.Float64,
        
        # Macro data
        'umcsi': pl.Float64,
        'us_policy_uncertainty': pl.Float64,
        'cboe_VIXCLS': pl.Float64,
        
        # Calendar flags
        'is_holiday': pl.Boolean,
        'is_early_close': pl.Boolean,
        'early_close_minutes_local': pl.Float64,
        'is_weekly': pl.Boolean,
        'is_monthly_third_friday': pl.Boolean,
        'is_quarterly_eoq': pl.Boolean,
        'is_opx': pl.Boolean,
    }
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 use_parquet: bool = True,
                 validate: bool = True,
                 cache: bool = True):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to master_minute file
            use_parquet: Use parquet (recommended) vs CSV
            validate: Run validation checks
            cache: Cache loaded data in memory
        """
        self.data_path = Path(data_path)
        self.use_parquet = use_parquet
        self.validate = validate
        self.cache = cache
        
        self._data = None
        self._data_hash = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
    def load(self) -> pl.DataFrame:
        """
        Load master minute dataset.
        
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
            
        # Apply schema
        df = self._apply_schema(df)
        
        # Validation
        if self.validate:
            self._validate_data(df)
            
        # Compute hash
        self._data_hash = self._compute_hash(df)
        logger.info(f"Data hash: {self._data_hash}")
        
        if self.cache:
            self._data = df
            
        return df
    
    def _apply_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply expected schema with type casting."""
        for col, dtype in self.SCHEMA.items():
            if col in df.columns:
                try:
                    df = df.with_columns(pl.col(col).cast(dtype, strict=False))
                except Exception as e:
                    logger.warning(f"Could not cast {col} to {dtype}: {e}")
        return df
    
    def _add_session_date(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add session_date in NY timezone."""
        if 'timestamp' not in df.columns:
            return df
            
        df = df.with_columns(
            pl.col('timestamp')
            .dt.convert_time_zone('America/New_York')
            .dt.date()
            .alias('session_date')
        )
        return df
    
    def _validate_data(self, df: pl.DataFrame):
        """Run validation checks."""
        logger.info("Running validation checks...")
        
        # Check for required columns
        required_cols = ['timestamp', 'ohlcv_close', 'ohlcv_volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Check monotone timestamps
        if not df['timestamp'].is_sorted():
            raise ValueError("Timestamps not monotonically increasing")
            
        # Check for duplicates
        if df.select('timestamp').n_unique() != len(df):
            raise ValueError("Duplicate timestamps found")
            
        logger.info("Validation passed")
        
    def _compute_hash(self, df: pl.DataFrame) -> str:
        """Compute hash of data for versioning."""
        # Use first/last timestamp and row count
        first_ts = str(df['timestamp'].min())
        last_ts = str(df['timestamp'].max())
        n_rows = str(len(df))
        
        hash_str = f"{first_ts}_{last_ts}_{n_rows}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]


def load_master_minute(path: Union[str, Path],
                       use_parquet: bool = True,
                       validate: bool = True) -> pl.DataFrame:
    """
    Convenience function to load master minute data.
    
    Args:
        path: Path to data file
        use_parquet: Use parquet format
        validate: Run validation
        
    Returns:
        Validated Polars DataFrame
    """
    loader = MasterDataLoader(path, use_parquet, validate)
    return loader.load()