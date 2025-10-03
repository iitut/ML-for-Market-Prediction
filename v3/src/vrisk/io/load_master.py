"""
Robust master data loader that handles missing data gracefully.
Extends the original loader with better NA handling for SIP and IEX data.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import logging
from datetime import datetime, timezone  # <-- timezone added
import warnings

from typing import Union, Optional, Dict, Any, List, Tuple

from .load_master import MasterDataLoader

logger = logging.getLogger(__name__)


class RobustMasterDataLoader(MasterDataLoader):
    """Enhanced data loader with robust NA handling for missing columns."""
    
    # Columns that can be missing in early data
    OPTIONAL_COLUMNS = {
        'sip_': ['sip_Open', 'sip_High', 'sip_Low', 'sip_Close', 'sip_Volume'],
        'iex_': ['iex_bid_price', 'iex_bid_size', 'iex_ask_price', 'iex_ask_size']
    }
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 use_parquet: bool = True,
                 validate: bool = True,
                 cache: bool = True,
                 handle_missing: str = 'fill',
                 min_data_fraction: float = 0.8):
        """
        Initialize robust data loader.
        
        Args:
            data_path: Path to master_minute.parquet or .csv
            use_parquet: Use parquet format (recommended)
            validate: Run validation checks
            cache: Cache loaded data
            handle_missing: How to handle missing columns ('fill', 'drop', 'warn')
            min_data_fraction: Minimum fraction of non-null data required per column
        """
        super().__init__(data_path, use_parquet, validate, cache)
        self.handle_missing = handle_missing
        self.min_data_fraction = min_data_fraction
        self.missing_columns_report = {}
        
    def load(self) -> pl.DataFrame:
        """
        Load master minute dataset with robust NA handling.
        
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
                null_values=["", "NA", "null", "None", "NaN"]
            )
            
        # Handle missing columns
        df = self._handle_missing_columns(df)
        
        # Ensure UTC timestamps
        if 'timestamp' in df.columns:
            df = df.with_columns(
                pl.col('timestamp').dt.replace_time_zone('UTC')
            )
        
        # Derive session_date in NY timezone
        df = self._add_session_date(df)
        
        # Apply schema and type casting with NA handling
        df = self._apply_schema_robust(df)
        
        # Handle missing data patterns
        df = self._handle_missing_data_patterns(df)
        
        # Validation (less strict for missing data)
        if self.validate:
            self._validate_data_robust(df)
            
        # Compute data hash for versioning
        self._data_hash = self._compute_hash(df)
        logger.info(f"Data hash: {self._data_hash}")
        
        # Report missing data statistics
        self._report_missing_data(df)
        
        if self.cache:
            self._data = df
            
        return df
    
    def _handle_missing_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Check for and handle missing columns."""
        expected_cols = set(self.SCHEMA.keys())
        actual_cols = set(df.columns)
        missing_cols = expected_cols - actual_cols
        
        if missing_cols:
            logger.warning(f"Missing columns detected: {missing_cols}")
            
            # Add missing columns with appropriate null values
            for col in missing_cols:
                dtype = self.SCHEMA.get(col, pl.Float64)
                if dtype in [pl.Float64, pl.Float32]:
                    df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
                elif dtype == pl.Boolean:
                    df = df.with_columns(pl.lit(False).alias(col))
                elif dtype in [pl.Int32, pl.Int64]:
                    df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
                else:
                    df = df.with_columns(pl.lit(None).alias(col))
                    
            self.missing_columns_report['missing'] = list(missing_cols)
            
        return df
    
    def _apply_schema_robust(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply schema with better error handling."""
        for col, dtype in self.SCHEMA.items():
            if col not in df.columns:
                continue
                
            try:
                if dtype in [pl.Float64, pl.Float32]:
                    # Handle numeric columns with NA
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
            except Exception as e:
                logger.warning(f"Could not cast column {col} to {dtype}: {e}")
                # Keep original dtype if casting fails
        return df
    
    def _handle_missing_data_patterns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle known missing data patterns."""
        
        # Handle IEX data (missing before 2020-08-28)
        iex_start = datetime(2020, 8, 28, tzinfo=timezone.utc)  # <-- make aware
        iex_cols = [c for c in df.columns if c.startswith('iex_')]
        
        if iex_cols and 'timestamp' in df.columns:
            # Ensure IEX columns are null before start date
            for col in iex_cols:
                df = df.with_columns(
                    pl.when(pl.col('timestamp') < pl.lit(iex_start))  # <-- compare with pl.lit
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        
        # Handle SIP data (check availability)
        sip_cols = [c for c in df.columns if c.startswith('sip_')]
        if sip_cols and 'timestamp' in df.columns:
            # Check if SIP data is mostly missing in early periods
            early_cutoff = datetime(2020, 9, 1, tzinfo=timezone.utc)  # <-- make aware
            early_data = df.filter(pl.col('timestamp') < pl.lit(early_cutoff))  # <-- compare with pl.lit
            if len(early_data) > 0:
                sip_null_fraction = early_data.select(
                    [pl.col(c).null_count() / len(early_data) for c in sip_cols]
                ).to_numpy().mean()
                
                if sip_null_fraction > 0.9:
                    logger.info("SIP data largely missing in early period - will handle appropriately")
                    
        return df
    
    def _validate_data_robust(self, df: pl.DataFrame) -> None:
        """Validation that's more tolerant of missing data."""
        logger.info("Running robust validation checks...")
        
        # 1. Check monotone timestamps (critical)
        if not df['timestamp'].is_sorted():
            raise ValueError("Timestamps are not monotonically increasing")
            
        # 2. Check for duplicates (critical)
        if df.select('timestamp').n_unique() != len(df):
            raise ValueError("Duplicate timestamps found")
            
        # 3. For other checks, just warn instead of error
        
        # Check data completeness
        for prefix in ['ohlcv_', 'iex_', 'sip_']:
            cols = [c for c in df.columns if c.startswith(prefix)]
            if cols:
                null_fractions = df.select(
                    [pl.col(c).null_count() / len(df) for c in cols]
                ).to_numpy()[0]
                
                for col, null_frac in zip(cols, null_fractions):
                    if null_frac > 0.5:
                        logger.warning(f"{col} has {null_frac:.1%} missing values")
                    if null_frac > (1 - self.min_data_fraction):
                        if self.handle_missing == 'drop':
                            logger.warning(f"Would drop {col} due to {null_frac:.1%} missing")
                        
        logger.info("Robust validation completed")
    
    def _report_missing_data(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Generate detailed missing data report."""
        report = {}
        
        # Overall missing data statistics
        total_cells = len(df) * len(df.columns)
        total_missing = sum(df[col].null_count() for col in df.columns)
        report['overall_missing_pct'] = (total_missing / total_cells) * 100
        
        # Per column statistics
        col_stats = {}
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                col_stats[col] = {
                    'null_count': null_count,
                    'null_pct': (null_count / len(df)) * 100,
                    'first_valid': df.filter(pl.col(col).is_not_null())['timestamp'].min() if 'timestamp' in df.columns else None,
                    'last_valid': df.filter(pl.col(col).is_not_null())['timestamp'].max() if 'timestamp' in df.columns else None
                }
        
        report['column_stats'] = col_stats
        
        # Check for systematic patterns
        if 'timestamp' in df.columns:
            # Group by month and check missing patterns
            monthly_missing = {}
            for prefix in ['ohlcv_', 'iex_', 'sip_']:
                cols = [c for c in df.columns if c.startswith(prefix)]
                if cols:
                    monthly_df = df.with_columns(
                        pl.col('timestamp').dt.strftime('%Y-%m').alias('month')
                    ).group_by('month').agg([
                        pl.col(cols[0]).null_count().alias('null_count'),
                        pl.len().alias('total_count')
                    ])
                    monthly_missing[prefix] = monthly_df.to_dicts()
            
            report['monthly_patterns'] = monthly_missing
        
        self.missing_columns_report.update(report)
        
        # Log summary
        logger.info(f"Overall missing data: {report['overall_missing_pct']:.2f}%")
        # after computing `col_stats`…
        if col_stats:
            worst_cols = sorted(col_stats.items(), key=lambda x: x[1]['null_pct'], reverse=True)[:5]
            pretty = ", ".join([f"{c[0]}: {c[1]['null_pct']:.1f}%" for c in worst_cols])
            logger.info(f"Top missing columns: {pretty}")

        return report
    
    def get_usable_date_range(self, required_cols: Optional[List[str]] = None) -> Tuple[datetime, datetime]:
        """
        Get date range where required columns have sufficient data.
        
        Args:
            required_cols: List of required column names (None = all)
            
        Returns:
            Tuple of (start_date, end_date) for usable data
        """
        if self._data is None:
            self.load()
            
        df = self._data
        
        if required_cols is None:
            # Default to critical columns
            required_cols = [c for c in df.columns if c.startswith('ohlcv_')]
            
        # Find first date where all required columns have data
        valid_mask = pl.all([pl.col(c).is_not_null() for c in required_cols if c in df.columns])
        valid_df = df.filter(valid_mask)
        
        if len(valid_df) == 0:
            raise ValueError(f"No valid data found for required columns: {required_cols}")
            
        start_date = valid_df['timestamp'].min()
        end_date = valid_df['timestamp'].max()
        
        logger.info(f"Usable date range: {start_date} to {end_date}")
        return start_date, end_date


def load_master_minute_robust(
    path: Union[str, Path],
    use_parquet: bool = True,
    validate: bool = True,
    handle_missing: str = 'fill',
    min_data_fraction: float = 0.8
) -> pl.DataFrame:
    """
    Convenience function to load master minute data with robust NA handling.
    
    Args:
        path: Path to data file
        use_parquet: Use parquet format
        validate: Run validation checks
        handle_missing: How to handle missing data
        min_data_fraction: Minimum non-null fraction required
        
    Returns:
        Validated Polars DataFrame with missing data handled
    """
    loader = RobustMasterDataLoader(
        path, 
        use_parquet, 
        validate,
        handle_missing=handle_missing,
        min_data_fraction=min_data_fraction
    )
    return loader.load()
