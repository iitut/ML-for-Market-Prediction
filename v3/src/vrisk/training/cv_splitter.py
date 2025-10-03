"""
Time series cross-validation with purging and embargo.
Implements anchored expanding window approach.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TimeSeriesCVSplitter:
    """
    Anchored expanding window CV with purging and embargo.
    Ensures no data leakage in time series modeling.
    """
    
    def __init__(self,
                 n_splits: int = 8,
                 min_train_days: int = 252,
                 test_days: int = 63,
                 embargo_days: int = 1,
                 purge_days: int = 1,
                 anchored: bool = True):
        """
        Initialize CV splitter.
        
        Args:
            n_splits: Number of CV folds
            min_train_days: Minimum training days
            test_days: Days in each test fold
            embargo_days: Days to embargo after test
            purge_days: Days to purge before test
            anchored: Whether to use anchored (expanding) window
        """
        self.n_splits = n_splits
        self.min_train_days = min_train_days
        self.test_days = test_days
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.anchored = anchored
        
    def split(self, 
             dates: pd.Series,
             groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for cross-validation.
        
        Args:
            dates: Series of dates for each sample
            groups: Optional group labels (not used but kept for sklearn compatibility)
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
            
        # Get unique sorted dates
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)
        
        # Calculate fold parameters
        total_test_days = self.test_days * self.n_splits
        total_embargo_days = self.embargo_days * (self.n_splits - 1)
        total_purge_days = self.purge_days * self.n_splits
        
        required_days = self.min_train_days + total_test_days + total_embargo_days + total_purge_days
        
        if n_dates < required_days:
            raise ValueError(f"Not enough data. Need {required_days} days, have {n_dates}")
            
        # Generate splits
        splits = []
        
        for i in range(self.n_splits):
            # Calculate test start position
            if self.anchored:
                # Expanding window: train always starts from beginning
                train_start_idx = 0
                test_start_idx = self.min_train_days + i * (self.test_days + self.embargo_days + self.purge_days)
            else:
                # Rolling window: train window moves forward
                window_shift = i * (self.test_days + self.embargo_days)
                train_start_idx = window_shift
                test_start_idx = train_start_idx + self.min_train_days + self.purge_days
                
            # Apply purging
            train_end_idx = test_start_idx - self.purge_days
            
            # Define test window
            test_end_idx = min(test_start_idx + self.test_days, n_dates)
            
            # Get dates for train and test
            train_dates = unique_dates[train_start_idx:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]
            
            # Get indices
            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
                
                # Log split info
                logger.info(f"Fold {i+1}/{self.n_splits}:")
                logger.info(f"  Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
                logger.info(f"  Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} days)")
                logger.info(f"  Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
                
        return iter(splits)
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits (for sklearn compatibility)."""
        return self.n_splits
    
    def visualize_splits(self, dates: pd.Series) -> pd.DataFrame:
        """
        Create visualization of CV splits.
        
        Args:
            dates: Series of dates
            
        Returns:
            DataFrame showing train/test/purge periods
        """
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
            
        unique_dates = sorted(dates.unique())
        
        # Create visualization matrix
        viz_data = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(dates)):
            train_dates = dates.iloc[train_idx].unique()
            test_dates = dates.iloc[test_idx].unique()
            
            for date in unique_dates:
                if date in train_dates:
                    status = 'train'
                elif date in test_dates:
                    status = 'test'
                else:
                    # Check if in purge period
                    if any(test_dates):
                        min_test = min(test_dates)
                        if date > max(train_dates) and date < min_test:
                            status = 'purge'
                        else:
                            status = 'embargo'
                    else:
                        status = 'unused'
                        
                viz_data.append({
                    'date': date,
                    'fold': fold_idx + 1,
                    'status': status
                })
                
        return pd.DataFrame(viz_data)


class PurgedGroupTimeSeriesSplit:
    """
    Time series split with group-aware purging.
    Ensures no information leakage across groups (e.g., same day data).
    """
    
    def __init__(self,
                 n_splits: int = 8,
                 max_train_group_size: int = None,
                 test_group_size: int = 20,
                 purge_size: int = 1):
        """
        Initialize group-aware splitter.
        
        Args:
            n_splits: Number of splits
            max_train_group_size: Max training groups (None for expanding)
            test_group_size: Number of groups in test
            purge_size: Number of groups to purge
        """
        self.n_splits = n_splits
        self.max_train_group_size = max_train_group_size
        self.test_group_size = test_group_size
        self.purge_size = purge_size
        
    def split(self,
             X: np.ndarray,
             y: np.ndarray = None,
             groups: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features
            y: Target (unused but kept for compatibility)
            groups: Group labels (e.g., dates)
            
        Yields:
            Train and test indices
        """
        if groups is None:
            raise ValueError("Groups must be provided")
            
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_splits * self.test_group_size:
            raise ValueError(f"Not enough groups for {self.n_splits} splits")
            
        group_test_size = self.test_group_size
        group_purge_size = self.purge_size
        
        for i in range(self.n_splits):
            # Calculate test groups
            test_start = n_groups - (self.n_splits - i) * group_test_size
            test_end = test_start + group_test_size
            
            # Calculate train groups
            if self.max_train_group_size is None:
                # Expanding window
                train_start = 0
            else:
                # Rolling window
                train_start = max(0, test_start - self.max_train_group_size - group_purge_size)
                
            train_end = test_start - group_purge_size
            
            # Get group indices
            train_groups = unique_groups[train_start:train_end]
            test_groups = unique_groups[test_start:test_end]
            
            # Get sample indices
            train_indices = np.where(np.isin(groups, train_groups))[0]
            test_indices = np.where(np.isin(groups, test_groups))[0]
            
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits


def create_cv_splitter(config: dict) -> TimeSeriesCVSplitter:
    """
    Create CV splitter from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured CV splitter
    """
    return TimeSeriesCVSplitter(
        n_splits=config.get('n_splits', 8),
        min_train_days=config.get('min_train_days', 252),
        test_days=config.get('test_days', 63),
        embargo_days=config.get('embargo_days', 1),
        purge_days=config.get('purge_days', 1),
        anchored=config.get('anchored', True)
    )