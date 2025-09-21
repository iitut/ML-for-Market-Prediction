"""
Data loading and merging module
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v1\\data_loader.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import *

class DataLoader:
    def __init__(self):
        """Initialize data loader with paths from config"""
        self.master_data = None
        self.intraday_data = None
        self.external_data = {}
        self.calendar_data = {}
        
    def load_master_dataset(self):
        """Load main master dataset"""
        print("Loading master dataset...")
        self.master_data = pd.read_csv(MASTER_DATASET)
        if 'date' in self.master_data.columns:
            self.master_data['date'] = pd.to_datetime(self.master_data['date'])
            self.master_data = self.master_data.sort_values('date').reset_index(drop=True)
        print(f"Master dataset loaded: {self.master_data.shape}")
        return self.master_data
    
    def load_intraday_data(self):
        """Load minute-level data for advanced features"""
        print("Loading intraday data...")
        try:
            # Load OHLCV processed data (primary source)
            self.intraday_data = pd.read_csv(OHLCV_PROCESSED)
            if 'timestamp' in self.intraday_data.columns:
                self.intraday_data['timestamp'] = pd.to_datetime(self.intraday_data['timestamp'])
                self.intraday_data['date'] = self.intraday_data['timestamp'].dt.date
                self.intraday_data['date'] = pd.to_datetime(self.intraday_data['date'])
            print(f"Intraday data loaded: {self.intraday_data.shape}")
        except Exception as e:
            print(f"Warning: Could not load intraday data: {e}")
            self.intraday_data = None
        return self.intraday_data
    
    def load_external_data(self):
        """Load all external market indicators"""
        print("Loading external data...")
        
        # VIX data
        try:
            vix_data = pd.read_csv(CBOE_INDEX)
            if 'date' in vix_data.columns:
                vix_data['date'] = pd.to_datetime(vix_data['date'])
            self.external_data['vix'] = vix_data
            print(f"VIX data loaded: {vix_data.shape}")
        except Exception as e:
            print(f"Warning: Could not load VIX data: {e}")
        
        # Consumer sentiment
        try:
            sentiment_data = pd.read_csv(UMCSI)
            if 'date' in sentiment_data.columns:
                sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            self.external_data['sentiment'] = sentiment_data
            print(f"Consumer sentiment loaded: {sentiment_data.shape}")
        except Exception as e:
            print(f"Warning: Could not load sentiment data: {e}")
        
        # Policy uncertainty
        try:
            policy_data = pd.read_csv(POLICY_UNCERTAINTY)
            if 'date' in policy_data.columns:
                policy_data['date'] = pd.to_datetime(policy_data['date'])
            self.external_data['policy'] = policy_data
            print(f"Policy uncertainty loaded: {policy_data.shape}")
        except Exception as e:
            print(f"Warning: Could not load policy data: {e}")
        
        # NASDAQ composite
        try:
            nasdaq_data = pd.read_csv(COMP_NASDAQ)
            if 'date' in nasdaq_data.columns:
                nasdaq_data['date'] = pd.to_datetime(nasdaq_data['date'])
            self.external_data['nasdaq'] = nasdaq_data
            print(f"NASDAQ composite loaded: {nasdaq_data.shape}")
        except Exception as e:
            print(f"Warning: Could not load NASDAQ data: {e}")
            
        return self.external_data
    
    def load_calendar_data(self):
        """Load calendar and expiration data"""
        print("Loading calendar data...")
        
        # Market calendar
        try:
            calendar = pd.read_csv(NASDAQ_CALENDAR)
            if 'date' in calendar.columns:
                calendar['date'] = pd.to_datetime(calendar['date'])
            self.calendar_data['holidays'] = calendar
            print(f"Market calendar loaded: {calendar.shape}")
        except Exception as e:
            print(f"Warning: Could not load calendar: {e}")
        
        # Options expirations
        try:
            expirations = pd.read_csv(OPTIONS_EXPIRATIONS)
            if 'date' in expirations.columns:
                expirations['date'] = pd.to_datetime(expirations['date'])
            self.calendar_data['expirations'] = expirations
            print(f"Options expirations loaded: {expirations.shape}")
        except Exception as e:
            print(f"Warning: Could not load expirations: {e}")
            
        return self.calendar_data
    
    def merge_all_data(self):
        """Merge all data sources by date"""
        print("\nMerging all data sources...")
        
        # Start with master dataset
        if self.master_data is None:
            self.load_master_dataset()
        
        merged_df = self.master_data.copy()
        
        # Track original shape
        original_shape = merged_df.shape
        print(f"Starting shape: {original_shape}")
        
        # Merge external data if available
        for name, data in self.external_data.items():
            if data is not None and 'date' in data.columns:
                before_merge = merged_df.shape[1]
                merged_df = pd.merge(merged_df, data, on='date', how='left', suffixes=('', f'_{name}'))
                after_merge = merged_df.shape[1]
                print(f"Merged {name}: added {after_merge - before_merge} columns")
        
        print(f"Final merged shape: {merged_df.shape}")
        return merged_df
    
    def validate_data_integrity(self, df):
        """Check for data leakage and temporal consistency"""
        print("\nValidating data integrity...")
        
        # Check date ordering
        if 'date' in df.columns:
            is_sorted = df['date'].is_monotonic_increasing
            print(f"Date ordering correct: {is_sorted}")
            
            # Check for duplicates
            duplicates = df['date'].duplicated().sum()
            print(f"Duplicate dates found: {duplicates}")
            
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            print(f"Warning: Columns with >50% missing values:")
            print(high_missing)
        
        # Check for future data leakage indicators
        if 'close' in df.columns:
            # Check if any future prices appear in current row
            for i in range(1, min(5, len(df))):
                future_leak = (df['close'].shift(-i) == df['close']).sum()
                if future_leak > 0:
                    print(f"Warning: Potential future leakage detected at lag {i}: {future_leak} cases")
        
        return True
    
    def get_train_test_indices(self, total_length, test_size=0.2):
        """Get train/test split indices ensuring no data leakage"""
        train_size = int(total_length * (1 - test_size))
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, total_length))
        
        print(f"\nData split:")
        print(f"Training samples: {len(train_indices)} (indices 0-{train_size-1})")
        print(f"Testing samples: {len(test_indices)} (indices {train_size}-{total_length-1})")
        
        return train_indices, test_indices

# Test the data loader
if __name__ == "__main__":
    loader = DataLoader()
    
    # Load all data
    master = loader.load_master_dataset()
    intraday = loader.load_intraday_data()
    external = loader.load_external_data()
    calendar = loader.load_calendar_data()
    
    # Merge and validate
    merged = loader.merge_all_data()
    loader.validate_data_integrity(merged)
    
    # Get train/test split
    train_idx, test_idx = loader.get_train_test_indices(len(merged))
    
    print("\nData loading complete!")
    print(f"Total features available: {merged.shape[1]}")
    print(f"Total samples: {merged.shape[0]}")