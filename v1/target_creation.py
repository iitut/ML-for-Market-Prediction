"""
Target variable creation for crash/boom classification and volatility prediction
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v1\\target_creation.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import *

class TargetCreator:
    def __init__(self, features_df, gamma=2.0):
        """
        Initialize target creator
        gamma: threshold for crash/boom classification (default 2 std devs)
        """
        self.features_df = features_df.copy()
        self.gamma = gamma
        self.targets_df = pd.DataFrame(index=features_df.index)
        
    def calculate_returns(self):
        """Calculate daily returns"""
        if 'close' in self.features_df.columns:
            self.features_df['daily_return'] = (
                self.features_df['close'].pct_change()
            )
            # Also calculate log returns for better statistical properties
            mask = self.features_df['close'] > 0
            self.features_df.loc[mask, 'log_return'] = np.log(
                self.features_df.loc[mask, 'close'] / 
                self.features_df.loc[mask, 'close'].shift(1)
            )
        return self.features_df
    
    def estimate_volatility(self, lookback=20):
        """
        Estimate volatility using multiple methods and combine them
        """
        vol_estimates = []
        
        # 1. Historical volatility (standard deviation of returns)
        if 'log_return' in self.features_df.columns:
            hist_vol = self.features_df['log_return'].rolling(
                window=lookback, min_periods=10
            ).std()
            vol_estimates.append(hist_vol)
        
        # 2. EWMA volatility
        if 'realized_variance' in self.features_df.columns:
            ewma_vol = np.sqrt(self.features_df['realized_variance'].ewm(
                alpha=1-EWMA_LAMBDA, min_periods=10
            ).mean())
            vol_estimates.append(ewma_vol)
        
        # 3. Parkinson estimator (if available)
        if 'parkinson_estimator' in self.features_df.columns:
            park_vol = np.sqrt(self.features_df['parkinson_estimator'].rolling(
                window=lookback, min_periods=10
            ).mean())
            vol_estimates.append(park_vol)
        
        # Combine volatility estimates (simple average)
        if vol_estimates:
            combined_vol = pd.concat(vol_estimates, axis=1).mean(axis=1)
            self.features_df['volatility_estimate'] = combined_vol
        else:
            # Fallback to simple historical volatility
            self.features_df['volatility_estimate'] = self.features_df['daily_return'].rolling(
                window=lookback, min_periods=10
            ).std()
        
        # Fill any remaining NaN values
        self.features_df['volatility_estimate'] = self.features_df['volatility_estimate'].fillna(
            method='ffill'
        ).fillna(0.02)  # Default to 2% volatility if no data
        
        return self.features_df['volatility_estimate']
    
    def create_crash_boom_targets(self):
        """
        Create crash/boom/normal classification targets
        These are FORWARD-LOOKING (predicting next day)
        """
        # Calculate returns and volatility
        self.calculate_returns()
        volatility = self.estimate_volatility()
        
        # Calculate standardized returns for NEXT day
        next_day_return = self.features_df['log_return'].shift(-1)  # Next day's return
        standardized_return = next_day_return / volatility
        
        # Create labels
        self.targets_df['standardized_return'] = standardized_return
        self.targets_df['crash'] = (standardized_return <= -self.gamma).astype(int)
        self.targets_df['boom'] = (standardized_return >= self.gamma).astype(int)
        self.targets_df['normal'] = ((standardized_return > -self.gamma) & 
                                     (standardized_return < self.gamma)).astype(int)
        
        # Create single label column for classification
        conditions = [
            self.targets_df['crash'] == 1,
            self.targets_df['boom'] == 1,
            self.targets_df['normal'] == 1
        ]
        choices = ['crash', 'boom', 'normal']
        self.targets_df['crash_boom_label'] = np.select(conditions, choices, default='normal')
        
        # Store actual next day return for analysis
        self.targets_df['next_day_return'] = next_day_return
        
        print(f"Crash/Boom target distribution:")
        print(self.targets_df['crash_boom_label'].value_counts())
        print(f"Crash rate: {self.targets_df['crash'].mean():.2%}")
        print(f"Boom rate: {self.targets_df['boom'].mean():.2%}")
        
        return self.targets_df
    
    def create_volatility_targets(self):
        """
        Create next-day volatility targets (log realized variance)
        """
        if 'realized_variance' in self.features_df.columns:
            # Next day's realized variance (shifted)
            next_rv = self.features_df['realized_variance'].shift(-1)
            
            # Log transform for better properties
            self.targets_df['log_rv_next'] = np.log(next_rv + 1e-8)  # Add small constant to avoid log(0)
            
            # Also store raw RV for reference
            self.targets_df['rv_next'] = next_rv
            
            print(f"\nVolatility target statistics:")
            print(f"Mean log(RV): {self.targets_df['log_rv_next'].mean():.4f}")
            print(f"Std log(RV): {self.targets_df['log_rv_next'].std():.4f}")
        else:
            print("Warning: No realized variance available for volatility targets")
            # Use squared returns as fallback
            if 'log_return' in self.features_df.columns:
                next_squared_return = self.features_df['log_return'].shift(-1)**2
                self.targets_df['log_rv_next'] = np.log(next_squared_return + 1e-8)
                self.targets_df['rv_next'] = next_squared_return
        
        return self.targets_df
    
    def create_quantile_targets(self, quantiles=[0.05, 0.50, 0.95]):
        """
        Create return quantile targets for distributional forecasting
        """
        if 'log_return' in self.features_df.columns:
            next_return = self.features_df['log_return'].shift(-1)
            
            # Calculate rolling quantiles based on historical data
            for q in quantiles:
                # Use expanding window to avoid look-ahead bias
                rolling_quantile = self.features_df['log_return'].expanding(
                    min_periods=20
                ).quantile(q)
                
                # Store the quantile values
                self.targets_df[f'quantile_{int(q*100)}'] = rolling_quantile.shift(1)
                
                # Store whether next return exceeds this quantile
                self.targets_df[f'exceeds_q{int(q*100)}'] = (
                    next_return > rolling_quantile
                ).astype(int)
            
            # Store actual quantiles of next return for evaluation
            self.targets_df['next_return_quantile'] = next_return
            
            print(f"\nQuantile target statistics:")
            for q in quantiles:
                col = f'quantile_{int(q*100)}'
                if col in self.targets_df.columns:
                    print(f"Q{int(q*100)}: mean={self.targets_df[col].mean():.4f}, "
                          f"std={self.targets_df[col].std():.4f}")
        
        return self.targets_df
    
    def create_all_targets(self):
        """
        Create all target variables
        """
        print("Creating target variables...")
        
        # 1. Crash/Boom classification
        self.create_crash_boom_targets()
        
        # 2. Volatility prediction
        self.create_volatility_targets()
        
        # 3. Quantile targets
        self.create_quantile_targets()
        
        # Add date for reference
        if 'date' in self.features_df.columns:
            self.targets_df['date'] = self.features_df['date']
        
        # Remove last row (no next-day target available)
        self.targets_df = self.targets_df[:-1]
        
        print(f"\nTotal targets created: {self.targets_df.shape[1]}")
        print(f"Valid samples (excluding last day): {len(self.targets_df)}")
        
        return self.targets_df
    
    def verify_no_leakage(self, train_idx, test_idx):
        """
        Verify that there's no data leakage between train and test
        """
        print("\nVerifying no data leakage...")
        
        # Check that test indices come after train indices
        max_train = max(train_idx)
        min_test = min(test_idx)
        
        if min_test <= max_train:
            print("WARNING: Test data overlaps with training data!")
            return False
        
        print(f"Train data: indices 0 to {max_train}")
        print(f"Test data: indices {min_test} to {max(test_idx)}")
        print(f"Gap between train and test: {min_test - max_train - 1} samples")
        
        # Check dates if available
        if 'date' in self.targets_df.columns:
            train_dates = self.targets_df.iloc[train_idx]['date']
            test_dates = self.targets_df.iloc[test_idx]['date']
            
            max_train_date = train_dates.max()
            min_test_date = test_dates.min()
            
            print(f"Train period: up to {max_train_date}")
            print(f"Test period: from {min_test_date}")
            
            if min_test_date <= max_train_date:
                print("WARNING: Test dates overlap with training dates!")
                return False
        
        print("✓ No data leakage detected")
        return True
    
    def get_target_correlations(self):
        """
        Calculate correlations between different targets
        """
        numeric_targets = self.targets_df.select_dtypes(include=[np.number])
        
        # Select key targets for correlation
        key_targets = []
        for col in ['standardized_return', 'log_rv_next', 'next_day_return']:
            if col in numeric_targets.columns:
                key_targets.append(col)
        
        if key_targets:
            corr_matrix = numeric_targets[key_targets].corr()
            print("\nTarget correlations:")
            print(corr_matrix)
            return corr_matrix
        
        return None

# Test the target creation
if __name__ == "__main__":
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    # Load and prepare data
    loader = DataLoader()
    master = loader.load_master_dataset()
    intraday = loader.load_intraday_data()
    
    # Engineer features
    engineer = FeatureEngineer(master, intraday)
    features = engineer.engineer_all_features()
    
    # Create targets
    target_creator = TargetCreator(features, gamma=CRASH_BOOM_THRESHOLD)
    targets = target_creator.create_all_targets()
    
    print("\nTarget creation complete!")
    print(f"Targets shape: {targets.shape}")
    
    # Verify no leakage
    train_idx, test_idx = loader.get_train_test_indices(len(targets))
    target_creator.verify_no_leakage(train_idx, test_idx)
    
    # Get correlations
    target_creator.get_target_correlations()