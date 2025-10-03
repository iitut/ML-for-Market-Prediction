"""
Feature engineering module for volatility and market features
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v1\\feature_engineering.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import *

class FeatureEngineer:
    def __init__(self, daily_data, intraday_data=None):
        """
        Initialize with daily and optionally intraday data
        """
        self.daily_data = daily_data.copy()
        self.intraday_data = intraday_data
        self.features_df = daily_data.copy()
        
    def calculate_realized_variance(self, window=1):
        """Calculate realized variance from intraday returns"""
        if self.intraday_data is None:
            print("Warning: No intraday data for RV calculation")
            return None
            
        rv_daily = []
        dates = self.daily_data['date'].unique()
        
        for date in dates:
            day_data = self.intraday_data[self.intraday_data['date'] == date]
            if len(day_data) > 1:
                # Calculate minute log returns
                returns = np.log(day_data['close'] / day_data['close'].shift(1))
                rv = np.sum(returns**2)
                rv_daily.append({'date': date, 'realized_variance': rv})
            else:
                rv_daily.append({'date': date, 'realized_variance': np.nan})
        
        rv_df = pd.DataFrame(rv_daily)
        self.features_df = pd.merge(self.features_df, rv_df, on='date', how='left')
        return rv_df
    
    def calculate_parkinson_estimator(self):
        """Parkinson volatility estimator using high/low"""
        if 'high' in self.daily_data.columns and 'low' in self.daily_data.columns:
            mask = (self.daily_data['high'] > 0) & (self.daily_data['low'] > 0)
            self.features_df.loc[mask, 'parkinson_estimator'] = (
                (1 / (4 * np.log(2))) * 
                (np.log(self.daily_data.loc[mask, 'high'] / self.daily_data.loc[mask, 'low']))**2
            )
        return self.features_df
    
    def calculate_garman_klass(self):
        """Garman-Klass volatility estimator"""
        required = ['high', 'low', 'close', 'open']
        if all(col in self.daily_data.columns for col in required):
            mask = (self.daily_data[required] > 0).all(axis=1)
            
            hl_term = 0.5 * (np.log(self.daily_data.loc[mask, 'high'] / 
                                   self.daily_data.loc[mask, 'low']))**2
            co_term = (2 * np.log(2) - 1) * (np.log(self.daily_data.loc[mask, 'close'] / 
                                                   self.daily_data.loc[mask, 'open']))**2
            
            self.features_df.loc[mask, 'garman_klass'] = hl_term - co_term
        return self.features_df
    
    def calculate_rogers_satchell(self):
        """Rogers-Satchell volatility estimator (drift-free)"""
        required = ['high', 'low', 'close', 'open']
        if all(col in self.daily_data.columns for col in required):
            mask = (self.daily_data[required] > 0).all(axis=1)
            
            hc_ho = (np.log(self.daily_data.loc[mask, 'high'] / self.daily_data.loc[mask, 'close']) * 
                    np.log(self.daily_data.loc[mask, 'high'] / self.daily_data.loc[mask, 'open']))
            lc_lo = (np.log(self.daily_data.loc[mask, 'low'] / self.daily_data.loc[mask, 'close']) * 
                    np.log(self.daily_data.loc[mask, 'low'] / self.daily_data.loc[mask, 'open']))
            
            self.features_df.loc[mask, 'rogers_satchell'] = hc_ho + lc_lo
        return self.features_df
    
    def calculate_bipower_variation(self):
        """Calculate bipower variation from intraday data"""
        if self.intraday_data is None:
            return None
            
        bv_daily = []
        dates = self.daily_data['date'].unique()
        
        for date in dates:
            day_data = self.intraday_data[self.intraday_data['date'] == date]
            if len(day_data) > 2:
                returns = np.log(day_data['close'] / day_data['close'].shift(1))
                abs_returns = np.abs(returns.dropna())
                if len(abs_returns) > 1:
                    bv = (np.pi / 2) * np.sum(abs_returns[:-1] * abs_returns[1:])
                    bv_daily.append({'date': date, 'bipower_variation': bv})
                else:
                    bv_daily.append({'date': date, 'bipower_variation': np.nan})
            else:
                bv_daily.append({'date': date, 'bipower_variation': np.nan})
        
        bv_df = pd.DataFrame(bv_daily)
        self.features_df = pd.merge(self.features_df, bv_df, on='date', how='left')
        return bv_df
    
    def calculate_jump_variation(self):
        """Calculate jump variation as max(RV - BV, 0)"""
        if 'realized_variance' in self.features_df.columns and 'bipower_variation' in self.features_df.columns:
            self.features_df['jump_variation'] = np.maximum(
                self.features_df['realized_variance'] - self.features_df['bipower_variation'], 
                0
            )
        return self.features_df
    
    def calculate_realized_moments(self):
        """Calculate realized skewness and kurtosis from intraday data"""
        if self.intraday_data is None:
            return None
            
        moments_daily = []
        dates = self.daily_data['date'].unique()
        
        for date in dates:
            day_data = self.intraday_data[self.intraday_data['date'] == date]
            if len(day_data) > 10:  # Need enough data for meaningful moments
                returns = np.log(day_data['close'] / day_data['close'].shift(1)).dropna()
                if len(returns) > 0:
                    skew = stats.skew(returns)
                    kurt = stats.kurtosis(returns)
                    moments_daily.append({
                        'date': date, 
                        'realized_skewness': skew,
                        'realized_kurtosis': kurt
                    })
                else:
                    moments_daily.append({
                        'date': date, 
                        'realized_skewness': np.nan,
                        'realized_kurtosis': np.nan
                    })
            else:
                moments_daily.append({
                    'date': date, 
                    'realized_skewness': np.nan,
                    'realized_kurtosis': np.nan
                })
        
        moments_df = pd.DataFrame(moments_daily)
        self.features_df = pd.merge(self.features_df, moments_df, on='date', how='left')
        return moments_df
    
    def calculate_ewma_rv(self, lambda_param=0.94):
        """Calculate EWMA of realized variance"""
        if 'realized_variance' in self.features_df.columns:
            rv = self.features_df['realized_variance'].fillna(method='ffill')
            ewma = pd.Series(index=rv.index, dtype=float)
            
            # Initialize with first non-null value
            first_valid = rv.first_valid_index()
            if first_valid is not None:
                ewma.iloc[first_valid] = rv.iloc[first_valid]
                
                # Calculate EWMA
                for i in range(first_valid + 1, len(rv)):
                    ewma.iloc[i] = lambda_param * ewma.iloc[i-1] + (1 - lambda_param) * rv.iloc[i]
            
            self.features_df['ewma_rv'] = ewma
            
            # Volatility ratio
            self.features_df['volatility_ratio'] = (
                self.features_df['realized_variance'] / self.features_df['ewma_rv']
            )
        return self.features_df
    
    def calculate_gap(self):
        """Calculate overnight gap return"""
        if 'open' in self.daily_data.columns and 'close' in self.daily_data.columns:
            prev_close = self.daily_data['close'].shift(1)
            mask = (self.daily_data['open'] > 0) & (prev_close > 0)
            self.features_df.loc[mask, 'gap'] = np.log(
                self.daily_data.loc[mask, 'open'] / prev_close[mask]
            )
        return self.features_df
    
    def calculate_intraday_range_ratio(self):
        """Calculate normalized daily range"""
        if all(col in self.daily_data.columns for col in ['high', 'low', 'close']):
            mask = self.daily_data['close'] > 0
            self.features_df.loc[mask, 'intraday_range_ratio'] = (
                (self.daily_data.loc[mask, 'high'] - self.daily_data.loc[mask, 'low']) / 
                self.daily_data.loc[mask, 'close']
            )
        return self.features_df
    
    def calculate_last_hour_return(self):
        """Calculate last hour return from intraday data"""
        if self.intraday_data is None:
            return None
            
        last_hour_returns = []
        dates = self.daily_data['date'].unique()
        
        for date in dates:
            day_data = self.intraday_data[self.intraday_data['date'] == date]
            if len(day_data) > 60:  # At least 60 minutes of data
                close_price = day_data.iloc[-1]['close']
                hour_ago_price = day_data.iloc[-60]['close']
                if close_price > 0 and hour_ago_price > 0:
                    ret = np.log(close_price / hour_ago_price)
                    last_hour_returns.append({'date': date, 'last_hour_return': ret})
                else:
                    last_hour_returns.append({'date': date, 'last_hour_return': np.nan})
            else:
                last_hour_returns.append({'date': date, 'last_hour_return': np.nan})
        
        lhr_df = pd.DataFrame(last_hour_returns)
        self.features_df = pd.merge(self.features_df, lhr_df, on='date', how='left')
        return lhr_df
    
    def calculate_dollar_volume(self):
        """Calculate log dollar volume"""
        if self.intraday_data is not None:
            dvol_daily = []
            dates = self.daily_data['date'].unique()
            
            for date in dates:
                day_data = self.intraday_data[self.intraday_data['date'] == date]
                if len(day_data) > 0 and 'volume' in day_data.columns:
                    dollar_vol = np.sum(day_data['close'] * day_data['volume'])
                    dvol_daily.append({'date': date, 'dollar_volume': np.log(1 + dollar_vol)})
                else:
                    dvol_daily.append({'date': date, 'dollar_volume': np.nan})
            
            dvol_df = pd.DataFrame(dvol_daily)
            self.features_df = pd.merge(self.features_df, dvol_df, on='date', how='left')
        elif 'volume' in self.daily_data.columns and 'close' in self.daily_data.columns:
            # Fallback to daily data
            self.features_df['dollar_volume'] = np.log(
                1 + self.daily_data['close'] * self.daily_data['volume']
            )
        return self.features_df
    
    def calculate_volume_volatility(self):
        """Calculate volume volatility"""
        if self.intraday_data is not None:
            vvol_daily = []
            dates = self.daily_data['date'].unique()
            
            for date in dates:
                day_data = self.intraday_data[self.intraday_data['date'] == date]
                if len(day_data) > 1 and 'volume' in day_data.columns:
                    log_volumes = np.log(1 + day_data['volume'])
                    vol_std = log_volumes.std()
                    vvol_daily.append({'date': date, 'volume_volatility': vol_std})
                else:
                    vvol_daily.append({'date': date, 'volume_volatility': np.nan})
            
            vvol_df = pd.DataFrame(vvol_daily)
            self.features_df = pd.merge(self.features_df, vvol_df, on='date', how='left')
        return self.features_df
    
    def engineer_all_features(self):
        """Calculate all features"""
        print("Engineering features...")
        
        # Volatility features
        print("Calculating volatility features...")
        self.calculate_realized_variance()
        self.calculate_parkinson_estimator()
        self.calculate_garman_klass()
        self.calculate_rogers_satchell()
        self.calculate_bipower_variation()
        self.calculate_jump_variation()
        self.calculate_realized_moments()
        self.calculate_ewma_rv()
        
        # Trend features
        print("Calculating trend features...")
        self.calculate_gap()
        self.calculate_intraday_range_ratio()
        self.calculate_last_hour_return()
        
        # Liquidity features
        print("Calculating liquidity features...")
        self.calculate_dollar_volume()
        self.calculate_volume_volatility()
        
        print(f"Total features created: {self.features_df.shape[1]}")
        
        # Handle missing values
        print("Handling missing values...")
        for col in self.features_df.columns:
            if self.features_df[col].dtype in ['float64', 'int64']:
                # Forward fill then backward fill for numeric columns
                self.features_df[col] = self.features_df[col].fillna(method='ffill').fillna(method='bfill')
        
        return self.features_df
    
    def get_feature_importance_stats(self):
        """Get basic statistics about engineered features"""
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        stats = self.features_df[numeric_cols].describe()
        
        # Add missing value counts
        missing = pd.DataFrame({
            'missing_count': self.features_df[numeric_cols].isnull().sum(),
            'missing_pct': (self.features_df[numeric_cols].isnull().sum() / len(self.features_df) * 100).round(2)
        })
        
        return stats, missing

# Test the feature engineering
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    master = loader.load_master_dataset()
    intraday = loader.load_intraday_data()
    
    # Engineer features
    engineer = FeatureEngineer(master, intraday)
    features = engineer.engineer_all_features()
    
    print("\nFeature engineering complete!")
    print(f"Final feature matrix shape: {features.shape}")
    
    # Get statistics
    stats, missing = engineer.get_feature_importance_stats()
    print("\nFeature statistics:")
    print(stats.iloc[:, :5])  # Show first 5 features
    print("\nMissing values:")
    print(missing[missing['missing_count'] > 0])