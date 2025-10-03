"""
Enhanced feature engineering with additional technical indicators
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v2\\enhanced_features.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    def __init__(self, daily_data, intraday_data=None):
        """Initialize with enhanced feature engineering"""
        self.daily_data = daily_data.copy()
        self.intraday_data = intraday_data
        self.features_df = daily_data.copy()
        
    # ===== ENHANCED VOLATILITY MEASURES =====
    
    def calculate_yang_zhang_volatility(self, window=20):
        """Yang-Zhang volatility estimator (most efficient)"""
        if all(col in self.daily_data.columns for col in ['open', 'high', 'low', 'close']):
            k = 0.34 / (1 + (window + 1) / (window - 1))
            
            # Overnight volatility
            co = np.log(self.daily_data['open'] / self.daily_data['close'].shift(1))
            co_var = co.rolling(window).var()
            
            # Open-to-close volatility
            oc = np.log(self.daily_data['close'] / self.daily_data['open'])
            oc_var = oc.rolling(window).var()
            
            # Rogers-Satchell volatility
            rs = (np.log(self.daily_data['high'] / self.daily_data['close']) * 
                  np.log(self.daily_data['high'] / self.daily_data['open']) +
                  np.log(self.daily_data['low'] / self.daily_data['close']) * 
                  np.log(self.daily_data['low'] / self.daily_data['open']))
            rs_var = rs.rolling(window).var()
            
            self.features_df['yang_zhang_vol'] = np.sqrt(co_var + k * oc_var + (1 - k) * rs_var)
        return self.features_df
    
    def calculate_close_to_close_volatility(self, window=20):
        """Simple close-to-close volatility"""
        if 'close' in self.daily_data.columns:
            returns = np.log(self.daily_data['close'] / self.daily_data['close'].shift(1))
            self.features_df['close_to_close_vol'] = returns.rolling(window).std() * np.sqrt(252)
        return self.features_df
    
    # ===== TECHNICAL INDICATORS =====
    
    def calculate_rsi(self, period=14):
        """Relative Strength Index"""
        if 'close' in self.daily_data.columns:
            delta = self.daily_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            self.features_df['rsi'] = 100 - (100 / (1 + rs))
        return self.features_df
    
    def calculate_macd(self):
        """MACD indicator"""
        if 'close' in self.daily_data.columns:
            exp1 = self.daily_data['close'].ewm(span=12, adjust=False).mean()
            exp2 = self.daily_data['close'].ewm(span=26, adjust=False).mean()
            self.features_df['macd'] = exp1 - exp2
            self.features_df['macd_signal'] = self.features_df['macd'].ewm(span=9, adjust=False).mean()
            self.features_df['macd_histogram'] = self.features_df['macd'] - self.features_df['macd_signal']
        return self.features_df
    
    def calculate_bollinger_bands(self, window=20, num_std=2):
        """Bollinger Bands and position within bands"""
        if 'close' in self.daily_data.columns:
            sma = self.daily_data['close'].rolling(window).mean()
            std = self.daily_data['close'].rolling(window).std()
            
            self.features_df['bb_upper'] = sma + (std * num_std)
            self.features_df['bb_lower'] = sma - (std * num_std)
            self.features_df['bb_middle'] = sma
            
            # Position within bands (0 = lower band, 1 = upper band)
            self.features_df['bollinger_position'] = (
                (self.daily_data['close'] - self.features_df['bb_lower']) / 
                (self.features_df['bb_upper'] - self.features_df['bb_lower'])
            )
        return self.features_df
    
    def calculate_momentum_indicators(self):
        """Various momentum indicators"""
        if 'close' in self.daily_data.columns:
            # Momentum
            self.features_df['momentum_10'] = self.daily_data['close'] - self.daily_data['close'].shift(10)
            self.features_df['momentum_20'] = self.daily_data['close'] - self.daily_data['close'].shift(20)
            
            # Rate of Change
            self.features_df['roc_10'] = (
                (self.daily_data['close'] - self.daily_data['close'].shift(10)) / 
                self.daily_data['close'].shift(10) * 100
            )
            
            # Williams %R
            if 'high' in self.daily_data.columns and 'low' in self.daily_data.columns:
                high_14 = self.daily_data['high'].rolling(14).max()
                low_14 = self.daily_data['low'].rolling(14).min()
                self.features_df['williams_r'] = (
                    -100 * (high_14 - self.daily_data['close']) / (high_14 - low_14)
                )
        return self.features_df
    
    def calculate_stochastic_oscillator(self, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        if all(col in self.daily_data.columns for col in ['high', 'low', 'close']):
            low_min = self.daily_data['low'].rolling(window=k_period).min()
            high_max = self.daily_data['high'].rolling(window=k_period).max()
            
            self.features_df['stochastic_k'] = (
                100 * (self.daily_data['close'] - low_min) / (high_max - low_min)
            )
            self.features_df['stochastic_d'] = self.features_df['stochastic_k'].rolling(window=d_period).mean()
        return self.features_df
    
    # ===== MOVING AVERAGES =====
    
    def calculate_moving_averages(self):
        """Various moving averages"""
        if 'close' in self.daily_data.columns:
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                self.features_df[f'sma_{period}'] = self.daily_data['close'].rolling(period).mean()
                # Price relative to SMA
                self.features_df[f'price_to_sma_{period}'] = (
                    self.daily_data['close'] / self.features_df[f'sma_{period}']
                )
            
            # Exponential Moving Averages
            for period in [12, 26, 50]:
                self.features_df[f'ema_{period}'] = self.daily_data['close'].ewm(span=period, adjust=False).mean()
        return self.features_df
    
    # ===== VOLUME INDICATORS =====
    
    def calculate_volume_indicators(self):
        """Enhanced volume-based indicators"""
        if 'volume' in self.daily_data.columns and 'close' in self.daily_data.columns:
            # On-Balance Volume
            obv = (np.sign(self.daily_data['close'].diff()) * self.daily_data['volume']).fillna(0).cumsum()
            self.features_df['obv'] = obv
            self.features_df['obv_ratio'] = obv / obv.rolling(20).mean()
            
            # Money Flow Index
            typical_price = (self.daily_data['high'] + self.daily_data['low'] + self.daily_data['close']) / 3
            money_flow = typical_price * self.daily_data['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            mfi_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum()
            self.features_df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # Volume Rate of Change
            self.features_df['volume_roc'] = (
                (self.daily_data['volume'] - self.daily_data['volume'].shift(10)) / 
                self.daily_data['volume'].shift(10) * 100
            )
            
            # Accumulation/Distribution Line
            clv = ((self.daily_data['close'] - self.daily_data['low']) - 
                   (self.daily_data['high'] - self.daily_data['close'])) / \
                  (self.daily_data['high'] - self.daily_data['low'])
            self.features_df['ad_line'] = (clv * self.daily_data['volume']).cumsum()
            
        return self.features_df
    
    # ===== VOLATILITY INDICATORS =====
    
    def calculate_atr(self, period=14):
        """Average True Range"""
        if all(col in self.daily_data.columns for col in ['high', 'low', 'close']):
            high_low = self.daily_data['high'] - self.daily_data['low']
            high_close = np.abs(self.daily_data['high'] - self.daily_data['close'].shift())
            low_close = np.abs(self.daily_data['low'] - self.daily_data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.features_df['atr'] = true_range.rolling(period).mean()
            
            # ATR as percentage of close
            self.features_df['atr_percent'] = (self.features_df['atr'] / self.daily_data['close']) * 100
        return self.features_df
    
    def calculate_adx(self, period=14):
        """Average Directional Index"""
        if all(col in self.daily_data.columns for col in ['high', 'low', 'close']):
            plus_dm = self.daily_data['high'].diff()
            minus_dm = -self.daily_data['low'].diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = self.calculate_true_range()
            
            plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
            minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            self.features_df['adx'] = dx.rolling(period).mean()
            self.features_df['plus_di'] = plus_di
            self.features_df['minus_di'] = minus_di
        return self.features_df
    
    def calculate_true_range(self):
        """Helper function for True Range"""
        high_low = self.daily_data['high'] - self.daily_data['low']
        high_close = np.abs(self.daily_data['high'] - self.daily_data['close'].shift())
        low_close = np.abs(self.daily_data['low'] - self.daily_data['close'].shift())
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ===== MARKET MICROSTRUCTURE =====
    
    def calculate_amihud_illiquidity(self):
        """Amihud illiquidity measure"""
        if 'volume' in self.daily_data.columns and 'close' in self.daily_data.columns:
            returns = self.daily_data['close'].pct_change()
            dollar_volume = self.daily_data['volume'] * self.daily_data['close']
            
            # Avoid division by zero
            dollar_volume = dollar_volume.replace(0, np.nan)
            
            self.features_df['amihud_illiquidity'] = (
                np.abs(returns) / dollar_volume * 1e6
            ).rolling(20).mean()
        return self.features_df
    
    def calculate_kyle_lambda(self):
        """Kyle's lambda - price impact measure"""
        if self.intraday_data is not None:
            kyle_daily = []
            dates = self.daily_data['date'].unique()
            
            for date in dates:
                day_data = self.intraday_data[self.intraday_data['date'] == date]
                if len(day_data) > 10:
                    # 5-minute returns and volumes
                    returns = day_data['close'].pct_change()
                    signed_volume = day_data['volume'] * np.sign(returns)
                    
                    # Regression of price changes on signed volume
                    valid_idx = ~(returns.isna() | signed_volume.isna())
                    if valid_idx.sum() > 10:
                        X = signed_volume[valid_idx].values.reshape(-1, 1)
                        y = returns[valid_idx].values
                        
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression().fit(X, y)
                        kyle_lambda = reg.coef_[0]
                        kyle_daily.append({'date': date, 'kyle_lambda': kyle_lambda})
                    else:
                        kyle_daily.append({'date': date, 'kyle_lambda': np.nan})
                else:
                    kyle_daily.append({'date': date, 'kyle_lambda': np.nan})
            
            kyle_df = pd.DataFrame(kyle_daily)
            self.features_df = pd.merge(self.features_df, kyle_df, on='date', how='left')
        return self.features_df
    
    # ===== PATTERN RECOGNITION =====
    
    def calculate_support_resistance(self, window=20):
        """Calculate support and resistance levels"""
        if 'high' in self.daily_data.columns and 'low' in self.daily_data.columns:
            self.features_df['resistance'] = self.daily_data['high'].rolling(window).max()
            self.features_df['support'] = self.daily_data['low'].rolling(window).min()
            
            # Distance from support/resistance
            self.features_df['distance_from_resistance'] = (
                (self.features_df['resistance'] - self.daily_data['close']) / 
                self.daily_data['close']
            )
            self.features_df['distance_from_support'] = (
                (self.daily_data['close'] - self.features_df['support']) / 
                self.daily_data['close']
            )
        return self.features_df
    
    # ===== REGIME INDICATORS =====
    
    def calculate_regime_indicators(self):
        """Market regime detection features"""
        if 'close' in self.daily_data.columns:
            # Trend strength
            returns = self.daily_data['close'].pct_change()
            
            # Rolling correlation with time (trend strength)
            window = 20
            time_index = np.arange(len(self.daily_data))
            
            rolling_corr = []
            for i in range(window, len(self.daily_data)):
                window_data = self.daily_data['close'].iloc[i-window:i]
                window_time = time_index[i-window:i]
                if len(window_data) == window:
                    corr = np.corrcoef(window_time, window_data)[0, 1]
                    rolling_corr.append(corr)
                else:
                    rolling_corr.append(np.nan)
            
            # Pad the beginning with NaN
            rolling_corr = [np.nan] * window + rolling_corr
            self.features_df['trend_strength'] = rolling_corr
            
            # Volatility regime (high/low vol)
            vol = returns.rolling(20).std()
            vol_percentile = vol.rolling(252).rank(pct=True)
            self.features_df['volatility_regime'] = vol_percentile
            
        return self.features_df
    
    def engineer_all_enhanced_features(self):
        """Calculate all enhanced features"""
        print("Engineering enhanced features...")
        
        # Original features (from v1)
        print("Calculating base volatility features...")
        self.calculate_yang_zhang_volatility()
        self.calculate_close_to_close_volatility()
        
        # Technical indicators
        print("Calculating technical indicators...")
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_momentum_indicators()
        self.calculate_stochastic_oscillator()
        
        # Moving averages
        print("Calculating moving averages...")
        self.calculate_moving_averages()
        
        # Volume indicators
        print("Calculating volume indicators...")
        self.calculate_volume_indicators()
        
        # Volatility indicators
        print("Calculating volatility indicators...")
        self.calculate_atr()
        self.calculate_adx()
        
        # Market microstructure
        print("Calculating market microstructure...")
        self.calculate_amihud_illiquidity()
        self.calculate_kyle_lambda()
        
        # Pattern recognition
        print("Calculating support/resistance...")
        self.calculate_support_resistance()
        
        # Regime indicators
        print("Calculating regime indicators...")
        self.calculate_regime_indicators()
        
        print(f"Total enhanced features created: {self.features_df.shape[1]}")
        
        # Handle missing values
        print("Handling missing values...")
        for col in self.features_df.columns:
            if self.features_df[col].dtype in ['float64', 'int64']:
                self.features_df[col] = self.features_df[col].fillna(method='ffill').fillna(method='bfill')
        
        return self.features_df