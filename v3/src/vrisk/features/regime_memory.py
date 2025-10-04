"""
Regime and memory features including EWMA, volatility ratios, and regime indicators.
"""

import polars as pl
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class RegimeMemoryFeatures:
    """Generate regime-aware and memory-based features."""
    
    PREFIX = 'reg_'
    
    def __init__(self,
                 vr_quantiles: int = 5,
                 ewma_spans: List[int] = [5, 10, 20, 60],
                 memory_lags: List[int] = [1, 5, 10, 20]):
        """
        Initialize regime and memory feature generator.
        
        Args:
            vr_quantiles: Number of VR quantiles for regime definition
            ewma_spans: EWMA spans for different horizons
            memory_lags: Lags for autoregressive features
        """
        self.vr_quantiles = vr_quantiles
        self.ewma_spans = ewma_spans
        self.memory_lags = memory_lags
        
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate regime and memory features.
        
        Args:
            df: DataFrame with volatility measures
            
        Returns:
            DataFrame with regime and memory features
        """
        logger.info("Generating regime and memory features")
        
        # Add EWMA features
        df = self._add_ewma_features(df)
        
        # Add volatility ratio (VR) features
        df = self._add_vr_features(df)
        
        # Add regime indicators
        df = self._add_regime_indicators(df)
        
        # Add memory/lag features
        df = self._add_memory_features(df)
        
        # Add regime transitions
        df = self._add_regime_transitions(df)
        
        # Add long-term memory
        df = self._add_long_memory(df)
        
        return df
    
    def _add_ewma_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add EWMA volatility and related features."""
        
        for span in self.ewma_spans:
            alpha = 2 / (span + 1)
            
            df = df.with_columns([
                # EWMA of RV
                pl.col('RV_daily')
                .ewm_mean(alpha=alpha)
                .alias(f'{self.PREFIX}ewma_rv_{span}d'),
                
                # EWMA of log RV (for stability)
                pl.col('log_RV_daily')
                .ewm_mean(alpha=alpha)
                .alias(f'{self.PREFIX}ewma_log_rv_{span}d'),
                
                # EWMA volatility (square root)
                pl.col('RV_daily')
                .ewm_mean(alpha=alpha)
                .sqrt()
                .alias(f'{self.PREFIX}ewma_vol_{span}d'),
                
                # Volatility of volatility (using EWMA)
                (pl.col('RV_daily') - pl.col('RV_daily').ewm_mean(alpha=alpha)).pow(2)
                .ewm_mean(alpha=alpha)
                .sqrt()
                .alias(f'{self.PREFIX}vol_of_vol_{span}d')
            ])
            
        # Multi-scale EWMA ratios
        if len(self.ewma_spans) >= 2:
            for i, short_span in enumerate(self.ewma_spans[:-1]):
                for long_span in self.ewma_spans[i+1:]:
                    df = df.with_columns([
                        (pl.col(f'{self.PREFIX}ewma_rv_{short_span}d') / 
                         pl.col(f'{self.PREFIX}ewma_rv_{long_span}d'))
                        .alias(f'{self.PREFIX}ewma_ratio_{short_span}_{long_span}d')
                    ])
                    
        return df
    
    def _add_vr_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility ratio (VR) features."""
        
        # Calculate VR at different horizons
        for span in self.ewma_spans:
            df = df.with_columns([
                # Volatility Ratio: RV_t / EWMA(RV)_{t-1}
                (pl.col('RV_daily') / 
                 pl.col(f'{self.PREFIX}ewma_rv_{span}d').shift(1))
                .alias(f'{self.PREFIX}vr_{span}d'),
                
                # Log VR for stability
                (pl.col('RV_daily') / 
                 pl.col(f'{self.PREFIX}ewma_rv_{span}d').shift(1)).log()
                .alias(f'{self.PREFIX}log_vr_{span}d'),
                
                # VR deviation from 1
                ((pl.col('RV_daily') / 
                  pl.col(f'{self.PREFIX}ewma_rv_{span}d').shift(1)) - 1).abs()
                .alias(f'{self.PREFIX}vr_deviation_{span}d')
            ])
            
        # VR quantiles for regime definition
        df = df.with_columns([
            # Calculate VR quintiles
            pl.col(f'{self.PREFIX}vr_20d')
            .qcut(self.vr_quantiles, labels=[str(i) for i in range(self.vr_quantiles)])
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}vr_quintile'),
            
            # VR percentile rank
            # VR percentile (fixed for Polars compatibility)
            (pl.col(f'{self.PREFIX}vr_20d').rank().cast(pl.Float64) / pl.len())
            .alias(f'{self.PREFIX}vr_percentile')
        ])
        
        return df
    
    def _add_regime_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility regime indicators."""
        
        # Define regimes based on RV levels
        df = df.with_columns([
            # Absolute volatility regime (based on RV percentiles)
            pl.when(pl.col('RV_daily') > 
                    pl.col('RV_daily').rolling_quantile(0.9, window_size=252, min_periods=126))
            .then(3)  # Extreme high vol
            .when(pl.col('RV_daily') > 
                  pl.col('RV_daily').rolling_quantile(0.75, window_size=252, min_periods=126))
            .then(2)  # High vol
            .when(pl.col('RV_daily') > 
                  pl.col('RV_daily').rolling_quantile(0.25, window_size=252, min_periods=126))
            .then(1)  # Normal vol
            .otherwise(0)  # Low vol
            .alias(f'{self.PREFIX}vol_regime'),
            
            # Trend regime (based on EWMA differences)
            pl.when(pl.col(f'{self.PREFIX}ewma_rv_5d') > pl.col(f'{self.PREFIX}ewma_rv_20d'))
            .then(1)  # Increasing vol
            .otherwise(0)  # Decreasing vol
            .alias(f'{self.PREFIX}vol_trend'),
            
            # Jump regime
            (pl.col('jump_ratio') > 0.1).cast(pl.Int32)
            .alias(f'{self.PREFIX}jump_regime'),
            
            # Mean reversion regime (VR far from 1)
            (pl.col(f'{self.PREFIX}vr_deviation_20d') > 0.5).cast(pl.Int32)
            .alias(f'{self.PREFIX}mean_reversion_regime'),
            
            # Clustering regime (consecutive high vol days)
            (pl.col('RV_daily') > pl.col(f'{self.PREFIX}ewma_rv_20d'))
            .cast(pl.Int32)
            .rolling_sum(window_size=5, min_periods=1)
            .alias(f'{self.PREFIX}vol_clustering')
        ])
        
        # Add interaction with options expiration
        if 'is_opx' in df.columns:
            df = df.with_columns([
                # OPX in high vol regime
                (pl.col('is_opx') & (pl.col(f'{self.PREFIX}vol_regime') >= 2))
                .cast(pl.Int32)
                .alias(f'{self.PREFIX}opx_high_vol'),
                
                # OPX with mean reversion setup
                (pl.col('is_opx') & pl.col(f'{self.PREFIX}mean_reversion_regime'))
                .cast(pl.Int32)
                .alias(f'{self.PREFIX}opx_mean_revert')
            ])
            
        return df
    
    def _add_memory_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add autoregressive and memory features."""
        
        for lag in self.memory_lags:
            df = df.with_columns([
                # Lagged RV
                pl.col('RV_daily').shift(lag)
                .alias(f'{self.PREFIX}rv_lag_{lag}d'),
                
                # Lagged log RV
                pl.col('log_RV_daily').shift(lag)
                .alias(f'{self.PREFIX}log_rv_lag_{lag}d'),
                
                # Lagged returns
                pl.col('daily_return').shift(lag)
                .alias(f'{self.PREFIX}return_lag_{lag}d'),
                
                # Lagged z-score
                pl.col('z_score_next').shift(lag)
                .alias(f'{self.PREFIX}z_lag_{lag}d'),
                
                # Change from lag
                (pl.col('RV_daily') / pl.col('RV_daily').shift(lag) - 1)
                .alias(f'{self.PREFIX}rv_change_{lag}d')
            ])
            
        # Add partial autocorrelation proxies
        if len(self.memory_lags) > 1:
            # AR(1) residual effect
            df = df.with_columns([
                (pl.col('log_RV_daily') - 
                 pl.col(f'{self.PREFIX}log_rv_lag_1d') * 0.8)  # Simplified AR coefficient
                .alias(f'{self.PREFIX}ar1_residual')
            ])
            
        # Add memory persistence measures
        df = df.with_columns([
            # Rolling autocorrelation (simplified)
            pl.col('log_RV_daily')
            .rolling_map(
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0,
                window_size=60,
                min_periods=30
            )
            .alias(f'{self.PREFIX}rolling_autocorr'),
            
            # Hurst exponent proxy (R/S statistic simplified)
            pl.col('log_RV_daily')
            .rolling_map(
                lambda x: (np.max(np.cumsum(x - np.mean(x))) - 
                          np.min(np.cumsum(x - np.mean(x)))) / (np.std(x) + 1e-8)
                          if len(x) > 1 else 1,
                window_size=60,
                min_periods=30
            )
            .alias(f'{self.PREFIX}hurst_proxy')
        ])
        
        return df
    
    def _add_regime_transitions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add regime transition features."""
        
        # Calculate regime changes
        df = df.with_columns([
            # Volatility regime change
            (pl.col(f'{self.PREFIX}vol_regime') != 
             pl.col(f'{self.PREFIX}vol_regime').shift(1))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}regime_change'),
            
            # Days in current regime
            pl.col(f'{self.PREFIX}vol_regime')
            .rle()
            .struct.field('lengths')
            .alias(f'{self.PREFIX}regime_duration'),
            
            # Transition probabilities (simplified)
            # Count transitions in rolling window
            (pl.col(f'{self.PREFIX}vol_regime') != 
             pl.col(f'{self.PREFIX}vol_regime').shift(1))
            .cast(pl.Int32)
            .rolling_sum(window_size=60, min_periods=30)
            .alias(f'{self.PREFIX}transition_frequency'),
            
            # VR regime stability (consecutive days in same quintile)
            (pl.col(f'{self.PREFIX}vr_quintile') == 
             pl.col(f'{self.PREFIX}vr_quintile').shift(1))
            .cast(pl.Int32)
            .rolling_sum(window_size=5, min_periods=1)
            .alias(f'{self.PREFIX}vr_stability')
        ])
        
        # Add specific transition types
        for from_regime in range(4):  # 0-3 vol regimes
            for to_regime in range(4):
                if from_regime != to_regime:
                    df = df.with_columns([
                        ((pl.col(f'{self.PREFIX}vol_regime').shift(1) == from_regime) & 
                         (pl.col(f'{self.PREFIX}vol_regime') == to_regime))
                        .cast(pl.Int32)
                        .rolling_sum(window_size=60, min_periods=30)
                        .alias(f'{self.PREFIX}trans_{from_regime}_to_{to_regime}')
                    ])
                    
        return df
    
    def _add_long_memory(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add long-term memory features."""
        
        # Long-term averages
        long_windows = [60, 120, 252]
        
        for window in long_windows:
            df = df.with_columns([
                # Long-term RV average
                pl.col('RV_daily')
                .rolling_mean(window_size=window, min_periods=window//2)
                .alias(f'{self.PREFIX}rv_lta_{window}d'),
                
                # Distance from long-term average
                (pl.col('RV_daily') / 
                 pl.col('RV_daily').rolling_mean(window_size=window, min_periods=window//2) - 1)
                .alias(f'{self.PREFIX}rv_vs_lta_{window}d'),
                
                # Fractional differencing proxy (simplified)
                pl.col('log_RV_daily')
                .diff()
                .rolling_sum(window_size=window, min_periods=window//2)
                .alias(f'{self.PREFIX}frac_diff_{window}d')
            ])
            
        # Maximum/minimum over long horizons
        df = df.with_columns([
            # 52-week high/low
            (pl.col('RV_daily') / 
             pl.col('RV_daily').rolling_max(window_size=252, min_periods=126))
            .alias(f'{self.PREFIX}dist_from_52w_high'),
            
            (pl.col('RV_daily') / 
             pl.col('RV_daily').rolling_min(window_size=252, min_periods=126))
            .alias(f'{self.PREFIX}dist_from_52w_low'),
            
            # New high/low indicators
            (pl.col('RV_daily') == 
             pl.col('RV_daily').rolling_max(window_size=252, min_periods=126))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}new_52w_high'),
            
            (pl.col('RV_daily') == 
             pl.col('RV_daily').rolling_min(window_size=252, min_periods=126))
            .cast(pl.Int32)
            .alias(f'{self.PREFIX}new_52w_low')
        ])
        
        return df