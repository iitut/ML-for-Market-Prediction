"""
Volatility prediction head with HAR-RV baseline and ensemble models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class HARRVModel:
    """HAR-RV (Heterogeneous AutoRegressive Realized Volatility) baseline model."""
    
    def __init__(self):
        """Initialize HAR-RV model."""
        self.coef_ = None
        self.intercept_ = None
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit HAR-RV model.
        
        Args:
            X: Features [RV_daily, RV_weekly, RV_monthly]
            y: Target log(RV) next day
        """
        # Use ElasticNet for robustness
        self.model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
        self.model.fit(X[:, :3], y)  # Use only first 3 features (daily, weekly, monthly RV)
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        logger.info(f"HAR-RV coefficients: {self.coef_}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-day log RV."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X[:, :3])
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'coef': self.coef_.tolist() if self.coef_ is not None else None,
            'intercept': float(self.intercept_) if self.intercept_ is not None else None
        }


class VolatilityEnsemble:
    """Ensemble model for volatility prediction."""
    
    def __init__(self,
                 base_models: Optional[List[str]] = None,
                 meta_model: str = 'elastic_net',
                 use_har_baseline: bool = True):
        """
        Initialize volatility ensemble.
        
        Args:
            base_models: List of base model names
            meta_model: Meta-learner type
            use_har_baseline: Whether to include HAR-RV
        """
        self.base_models = base_models or ['har_rv', 'lightgbm', 'xgboost', 'random_forest']
        self.meta_model_type = meta_model
        self.use_har_baseline = use_har_baseline
        
        self.models_ = {}
        self.meta_model_ = None
        self.feature_names_ = None
        
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Fit volatility ensemble.
        
        Args:
            X: Training features
            y: Target (log RV next day)
            sample_weight: Sample weights
            eval_set: Validation set
        """
        logger.info("Training volatility ensemble")
        
        # Store feature names
        self.feature_names_ = [f'f_{i}' for i in range(X.shape[1])]
        
        # Train base models
        oof_predictions = []
        
        for model_name in self.base_models:
            logger.info(f"Training {model_name}")
            
            if model_name == 'har_rv':
                # HAR-RV baseline
                model = HARRVModel()
                # Create HAR features
                har_features = self._create_har_features(X)
                model.fit(har_features, y)
                oof_pred = model.predict(har_features)
                
            elif model_name == 'lightgbm':
                model = lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y, sample_weight=sample_weight)
                oof_pred = model.predict(X)
                
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y, sample_weight=sample_weight)
                oof_pred = model.predict(X)
                
            elif model_name == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X, y, sample_weight=sample_weight)
                oof_pred = model.predict(X)
                
            elif model_name == 'elastic_net':
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                model.fit(X, y)
                oof_pred = model.predict(X)
                
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
            self.models_[model_name] = model
            oof_predictions.append(oof_pred.reshape(-1, 1))
            
        # Stack predictions
        meta_features = np.hstack(oof_predictions)
        
        # Train meta-model
        logger.info(f"Training meta-model: {self.meta_model_type}")
        
        if self.meta_model_type == 'elastic_net':
            self.meta_model_ = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
        elif self.meta_model_type == 'lightgbm':
            self.meta_model_ = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=15,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta model: {self.meta_model_type}")
            
        self.meta_model_.fit(meta_features, y)
        
        logger.info("Volatility ensemble training completed")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-day log RV."""
        # Get base predictions
        base_predictions = []
        
        for model_name in self.base_models:
            model = self.models_[model_name]
            
            if model_name == 'har_rv':
                har_features = self._create_har_features(X)
                pred = model.predict(har_features)
            else:
                pred = model.predict(X)
                
            base_predictions.append(pred.reshape(-1, 1))
            
        # Stack and predict with meta-model
        meta_features = np.hstack(base_predictions)
        return self.meta_model_.predict(meta_features)
    
    def _create_har_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create HAR-RV features from raw features.
        Assumes certain columns contain RV at different horizons.
        """
        # Simplified: use first few columns as proxy for RV lags
        # In practice, would extract specific RV lag features
        har_features = X[:, :3]  # Daily, weekly, monthly RV
        return har_features