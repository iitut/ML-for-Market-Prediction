"""
XGBoost classifier for crash/boom/normal prediction.
Includes class balancing and GPU support.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """XGBoost for 3-class classification with imbalance handling."""
    
    def __init__(self,
                 params: Optional[Dict[str, Any]] = None,
                 class_weight: str = 'balanced',
                 sample_weight_power: float = 1.0,
                 use_gpu: bool = False):
        """
        Initialize XGBoost classifier.
        
        Args:
            params: XGBoost parameters
            class_weight: How to handle class imbalance
            sample_weight_power: Power for sample weighting
            use_gpu: Whether to use GPU
        """
        self.class_weight = class_weight
        self.sample_weight_power = sample_weight_power
        self.use_gpu = use_gpu
        
        # Default parameters
        default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': ['mlogloss', 'merror'],
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 1,
            'verbosity': 0,
            'n_jobs': -1,
            'random_state': 42
        }
        
        if use_gpu:
            default_params['tree_method'] = 'gpu_hist'
            default_params['predictor'] = 'gpu_predictor'
            
        self.params = {**default_params, **(params or {})}
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_weights = None
        
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            early_stopping_rounds: int = 50,
            verbose: int = 100):
        """
        Train the XGBoost classifier.
        
        Args:
            X: Training features
            y: Training labels
            sample_weight: Sample weights
            eval_set: Validation set (X_val, y_val)
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity level
        """
        logger.info("Training XGBoost classifier")
        
        # Convert to numpy if needed
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
            
        # Encode labels if strings
        if y.dtype == object:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
            self.label_encoder.classes_ = np.array(['crash', 'normal', 'boom'])
            
        # Calculate class weights
        if self.class_weight == 'balanced':
            self.class_weights = self._compute_class_weights(y_encoded)
        else:
            self.class_weights = np.ones(3)
            
        # Calculate sample weights
        if sample_weight is None:
            sample_weight = self._compute_sample_weights(X, y_encoded)
        else:
            sample_weight = sample_weight * self._compute_sample_weights(X, y_encoded)
            
        # Apply class weights
        for i in range(3):
            mask = y_encoded == i
            sample_weight[mask] *= self.class_weights[i]
            
        # Prepare eval set
        eval_list = []
        if eval_set is not None:
            X_val, y_val = eval_set
            if hasattr(X_val, 'to_numpy'):
                X_val = X_val.to_numpy()
            if hasattr(y_val, 'to_numpy'):
                y_val = y_val.to_numpy()
                
            if y_val.dtype == object:
                y_val = self.label_encoder.transform(y_val)
                
            eval_list = [(X, y_encoded, 'train'), (X_val, y_val, 'valid')]
        else:
            eval_list = [(X, y_encoded, 'train')]
            
        # Train model
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X, y_encoded,
            sample_weight=sample_weight,
            eval_set=eval_list,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose > 0
        )
        
        # Store feature names
        self.feature_names = [f'f_{i}' for i in range(X.shape[1])]
        
        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
            
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
            
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute balanced class weights."""
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        
        weights = n_samples / (n_classes * counts)
        weights = weights / weights.mean()
        
        logger.info(f"Class weights: {dict(zip(classes, weights))}")
        return weights
    
    def _compute_sample_weights(self, 
                               X: np.ndarray, 
                               y: np.ndarray) -> np.ndarray:
        """Compute sample weights based on extremeness."""
        weights = np.ones(len(y))
        return weights
    
    def get_feature_importance(self, 
                              importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False)
    
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model from file."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")