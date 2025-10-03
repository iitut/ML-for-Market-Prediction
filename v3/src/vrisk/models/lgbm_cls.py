"""
LightGBM classifier for crash/boom/normal prediction.
Includes class balancing and sample weighting.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class LightGBMClassifier:
    """LightGBM for 3-class classification with imbalance handling."""
    
    def __init__(self,
                 params: Optional[Dict[str, Any]] = None,
                 class_weight: str = 'balanced',
                 sample_weight_power: float = 1.0,
                 use_gpu: bool = False):
        """
        Initialize LightGBM classifier.
        
        Args:
            params: LightGBM parameters
            class_weight: How to handle class imbalance
            sample_weight_power: Power for sample weighting (1 + |z|)^power
            use_gpu: Whether to use GPU
        """
        self.class_weight = class_weight
        self.sample_weight_power = sample_weight_power
        self.use_gpu = use_gpu
        
        # Default parameters
        default_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': ['multi_logloss', 'multi_error'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'max_depth': -1,
            'min_gain_to_split': 0.0,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42
        }
        
        if use_gpu:
            default_params['device'] = 'gpu'
            default_params['gpu_platform_id'] = 0
            default_params['gpu_device_id'] = 0
            
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
        Train the LightGBM classifier.
        
        Args:
            X: Training features
            y: Training labels (can be strings or ints)
            sample_weight: Sample weights
            eval_set: Validation set (X_val, y_val)
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity level
        """
        logger.info("Training LightGBM classifier")
        
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
            # Combine with provided weights
            sample_weight = sample_weight * self._compute_sample_weights(X, y_encoded)
            
        # Apply class weights to sample weights
        for i in range(3):
            mask = y_encoded == i
            sample_weight[mask] *= self.class_weights[i]
            
        # Create dataset
        train_data = lgb.Dataset(
            X, label=y_encoded,
            weight=sample_weight,
            params={'verbose': -1}
        )
        
        # Create validation set if provided
        valid_sets = [train_data]
        if eval_set is not None:
            X_val, y_val = eval_set
            if hasattr(X_val, 'to_numpy'):
                X_val = X_val.to_numpy()
            if hasattr(y_val, 'to_numpy'):
                y_val = y_val.to_numpy()
                
            if y_val.dtype == object:
                y_val = self.label_encoder.transform(y_val)
                
            valid_data = lgb.Dataset(
                X_val, label=y_val,
                reference=train_data,
                params={'verbose': -1}
            )
            valid_sets.append(valid_data)
            
        # Train model
        callbacks = [
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(verbose)
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=valid_sets,
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        # Store feature names
        self.feature_names = [f'f_{i}' for i in range(X.shape[1])]
        
        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities (n_samples, 3)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
            
        proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Ensure proper shape
        if proba.ndim == 1:
            proba = proba.reshape(-1, 3)
            
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            
        Returns:
            Class labels
        """
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(y_pred)
    
    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute balanced class weights."""
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        
        weights = n_samples / (n_classes * counts)
        
        # Normalize
        weights = weights / weights.mean()
        
        logger.info(f"Class weights: {dict(zip(classes, weights))}")
        return weights
    
    def _compute_sample_weights(self, 
                               X: np.ndarray, 
                               y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights based on extremeness.
        Weight = (1 + |z_score|)^power
        """
        # Assuming z_score is in the features (would need to be passed)
        # For now, use uniform weights
        weights = np.ones(len(y))
        
        # Could enhance with actual z-scores if available
        # weights = (1 + np.abs(z_scores)) ** self.sample_weight_power
        
        return weights
    
    def get_feature_importance(self, 
                              importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: 'gain' or 'split'
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        importance = self.model.feature_importance(importance_type=importance_type)
        
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
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Model loaded from {path}")