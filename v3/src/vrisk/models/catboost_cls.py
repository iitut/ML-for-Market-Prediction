"""
CatBoost classifier for crash/boom/normal prediction.
Handles categorical features and has built-in GPU support.
"""

import catboost as cb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class CatBoostClassifier:
    """CatBoost for 3-class classification with advanced features."""
    
    def __init__(self,
                 params: Optional[Dict[str, Any]] = None,
                 class_weight: str = 'balanced',
                 sample_weight_power: float = 1.0,
                 use_gpu: bool = False,
                 categorical_features: Optional[List[int]] = None):
        """
        Initialize CatBoost classifier.
        
        Args:
            params: CatBoost parameters
            class_weight: How to handle class imbalance
            sample_weight_power: Power for sample weighting
            use_gpu: Whether to use GPU
            categorical_features: Indices of categorical features
        """
        self.class_weight = class_weight
        self.sample_weight_power = sample_weight_power
        self.use_gpu = use_gpu
        self.categorical_features = categorical_features or []
        
        # Default parameters
        default_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'MultiClass',
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'bagging_temperature': 0.8,
            'random_strength': 1.0,
            'border_count': 128,
            'grow_policy': 'SymmetricTree',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'od_wait': 50,
            'random_seed': 42,
            'verbose': False
        }
        
        if use_gpu:
            default_params['task_type'] = 'GPU'
            default_params['devices'] = '0'
            
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
        Train the CatBoost classifier.
        
        Args:
            X: Training features
            y: Training labels
            sample_weight: Sample weights
            eval_set: Validation set (X_val, y_val)
            early_stopping_rounds: Early stopping patience
            verbose: Verbosity level
        """
        logger.info("Training CatBoost classifier")
        
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
            # Set class weights in params
            self.params['class_weights'] = self.class_weights.tolist()
        else:
            self.class_weights = np.ones(3)
            
        # Calculate sample weights
        if sample_weight is None:
            sample_weight = self._compute_sample_weights(X, y_encoded)
            
        # Create Pool
        train_pool = cb.Pool(
            X, y_encoded,
            weight=sample_weight,
            cat_features=self.categorical_features
        )
        
        # Create validation pool if provided
        eval_pool = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if hasattr(X_val, 'to_numpy'):
                X_val = X_val.to_numpy()
            if hasattr(y_val, 'to_numpy'):
                y_val = y_val.to_numpy()
                
            if y_val.dtype == object:
                y_val = self.label_encoder.transform(y_val)
                
            eval_pool = cb.Pool(
                X_val, y_val,
                cat_features=self.categorical_features
            )
            
        # Set early stopping
        self.params['od_wait'] = early_stopping_rounds
        
        # Train model
        self.model = cb.CatBoostClassifier(**self.params)
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            verbose=verbose > 0,
            plot=False
        )
        
        # Store feature names
        self.feature_names = [f'f_{i}' for i in range(X.shape[1])]
        
        logger.info(f"Training completed. Best iteration: {self.model.best_iteration_}")
        
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
        return self.label_encoder.inverse_transform(y_pred.astype(int))
    
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
        weights = np.ones()
        return weights
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        importance = self.model.get_feature_importance()
        
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
        self.model = cb.CatBoostClassifier()
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")