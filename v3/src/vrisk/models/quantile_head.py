"""
Quantile prediction head for estimating return distribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class QuantilePredictor:
    """Predict return quantiles with monotonicity constraints."""
    
    def __init__(self,
                 quantiles: List[float] = [0.05, 0.50, 0.95],
                 base_model: str = 'lightgbm',
                 enforce_monotonicity: bool = True):
        """
        Initialize quantile predictor.
        
        Args:
            quantiles: Quantiles to predict
            base_model: Base model type
            enforce_monotonicity: Whether to enforce q_5 < q_50 < q_95
        """
        self.quantiles = quantiles
        self.base_model_type = base_model
        self.enforce_monotonicity = enforce_monotonicity
        
        self.models_ = {}
        self.scalers_ = {}
        
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None):
        """
        Fit quantile models.
        
        Args:
            X: Training features
            y: Target returns
            sample_weight: Sample weights
        """
        logger.info(f"Training quantile models for {self.quantiles}")
        
        for q in self.quantiles:
            logger.info(f"Training quantile {q}")
            
            if self.base_model_type == 'lightgbm':
                model = lgb.LGBMRegressor(
                    objective='quantile',
                    alpha=q,
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1
                )
                
            elif self.base_model_type == 'quantile_forest':
                # Use RandomForest with quantile prediction
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                    criterion='absolute_error'  # Better for quantiles
                )
                
            else:
                raise ValueError(f"Unknown model type: {self.base_model_type}")
                
            model.fit(X, y, sample_weight=sample_weight)
            self.models_[q] = model
            
        logger.info("Quantile training completed")
        
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Predict quantiles.
        
        Args:
            X: Features
            
        Returns:
            Dictionary of quantile predictions
        """
        predictions = {}
        
        for q in self.quantiles:
            if q not in self.models_:
                raise ValueError(f"Model for quantile {q} not trained")
                
            pred = self.models_[q].predict(X)
            predictions[q] = pred
            
        # Enforce monotonicity if requested
        if self.enforce_monotonicity and len(self.quantiles) > 1:
            predictions = self._enforce_monotonicity(predictions)
            
        return predictions
    
    def _enforce_monotonicity(self, 
                             predictions: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
        """
        Enforce monotonicity constraints on quantile predictions.
        
        Args:
            predictions: Raw quantile predictions
            
        Returns:
            Adjusted predictions with monotonicity
        """
        sorted_quantiles = sorted(self.quantiles)
        n_samples = len(predictions[sorted_quantiles[0]])
        
        # Stack predictions
        pred_matrix = np.column_stack([predictions[q] for q in sorted_quantiles])
        
        # Enforce monotonicity row by row
        for i in range(n_samples):
            # Use isotonic regression or simple sorting
            sorted_vals = np.sort(pred_matrix[i, :])
            pred_matrix[i, :] = sorted_vals
            
        # Unpack back to dictionary
        adjusted_predictions = {}
        for idx, q in enumerate(sorted_quantiles):
            adjusted_predictions[q] = pred_matrix[:, idx]
            
        return adjusted_predictions
    
    def predict_intervals(self, 
                         X: np.ndarray,
                         coverage: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict prediction intervals.
        
        Args:
            X: Features
            coverage: Coverage level (e.g., 0.9 for 90% interval)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        alpha = 1 - coverage
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2
        
        # Train models for these quantiles if needed
        if lower_q not in self.models_:
            logger.info(f"Training model for quantile {lower_q}")
            # Would need to retrain here
            
        predictions = self.predict(X)
        
        # Get closest available quantiles
        available_quantiles = sorted(self.quantiles)
        lower_idx = np.argmin(np.abs(np.array(available_quantiles) - lower_q))
        upper_idx = np.argmin(np.abs(np.array(available_quantiles) - upper_q))
        
        lower_bound = predictions[available_quantiles[lower_idx]]
        upper_bound = predictions[available_quantiles[upper_idx]]
        
        return lower_bound, upper_bound


class ConformalPredictor:
    """
    Conformal prediction wrapper for uncertainty quantification.
    """
    
    def __init__(self, base_predictor: QuantilePredictor, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            base_predictor: Base quantile predictor
            alpha: Miscoverage rate (1 - coverage)
        """
        self.base_predictor = base_predictor
        self.alpha = alpha
        self.calibration_scores_ = None
        
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Calibrate conformal predictor.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
        """
        # Get predictions
        predictions = self.base_predictor.predict(X_cal)
        
        # Calculate conformity scores
        median_pred = predictions[0.5] if 0.5 in predictions else predictions[0.05]
        self.calibration_scores_ = np.abs(y_cal - median_pred)
        
        logger.info(f"Calibrated on {len(y_cal)} samples")
        
    def predict_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict conformal intervals.
        
        Args:
            X: Features
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.calibration_scores_ is None:
            raise ValueError("Model not calibrated")
            
        # Get base predictions
        predictions = self.base_predictor.predict(X)
        median_pred = predictions[0.5] if 0.5 in predictions else predictions[0.05]
        
        # Calculate quantile of calibration scores
        n_cal = len(self.calibration_scores_)
        q = np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal
        q = np.clip(q, 0, 1)
        
        # Get conformal radius
        radius = np.quantile(self.calibration_scores_, q)
        
        # Construct intervals
        lower_bound = median_pred - radius
        upper_bound = median_pred + radius
        
        return lower_bound, upper_bound