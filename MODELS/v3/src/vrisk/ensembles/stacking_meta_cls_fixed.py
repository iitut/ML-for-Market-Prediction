"""
Stacking ensemble for classification with utility-aware optimization.
Combines multiple base learners using a meta-classifier.
FIXED VERSION - Correct implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class StackingMetaClassifier(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble with utility-aware meta-learning.
    Trains base models and combines with meta-classifier.
    """
    
    def __init__(self,
                 base_models: Dict[str, Any],
                 meta_model: str = 'lightgbm',
                 use_proba: bool = True,
                 use_features: bool = False,
                 cv_splitter: Any = None,
                 utility_matrix: Optional[np.ndarray] = None):
        """
        Initialize stacking classifier.
        
        Args:
            base_models: Dictionary of {name: model} for base learners
            meta_model: Type of meta-model ('lightgbm' or 'logistic')
            use_proba: Use probabilities (True) or predictions (False)
            use_features: Include original features in meta
            cv_splitter: Cross-validation splitter for OOF generation
            utility_matrix: Cost/utility matrix for optimization
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_proba = use_proba
        self.use_features = use_features
        self.cv_splitter = cv_splitter
        self.utility_matrix = utility_matrix
        
        self.base_models_ = {}
        self.meta_model_ = None
        self.oof_predictions_ = None
        self.classes_ = None
        
    def fit(self, 
           X: np.ndarray, 
           y: np.ndarray,
           dates: Optional[pd.Series] = None,
           sample_weight: Optional[np.ndarray] = None):
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Training labels
            dates: Dates for time series CV
            sample_weight: Sample weights
            
        Returns:
            Self
        """
        logger.info("Training stacking ensemble")
        
        # Store classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Generate out-of-fold predictions
        self.oof_predictions_ = self._generate_oof_predictions(
            X, y, dates, sample_weight
        )
        
        # Train base models on full data
        logger.info("Training base models on full data")
        for name, model in self.base_models.items():
            logger.info(f"Training {name}")
            model_clone = self._clone_model(model)
            
            if sample_weight is not None:
                if hasattr(model_clone, 'fit'):
                    try:
                        model_clone.fit(X, y, sample_weight=sample_weight)
                    except:
                        model_clone.fit(X, y)
                else:
                    model_clone.fit(X, y)
            else:
                model_clone.fit(X, y)
                
            self.base_models_[name] = model_clone
            
        # Prepare meta features
        meta_features = self._prepare_meta_features(self.oof_predictions_, X)
        
        # Train meta-model
        logger.info(f"Training meta-model: {self.meta_model}")
        self.meta_model_ = self._create_meta_model()
        
        if self.utility_matrix is not None:
            # Use utility-weighted training
            meta_weights = self._compute_utility_weights(y)
            self.meta_model_.fit(meta_features, y, sample_weight=meta_weights)
        else:
            self.meta_model_.fit(meta_features, y, sample_weight=sample_weight)
            
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        # Get base model predictions
        base_preds = self._get_base_predictions(X)
        
        # Prepare meta features
        meta_features = self._prepare_meta_features(base_preds, X)
        
        # Get meta predictions
        if hasattr(self.meta_model_, 'predict_proba'):
            proba = self.meta_model_.predict_proba(meta_features)
        else:
            # For models without predict_proba
            proba = self.meta_model_.predict(meta_features)
            if proba.ndim == 1:
                proba = proba.reshape(-1, len(self.classes_))
                
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
        return self.classes_[np.argmax(proba, axis=1)]
    
    def _generate_oof_predictions(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 dates: Optional[pd.Series],
                                 sample_weight: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Generate out-of-fold predictions for stacking.
        
        Args:
            X: Features
            y: Labels
            dates: Dates for CV
            sample_weight: Sample weights
            
        Returns:
            Dictionary of OOF predictions per model
        """
        logger.info("Generating out-of-fold predictions")
        
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        oof_preds = {}
        
        for name, model in self.base_models.items():
            logger.info(f"Generating OOF for {name}")
            
            if self.use_proba:
                oof = np.zeros((n_samples, n_classes))
            else:
                oof = np.zeros(n_samples)
                
            # Use CV splitter if provided
            if self.cv_splitter is not None and dates is not None:
                splits = self.cv_splitter.split(dates)
            else:
                # Fallback to simple k-fold
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=5, shuffle=False)
                splits = skf.split(X, y)
                
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone model for this fold
                model_clone = self._clone_model(model)
                
                # Train on fold
                if sample_weight is not None:
                    w_train = sample_weight[train_idx]
                    try:
                        model_clone.fit(X_train, y_train, sample_weight=w_train)
                    except:
                        model_clone.fit(X_train, y_train)
                else:
                    model_clone.fit(X_train, y_train)
                    
                # Predict on validation
                if self.use_proba and hasattr(model_clone, 'predict_proba'):
                    oof[val_idx] = model_clone.predict_proba(X_val)
                else:
                    preds = model_clone.predict(X_val)
                    if self.use_proba:
                        # Convert to one-hot
                        for i, pred in enumerate(preds):
                            class_idx = np.where(self.classes_ == pred)[0][0]
                            oof[val_idx[i], class_idx] = 1
                    else:
                        oof[val_idx] = preds
                        
            oof_preds[name] = oof
            
        return oof_preds
    
    def _get_base_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from trained base models."""
        base_preds = {}
        
        for name, model in self.base_models_.items():
            if self.use_proba and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X)
                if self.use_proba:
                    # Convert to probabilities
                    n_classes = len(self.classes_)
                    proba = np.zeros((len(preds), n_classes))
                    for i, pred in enumerate(preds):
                        class_idx = np.where(self.classes_ == pred)[0][0]
                        proba[i, class_idx] = 1
                    preds = proba
                    
            base_preds[name] = preds
            
        return base_preds
    
    def _prepare_meta_features(self,
                              base_preds: Dict[str, np.ndarray],
                              X: Optional[np.ndarray] = None) -> np.ndarray:
        """Prepare features for meta-model."""
        # Stack base predictions
        if self.use_proba:
            # Flatten probabilities
            meta_features = []
            for name in sorted(base_preds.keys()):
                meta_features.append(base_preds[name])
            meta_features = np.hstack(meta_features)
        else:
            # Stack predictions
            meta_features = np.column_stack([
                base_preds[name] for name in sorted(base_preds.keys())
            ])
            
        # Add diversity features
        if self.use_proba and len(base_preds) > 1:
            # Add disagreement metrics
            preds_array = np.array([base_preds[n].argmax(axis=1) 
                                   for n in sorted(base_preds.keys())])
            
            # Prediction variance
            pred_var = np.var(preds_array, axis=0)
            
            # Add to meta features
            meta_features = np.column_stack([meta_features, pred_var])
            
        # Include original features if requested
        if self.use_features and X is not None:
            meta_features = np.hstack([meta_features, X])
            
        return meta_features
    
    def _create_meta_model(self):
        """Create the meta-model."""
        if self.meta_model == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            )
        elif self.meta_model == 'logistic':
            return LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta model: {self.meta_model}")
            
    def _clone_model(self, model):
        """Clone a model instance."""
        from sklearn.base import clone
        try:
            return clone(model)
        except:
            # For custom models that don't support sklearn clone
            return model.__class__(**model.get_params()
                                 if hasattr(model, 'get_params') else {})
            
    def _compute_utility_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights based on utility matrix."""
        if self.utility_matrix is None:
            return None
            
        weights = np.ones(len(y))
        
        # Weight by potential utility impact
        for i, label in enumerate(y):
            class_idx = np.where(self.classes_ == label)[0][0]
            # Use diagonal of utility matrix as importance
            weights[i] = abs(self.utility_matrix[class_idx, class_idx])
            
        return weights
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from meta-model if available."""
        if self.meta_model_ is None:
            raise ValueError("Model not trained yet")
            
        importance_df = pd.DataFrame()
        
        if hasattr(self.meta_model_, 'feature_importances_'):
            # For tree-based models
            feature_names = []
            for name in sorted(self.base_models_.keys()):
                if self.use_proba:
                    for i in range(len(self.classes_)):
                        feature_names.append(f'{name}_class_{i}')
                else:
                    feature_names.append(name)
                    
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(self.meta_model_.feature_importances_)],
                'importance': self.meta_model_.feature_importances_
            })
            
        elif hasattr(self.meta_model_, 'coef_'):
            # For linear models
            feature_names = []
            for name in sorted(self.base_models_.keys()):
                if self.use_proba:
                    for i in range(len(self.classes_)):
                        feature_names.append(f'{name}_class_{i}')
                else:
                    feature_names.append(name)
                    
            importance_df = pd.DataFrame({
                'feature': feature_names[:self.meta_model_.coef_.shape[1]],
                'importance': np.abs(self.meta_model_.coef_).mean(axis=0)
            })
            
        return importance_df.sort_values('importance', ascending=False)