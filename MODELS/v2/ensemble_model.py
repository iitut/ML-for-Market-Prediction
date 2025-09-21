"""
Ensemble model combining multiple ML algorithms
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v2\\ensemble_model.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
from sklearn.impute import SimpleImputer



warnings.filterwarnings('ignore')

from config import *

class EnsembleMarketPredictor:
    def __init__(self, features_df, targets_df):
        """Initialize ensemble model system"""
        self.features_df = features_df[:-1]  # Remove last row to match targets
        self.targets_df = targets_df
        
        # Prepare feature columns
        exclude_cols = ['date'] + list(targets_df.columns)
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        self.feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Initialize model dictionaries
        self.classification_models = {}
        self.regression_models = {}
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Predictions storage
        self.ensemble_predictions = {}
        self.model_weights = {}
        
    def build_xgboost_models(self):
        """Build XGBoost models for classification and regression"""
        models = {}
        
        # Classification model
        models['xgb_classifier'] = xgb.XGBClassifier(**XGBOOST_PARAMS, use_label_encoder=False)
        
        # Regression model
        models['xgb_regressor'] = xgb.XGBRegressor(**XGBOOST_PARAMS)
        
        return models
    
    def build_lightgbm_models(self):
        """Build LightGBM models"""
        models = {}
        
        # Classification model
        models['lgb_classifier'] = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
        
        # Regression model
        models['lgb_regressor'] = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
        
        return models
    
    def build_catboost_models(self):
        """Build CatBoost models"""
        models = {}
        
        # Classification model
        models['cat_classifier'] = CatBoostClassifier(**CATBOOST_PARAMS)
        
        # Regression model
        models['cat_regressor'] = CatBoostRegressor(**CATBOOST_PARAMS)
        
        return models
    
    def build_random_forest_models(self):
        """Build Random Forest models"""
        models = {}
        
        # Classification model
        models['rf_classifier'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Regression model
        models['rf_regressor'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        return models
    
    def build_gradient_boosting_models(self):
        """Build Gradient Boosting models"""
        models = {}
        
        # Classification model
        models['gb_classifier'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            random_state=42
        )
        
        # Regression model
        models['gb_regressor'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            random_state=42
        )
        
        return models
    
    def build_neural_network(self, input_dim, output_dim, task='classification'):
        """Build neural network model"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, (units, dropout) in enumerate(zip(NN_ARCHITECTURE['hidden_layers'], 
                                                  NN_ARCHITECTURE['dropout_rates'])):
            model.add(layers.Dense(units, activation=NN_ARCHITECTURE['activation']))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        # Output layer
        if task == 'classification':
            model.add(layers.Dense(output_dim, activation='softmax'))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=NN_ARCHITECTURE['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(layers.Dense(output_dim))
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=NN_ARCHITECTURE['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def initialize_all_models(self):
        """Initialize all ensemble models"""
        print("Initializing ensemble models...")
        
        # Build all model types
        xgb_models = self.build_xgboost_models()
        lgb_models = self.build_lightgbm_models()
        cat_models = self.build_catboost_models()
        rf_models = self.build_random_forest_models()
        gb_models = self.build_gradient_boosting_models()
        
        # Combine classification models
        self.classification_models = {
            'xgboost': xgb_models['xgb_classifier'],
            'lightgbm': lgb_models['lgb_classifier'],
            'catboost': cat_models['cat_classifier'],
            'random_forest': rf_models['rf_classifier'],
            'gradient_boosting': gb_models['gb_classifier']
        }
        
        # Combine regression models
        self.regression_models = {
            'xgboost': xgb_models['xgb_regressor'],
            'lightgbm': lgb_models['lgb_regressor'],
            'catboost': cat_models['cat_regressor'],
            'random_forest': rf_models['rf_regressor'],
            'gradient_boosting': gb_models['gb_regressor']
        }
        
        print(f"Initialized {len(self.classification_models)} classification models")
        print(f"Initialized {len(self.regression_models)} regression models")
        
        return self.classification_models, self.regression_models
    
    def train_classification_ensemble(self, X_train, y_train, X_val, y_val, target_name, X_test=None):
        """Train ensemble of classification models and (optionally) predict on X_test."""
        predictions_val = {}
        predictions_test = {}
        scores = {}

        # Encode labels...
        if target_name not in self.label_encoders:
            self.label_encoders[target_name] = LabelEncoder()
            y_train_encoded = self.label_encoders[target_name].fit_transform(y_train)
        else:
            y_train_encoded = self.label_encoders[target_name].transform(y_train)

        y_val_encoded = self.label_encoders[target_name].transform(y_val)

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp   = imputer.transform(X_val)
        X_train_imp = np.nan_to_num(X_train_imp, nan=0.0)
        X_val_imp   = np.nan_to_num(X_val_imp,   nan=0.0)

        X_test_imp = None
        if X_test is not None:
            X_test_imp = imputer.transform(X_test)
            X_test_imp = np.nan_to_num(X_test_imp, nan=0.0)

        sklearn_needs_impute = ("random_forest", "gradient_boosting")

        for name, model in self.classification_models.items():
            try:
                Xt = X_train_imp if name in sklearn_needs_impute else X_train
                Xv = X_val_imp   if name in sklearn_needs_impute else X_val
                Xte = X_test_imp if (X_test is not None and name in sklearn_needs_impute) else X_test

                if name == 'catboost':
                    model.fit(Xt, y_train_encoded, eval_set=(Xv, y_val_encoded), verbose=False)
                else:
                    model.fit(Xt, y_train_encoded)

                # --- VAL predictions (for weighting)
                if hasattr(model, "predict_proba"):
                    pred_proba_val = model.predict_proba(Xv)
                else:
                    pred_val = model.predict(Xv)
                    n_classes = len(np.unique(y_train_encoded))
                    eye = np.eye(n_classes)
                    pred_proba_val = eye[pred_val]

                score = accuracy_score(y_val_encoded, np.argmax(pred_proba_val, axis=1))
                predictions_val[name] = pred_proba_val
                scores[name] = score

                # --- TEST predictions (to be ensembled with the same weights)
                if X_test is not None:
                    if hasattr(model, "predict_proba"):
                        predictions_test[name] = model.predict_proba(Xte)
                    else:
                        pred_te = model.predict(Xte)
                        n_classes = len(np.unique(y_train_encoded))
                        eye = np.eye(n_classes)
                        predictions_test[name] = eye[pred_te]

            except Exception as e:
                print(f"Warning: {name} failed for {target_name}: {e}")
                predictions_val[name] = None
                scores[name] = 0.0
                if X_test is not None:
                    predictions_test[name] = None

        total_score = sum(scores.values())
        weights = {k: (v / total_score if total_score > 0 else 1/len(scores)) for k, v in scores.items()}

        return predictions_val, predictions_test, weights, scores

    
    def train_regression_ensemble(self, X_train, y_train, X_val, y_val, target_name, X_test=None):
        """Train ensemble of regression models and (optionally) predict on X_test."""
        predictions_val = {}
        predictions_test = {}
        scores = {}

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp   = imputer.transform(X_val)
        X_train_imp = np.nan_to_num(X_train_imp, nan=0.0)
        X_val_imp   = np.nan_to_num(X_val_imp,   nan=0.0)

        X_test_imp = None
        if X_test is not None:
            X_test_imp = imputer.transform(X_test)
            X_test_imp = np.nan_to_num(X_test_imp, nan=0.0)

        sklearn_needs_impute = ("random_forest", "gradient_boosting")

        for name, model in self.regression_models.items():
            try:
                Xt = X_train_imp if name in sklearn_needs_impute else X_train
                Xv = X_val_imp   if name in sklearn_needs_impute else X_val
                Xte = X_test_imp if (X_test is not None and name in sklearn_needs_impute) else X_test

                if name == 'catboost':
                    model.fit(Xt, y_train, eval_set=(Xv, y_val), verbose=False)
                else:
                    model.fit(Xt, y_train)

                pred_val = model.predict(Xv)
                score = -mean_squared_error(y_val, pred_val)
                predictions_val[name] = pred_val
                scores[name] = score

                if X_test is not None:
                    predictions_test[name] = model.predict(Xte)

            except Exception as e:
                print(f"Warning: {name} failed for {target_name}: {e}")
                predictions_val[name] = None
                scores[name] = -999.0
                if X_test is not None:
                    predictions_test[name] = None

        min_score = min(scores.values())
        shifted = {k: v - min_score + 1 for k, v in scores.items()}
        total = sum(shifted.values())
        weights = {k: (v / total if total > 0 else 1/len(scores)) for k, v in shifted.items()}

        return predictions_val, predictions_test, weights, scores

    
    def get_ensemble_prediction(self, predictions, weights, task='classification'):
   
        # Collect only models that produced a prediction AND have a weight
        valid = []
        for name, pred in predictions.items():
            if pred is None:
                continue
            if name not in weights:
                continue
            # Ensure numpy arrays for safe math
            valid.append((np.asarray(pred), float(weights[name])))

        if not valid:
            return None

        if task == 'classification':
            # pred shape: (n_samples, n_classes)
            weighted_sum = None
            total_w = 0.0
            for pred, w in valid:
                weighted_sum = pred * w if weighted_sum is None else (weighted_sum + pred * w)
                total_w += w
            return None if total_w == 0.0 else (weighted_sum / total_w)

        # task == 'regression' — pred shape: (n_samples,)
        weighted_sum = None
        total_w = 0.0
        for pred, w in valid:
            # make sure it's a 1D vector for summation
            pred_vec = pred.reshape(-1)
            weighted_sum = pred_vec * w if weighted_sum is None else (weighted_sum + pred_vec * w)
            total_w += w
        return None if total_w == 0.0 else (weighted_sum / total_w)
    
    def predict_all_outputs(self, X_train, X_val, X_test, y_train_dict, y_val_dict, y_test_dict):
        """Train models on (train,val), weight by val performance, and predict for both val & test."""
        all_predictions = {}

        # --- Classification targets ---
        classification_targets = [
            'market_direction_7class',
            'direction_3class',
            'week_outlook'
        ]
        for target in classification_targets:
            if target in y_train_dict:
                print(f"Training ensemble for {target}...")

                preds_val, preds_test, weights, scores = self.train_classification_ensemble(
                    X_train, y_train_dict[target],
                    X_val,   y_val_dict[target],
                    target,
                    X_test=X_test
                )

                ensemble_val  = self.get_ensemble_prediction(preds_val,  weights, 'classification')
                ensemble_test = self.get_ensemble_prediction(preds_test, weights, 'classification')

                all_predictions[target] = {
                    'individual_val':  preds_val,
                    'individual_test': preds_test,
                    'ensemble_val':    ensemble_val,
                    'ensemble_test':   ensemble_test,
                    'weights':         weights,
                    'scores':          scores
                }

        # --- Regression targets ---
        regression_targets = [
            'next_day_return',
            'next_week_return',
            'next_month_return',
            'hour_after_open_return',
            'volatility_forecast'
        ]
        for target in regression_targets:
            if target in y_train_dict:
                print(f"Training ensemble for {target}...")

                preds_val, preds_test, weights, scores = self.train_regression_ensemble(
                    X_train, y_train_dict[target],
                    X_val,   y_val_dict[target],
                    target,
                    X_test=X_test
                )

                ensemble_val  = self.get_ensemble_prediction(preds_val,  weights, 'regression')
                ensemble_test = self.get_ensemble_prediction(preds_test, weights, 'regression')

                all_predictions[target] = {
                    'individual_val':  preds_val,
                    'individual_test': preds_test,
                    'ensemble_val':    ensemble_val,
                    'ensemble_test':   ensemble_test,
                    'weights':         weights,
                    'scores':          scores
                }

        return all_predictions

    def calculate_feature_importance(self):
        """Calculate and combine feature importance from all *fitted* models
        whose importance vectors match the feature count."""
        importance_dict = {}

        # collect importances from models that expose them and match feature count
        for name, model in self.classification_models.items():
            try:
                if hasattr(model, "feature_importances_"):
                    imps = np.asarray(model.feature_importances_)
                    if imps.ndim == 1 and imps.shape[0] == len(self.feature_cols):
                        importance_dict[name] = imps
                    else:
                        print(f"[feature_importance] Skipping {name}: "
                            f"len(importances)={imps.shape[0]} != n_features={len(self.feature_cols)}")
            except Exception as e:
                print(f"[feature_importance] Skipping {name}: {e}")

        if not importance_dict:
            print("[feature_importance] No compatible feature importances available.")
            return None

        # average across models
        importance_df = pd.DataFrame(importance_dict, index=self.feature_cols)
        mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)
        return mean_importance

