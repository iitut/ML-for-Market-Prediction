"""
Main ML model implementation with XGBoost and neural networks
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v1\\model.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

from config import *

class MarketPredictionModel:
    def __init__(self, features_df, targets_df):
        """
        Initialize the model with features and targets
        """
        self.features_df = features_df[:-1]  # Remove last row to match targets
        self.targets_df = targets_df
        
        # Prepare feature columns (exclude non-numeric and target-like columns)
        exclude_cols = ['date', 'crash', 'boom', 'normal', 'crash_boom_label', 
                       'standardized_return', 'next_day_return', 'log_rv_next', 
                       'rv_next', 'next_return_quantile']
        
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        self.feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Initialize models
        self.crash_boom_model = None
        self.volatility_model = None
        self.quantile_models = {}
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Training history
        self.training_history = {}
        self.predictions_history = []
        
    def prepare_data(self, train_idx, test_idx):
        """
        Prepare train and test data ensuring no leakage
        """
        # Features
        X_train = self.features_df.iloc[train_idx][self.feature_cols]
        X_test = self.features_df.iloc[test_idx][self.feature_cols]
        
        # Handle missing values
        X_train = X_train.fillna(method='ffill').fillna(0)
        X_test = X_test.fillna(method='ffill').fillna(0)
        
        # Scale features (fit only on training data)
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Targets
        y_crash_boom_train = self.targets_df.iloc[train_idx]['crash_boom_label']
        y_crash_boom_test = self.targets_df.iloc[test_idx]['crash_boom_label']
        
        y_vol_train = self.targets_df.iloc[train_idx]['log_rv_next'].fillna(0)
        y_vol_test = self.targets_df.iloc[test_idx]['log_rv_next'].fillna(0)
        
        # Store for monitoring
        self.train_dates = self.targets_df.iloc[train_idx]['date'] if 'date' in self.targets_df else None
        self.test_dates = self.targets_df.iloc[test_idx]['date'] if 'date' in self.targets_df else None
        
        return (X_train_scaled, X_test_scaled, 
                y_crash_boom_train, y_crash_boom_test,
                y_vol_train, y_vol_test)
    
    def build_crash_boom_xgboost(self):
        """
        Build XGBoost model for crash/boom classification
        """
        self.crash_boom_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False,
            tree_method='hist',  # Faster training
            device='cpu'  # Use GPU if available: 'cuda'
        )
        return self.crash_boom_model
    
    def build_volatility_neural_net(self, input_dim):
        """
        Build neural network for volatility prediction
        """
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Output: log(RV)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_quantile_model(self, quantile, input_dim):
        """
        Build model for quantile prediction
        """
        def quantile_loss(q):
            def loss(y_true, y_pred):
                error = y_true - y_pred
                return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            return loss
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=quantile_loss(quantile)
        )
        
        return model
    
    def train_expanding_window(self, initial_train_size=252):
        """
        Train using expanding window approach
        """
        print(f"\nTraining with expanding window (initial size: {initial_train_size})")
        
        total_samples = len(self.features_df)
        predictions = []
        
        # Start from initial_train_size
        for current_end in range(initial_train_size, total_samples):
            if current_end % 100 == 0 or current_end == initial_train_size:
                print(f"\nTraining up to sample {current_end}/{total_samples}")
            
            # Use all data up to current_end for training
            train_idx = list(range(current_end))
            test_idx = [current_end]  # Predict only next point
            
            # Prepare data
            X_train, X_test, y_cb_train, y_cb_test, y_vol_train, y_vol_test = self.prepare_data(
                train_idx, test_idx
            )
            
            # Skip if not enough data
            if len(X_train) < 20:
                continue
            
            # Train crash/boom model
            if current_end % 20 == 0 or current_end == initial_train_size:  # Retrain periodically
                self.build_crash_boom_xgboost()
                
                # Encode labels
                le = LabelEncoder()
                y_cb_train_encoded = le.fit_transform(y_cb_train)
                
                self.crash_boom_model.fit(
                    X_train, y_cb_train_encoded,
                    eval_set=[(X_train[-50:], y_cb_train_encoded[-50:])],
                    verbose=False
                )
            
            # Make predictions
            if self.crash_boom_model is not None:
                cb_pred_proba = self.crash_boom_model.predict_proba(X_test)
                cb_pred = le.inverse_transform([np.argmax(cb_pred_proba)])[0]
            else:
                cb_pred = 'normal'
                cb_pred_proba = [[0.33, 0.33, 0.34]]
            
            # Train volatility model (simplified for speed)
            if current_end % 50 == 0 or current_end == initial_train_size:
                self.volatility_model = self.build_volatility_neural_net(X_train.shape[1])
                
                # Quick training
                self.volatility_model.fit(
                    X_train, y_vol_train,
                    epochs=10,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.1
                )
            
            # Volatility prediction
            if self.volatility_model is not None:
                vol_pred = self.volatility_model.predict(X_test, verbose=0)[0, 0]
            else:
                vol_pred = y_vol_train.mean()
            
            # Store prediction
            prediction_record = {
                'index': current_end,
                'date': self.targets_df.iloc[current_end]['date'] if 'date' in self.targets_df else current_end,
                'actual_label': y_cb_test.iloc[0] if len(y_cb_test) > 0 else 'normal',
                'predicted_label': cb_pred,
                'crash_prob': cb_pred_proba[0][0] if len(cb_pred_proba[0]) > 0 else 0,
                'normal_prob': cb_pred_proba[0][1] if len(cb_pred_proba[0]) > 1 else 0,
                'boom_prob': cb_pred_proba[0][2] if len(cb_pred_proba[0]) > 2 else 0,
                'actual_log_rv': y_vol_test.iloc[0] if len(y_vol_test) > 0 else 0,
                'predicted_log_rv': vol_pred,
                'training_size': len(train_idx)
            }
            
            predictions.append(prediction_record)
            
            # Monitor specific days
            if current_end in MONITOR_DAYS or current_end == initial_train_size:
                self.display_training_info(current_end, train_idx, prediction_record)
        
        self.predictions_history = pd.DataFrame(predictions)
        return self.predictions_history
    
    def display_training_info(self, day_index, train_idx, prediction):
        """
        Display detailed information for monitored days
        """
        print(f"\n{'='*60}")
        print(f"DETAILED MONITORING - Day {day_index}")
        print(f"{'='*60}")
        
        print(f"\nTraining Data Used:")
        print(f"- Training samples: {len(train_idx)}")
        print(f"- Training range: indices {min(train_idx)} to {max(train_idx)}")
        
        if 'date' in self.targets_df:
            train_dates = self.targets_df.iloc[train_idx]['date']
            print(f"- Date range: {train_dates.min()} to {train_dates.max()}")
        
        print(f"\nFeatures used ({len(self.feature_cols)} total):")
        print(f"- First 5 features: {self.feature_cols[:5]}")
        
        print(f"\nPrediction Results:")
        print(f"- Actual label: {prediction['actual_label']}")
        print(f"- Predicted label: {prediction['predicted_label']}")
        print(f"- Crash probability: {prediction['crash_prob']:.2%}")
        print(f"- Normal probability: {prediction['normal_prob']:.2%}")
        print(f"- Boom probability: {prediction['boom_prob']:.2%}")
        print(f"- Actual log(RV): {prediction['actual_log_rv']:.4f}")
        print(f"- Predicted log(RV): {prediction['predicted_log_rv']:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.crash_boom_model, 'feature_importances_'):
            top_features_idx = np.argsort(self.crash_boom_model.feature_importances_)[-5:]
            print(f"\nTop 5 Important Features:")
            for idx in top_features_idx:
                print(f"- {self.feature_cols[idx]}: {self.crash_boom_model.feature_importances_[idx]:.4f}")
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        """
        if len(self.predictions_history) == 0:
            print("No predictions to evaluate")
            return None
        
        df = self.predictions_history
        
        # Classification metrics
        accuracy = accuracy_score(df['actual_label'], df['predicted_label'])
        
        # Volatility metrics
        vol_mse = mean_squared_error(df['actual_log_rv'], df['predicted_log_rv'])
        vol_mae = np.mean(np.abs(df['actual_log_rv'] - df['predicted_log_rv']))
        
        # Calculate percentage accuracy for different targets
        crash_mask = df['actual_label'] == 'crash'
        boom_mask = df['actual_label'] == 'boom'
        normal_mask = df['actual_label'] == 'normal'
        
        metrics = {
            'overall_accuracy': accuracy,
            'crash_precision': (df[crash_mask]['predicted_label'] == 'crash').mean() if crash_mask.any() else 0,
            'boom_precision': (df[boom_mask]['predicted_label'] == 'boom').mean() if boom_mask.any() else 0,
            'normal_precision': (df[normal_mask]['predicted_label'] == 'normal').mean() if normal_mask.any() else 0,
            'volatility_mse': vol_mse,
            'volatility_mae': vol_mae,
            'volatility_accuracy': 1 - (vol_mae / df['actual_log_rv'].std())  # Normalized accuracy
        }
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"Classification Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"Crash Detection Rate: {metrics['crash_precision']:.2%}")
        print(f"Boom Detection Rate: {metrics['boom_precision']:.2%}")
        print(f"Normal Detection Rate: {metrics['normal_precision']:.2%}")
        print(f"Volatility MAE: {metrics['volatility_mae']:.4f}")
        print(f"Volatility Accuracy: {metrics['volatility_accuracy']:.2%}")
        
        return metrics

# Test the model
if __name__ == "__main__":
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from target_creation import TargetCreator
    
    print("Starting model training pipeline...")
    
    # Load data
    loader = DataLoader()
    master = loader.load_master_dataset()
    intraday = loader.load_intraday_data()
    
    # Engineer features
    engineer = FeatureEngineer(master, intraday)
    features = engineer.engineer_all_features()
    
    # Create targets
    target_creator = TargetCreator(features, gamma=CRASH_BOOM_THRESHOLD)
    targets = target_creator.create_all_targets()
    
    # Initialize and train model
    model = MarketPredictionModel(features, targets)
    print(f"Using {len(model.feature_cols)} features for prediction")
    
    # Train with expanding window
    predictions = model.train_expanding_window(initial_train_size=252)
    
    # Calculate metrics
    metrics = model.calculate_metrics()
    
    print("\nTraining complete!")
    print(f"Total predictions made: {len(predictions)}")