"""
Main execution script for ML Model v2 - Enhanced Version
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v2\\main_v2.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from sklearn.impute import SimpleImputer
from visualization_v2 import create_all_visualizations   # <-- add this


warnings.filterwarnings('ignore')

def proba_to_labels(proba: np.ndarray, encoder) -> np.ndarray:
    # proba shape: (n_samples, n_classes)
    idx = np.argmax(proba, axis=1)
    return encoder.classes_[idx]

from ensemble_model import EnsembleMarketPredictor as EMClass
import ensemble_model as em

print(">>> ensemble_model loaded from:", em.__file__)
print(">>> Has initialize_all_models:", hasattr(EMClass, "initialize_all_models"))


# Add parent directory to path for imports
sys.path.append(r"C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\MODELS\v1")

# Import v1 modules that we'll reuse
from data_loader import DataLoader
from feature_engineering import FeatureEngineer

# Import v2 modules
from config import *
from enhanced_features import EnhancedFeatureEngineer
from ensemble_model import EnsembleMarketPredictor

def create_enhanced_targets(features_df):
    """Create all target variables for v2"""
    targets = {}
    
    if 'close' in features_df.columns:
        # Calculate returns
        features_df['daily_return'] = features_df['close'].pct_change()
        features_df['log_return'] = np.log(features_df['close'] / features_df['close'].shift(1))
        
        # 1. Market direction (7 classes) based on daily returns
        next_return = features_df['daily_return'].shift(-1)
        
        conditions = [
            next_return <= THRESHOLDS['extreme_crash'],
            (next_return > THRESHOLDS['extreme_crash']) & (next_return <= THRESHOLDS['crash']),
            (next_return > THRESHOLDS['crash']) & (next_return <= THRESHOLDS['mild_down']),
            (next_return > THRESHOLDS['mild_down']) & (next_return < THRESHOLDS['mild_up']),
            (next_return >= THRESHOLDS['mild_up']) & (next_return < THRESHOLDS['boom']),
            (next_return >= THRESHOLDS['boom']) & (next_return < THRESHOLDS['extreme_boom']),
            next_return >= THRESHOLDS['extreme_boom']
        ]
        
        choices = ['extreme_crash', 'crash', 'mild_down', 'normal', 'mild_up', 'boom', 'extreme_boom']
        targets['market_direction_7class'] = np.select(conditions, choices, default='normal')
        
        # 2. Simple direction (3 classes)
        targets['direction_3class'] = np.where(
            next_return < -0.002, 'down',
            np.where(next_return > 0.002, 'up', 'neutral')
        )
        
        # 3. Next day return (continuous)
        targets['next_day_return'] = next_return
        
        # 4. Next week return (continuous)
        targets['next_week_return'] = features_df['close'].pct_change(periods=5).shift(-5)
        
        # 5. Next month return (continuous)
        targets['next_month_return'] = features_df['close'].pct_change(periods=21).shift(-21)
        
        # 6. Week outlook (classification)
        week_return = targets['next_week_return']
        targets['week_outlook'] = np.where(
            week_return < -0.03, 'week_crash',
            np.where(week_return > 0.03, 'week_boom', 'week_normal')
        )
        
        # 7. Crash/Boom probabilities (will be calculated from ensemble)
        targets['crash_probability_1d'] = (next_return < THRESHOLDS['crash']).astype(int)
        targets['boom_probability_1d'] = (next_return > THRESHOLDS['boom']).astype(int)
        targets['crash_probability_1w'] = (week_return < -0.03).astype(int)
        targets['boom_probability_1w'] = (week_return > 0.03).astype(int)
        
        # 8. Volatility forecast (if realized variance available)
        if 'realized_variance' in features_df.columns:
            targets['volatility_forecast'] = np.log(features_df['realized_variance'].shift(-1) + 1e-8)
        else:
            targets['volatility_forecast'] = features_df['daily_return'].rolling(20).std().shift(-1)
    
    # Add date for reference
    targets['date'] = features_df['date'] if 'date' in features_df else features_df.index
    
    return pd.DataFrame(targets)[:-1]  # Remove last row (no future data)

def calculate_intraday_targets(features_df, intraday_data):
    """Calculate hour after open return"""
    if intraday_data is None:
        return None
    
    hour_returns = []
    dates = features_df['date'].unique() if 'date' in features_df else []
    
    for date in dates:
        day_data = intraday_data[intraday_data['date'] == date]
        if len(day_data) > 60:
            # Get open price and price after 1 hour
            open_price = day_data.iloc[0]['close']
            hour_price = day_data.iloc[60]['close']
            
            if open_price > 0:
                hour_return = (hour_price - open_price) / open_price
                hour_returns.append({'date': date, 'hour_after_open_return': hour_return})
            else:
                hour_returns.append({'date': date, 'hour_after_open_return': np.nan})
        else:
            hour_returns.append({'date': date, 'hour_after_open_return': np.nan})
    
    return pd.DataFrame(hour_returns)

def print_model_summary():
    """Print comprehensive model summary"""
    print("\n" + "="*80)
    print("MODEL v2 CONFIGURATION SUMMARY")
    print("="*80)
    
    print("\n📊 CLASSIFICATION THRESHOLDS:")
    print("-"*40)
    for name, value in THRESHOLDS.items():
        if isinstance(value, tuple):
            print(f"{name:15} : {value[0]:.1%} to {value[1]:.1%}")
        else:
            print(f"{name:15} : {value:.1%}")
    
    print("\n🎯 PREDICTION OUTPUTS:")
    print("-"*40)
    print("Classification:")
    for key, value in PREDICTION_OUTPUTS['classification'].items():
        print(f"  - {key}: {len(value) if isinstance(value, list) else value} classes")
    print("\nRegression:")
    for key in PREDICTION_OUTPUTS['regression'].keys():
        print(f"  - {key}")
    print("\nProbabilistic:")
    for key in PREDICTION_OUTPUTS['probabilistic'].keys():
        print(f"  - {key}")
    
    print("\n🤖 ENSEMBLE MODELS:")
    print("-"*40)
    for model in ENSEMBLE_MODELS:
        print(f"  ✓ {model}")
    
    print("\n📈 ENHANCED FEATURES:")
    print("-"*40)
    total_features = sum(len(v) for v in FEATURE_CATEGORIES.values())
    print(f"Total feature categories: {len(FEATURE_CATEGORIES)}")
    print(f"Total features planned: {total_features}")
    for category, features in FEATURE_CATEGORIES.items():
        print(f"  - {category}: {len(features)} features")

def generate_final_report(predictions_df, targets_df):
    """Generate comprehensive text report of results"""
    report = []
    report.append("\n" + "="*80)
    report.append("FINAL MODEL PERFORMANCE REPORT")
    report.append("="*80)
    
    # Overall statistics
    report.append("\n📊 OVERALL STATISTICS:")
    report.append("-"*40)
    report.append(f"Total predictions made: {len(predictions_df)}")
    
    if 'date' in predictions_df.columns:
        report.append(f"Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}")
    
    # Classification performance
    report.append("\n🎯 CLASSIFICATION PERFORMANCE:")
    report.append("-"*40)
    
    # 7-class performance (align by test indices stored in predictions_df['index'])
    test_idx = predictions_df['index'].astype(int)
    actual = targets_df.loc[test_idx, 'market_direction_7class'].reset_index(drop=True)
    if 'pred_market_direction_7class' in predictions_df.columns:
        predicted = predictions_df['pred_market_direction_7class'].reset_index(drop=True)

        # Drop rows where either side is missing
        mask = actual.notna() & predicted.notna()
        if mask.any():
            accuracy = (actual[mask] == predicted[mask]).mean()
            report.append(f"7-Class Market Direction Accuracy: {accuracy:.2%}")

            # Per-class accuracy
            for class_name in ['extreme_crash', 'crash', 'mild_down', 'normal', 'mild_up', 'boom', 'extreme_boom']:
                class_mask = (actual == class_name) & mask
                if class_mask.any():
                    class_acc = (predicted[class_mask] == class_name).mean()
                    class_count = int(class_mask.sum())
                    report.append(f"  {class_name:15} : {class_acc:6.2%} (n={class_count})")

    # Regression performance
    report.append("\n📈 REGRESSION PERFORMANCE:")
    report.append("-"*40)
    
    regression_targets = ['next_day_return', 'next_week_return', 'next_month_return']
    test_idx = predictions_df['index'].astype(int)

    for target in regression_targets:
        pred_col = f'pred_{target}'
        if target in targets_df.columns and pred_col in predictions_df.columns:
            actual = targets_df.loc[test_idx, target].to_numpy()
            predicted = predictions_df[pred_col].to_numpy()

            # Remove pairs with NaN
            mask = ~np.isnan(actual) & ~np.isnan(predicted)
            if mask.sum() >= 2:
                mae = np.mean(np.abs(actual[mask] - predicted[mask]))
                rmse = np.sqrt(np.mean((actual[mask] - predicted[mask])**2))
                correlation = np.corrcoef(actual[mask], predicted[mask])[0, 1]
            elif mask.sum() == 1:
                mae = np.abs(actual[mask][0] - predicted[mask][0])
                rmse = mae
                correlation = np.nan
            else:
                mae = rmse = correlation = np.nan

            report.append(f"\n{target.replace('_', ' ').title()}:")
            report.append(f"  MAE:         {mae:.4f}" if not np.isnan(mae) else "  MAE:         n/a")
            report.append(f"  RMSE:        {rmse:.4f}" if not np.isnan(rmse) else "  RMSE:        n/a")
            report.append(f"  Correlation: {correlation:.3f}" if not np.isnan(correlation) else "  Correlation: n/a")

    
    # Probability calibration
    report.append("\n🎲 PROBABILITY PREDICTIONS:")
    report.append("-"*40)
    
    prob_targets = ['crash_probability_1d', 'boom_probability_1d']
    for target in prob_targets:
        if target in targets_df.columns and f'pred_{target}' in predictions_df.columns:
            actual = targets_df[target].iloc[:len(predictions_df)]
            predicted = predictions_df[f'pred_{target}']
            
            # Calibration bins
            bins = np.linspace(0, 1, 11)
            calibration = []
            for i in range(len(bins)-1):
                mask = (predicted >= bins[i]) & (predicted < bins[i+1])
                if mask.any():
                    actual_rate = actual[mask].mean()
                    pred_rate = predicted[mask].mean()
                    calibration.append(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: Actual={actual_rate:.2%}, Predicted={pred_rate:.2%}")
            
            report.append(f"\n{target.replace('_', ' ').title()} Calibration:")
            for cal in calibration[:3]:  # Show first 3 bins
                report.append(cal)
    
    # Model ensemble weights
    report.append("\n🤝 ENSEMBLE MODEL WEIGHTS:")
    report.append("-"*40)
    # This would be populated from actual ensemble weights
    report.append("Weights are dynamically calculated based on validation performance")
    
    # Key insights
    report.append("\n💡 KEY INSIGHTS:")
    report.append("-"*40)
    
    # Calculate some insights
    if 'pred_next_day_return' in predictions_df.columns:
        avg_pred_return = predictions_df['pred_next_day_return'].mean()
        report.append(f"Average predicted daily return: {avg_pred_return:.4%}")
    
    if 'pred_volatility_forecast' in predictions_df.columns:
        avg_vol = np.exp(predictions_df['pred_volatility_forecast'].mean())
        report.append(f"Average predicted volatility: {avg_vol:.4f}")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)

def main():
    """Main execution function for v2"""
    print("="*80)
    print("MARKET PREDICTION MODEL v2.0 - ENHANCED VERSION")
    print("="*80)
    print(f"Execution started at: {datetime.now()}")
    print(f"Model path: {MODEL_PATH}")
    print("="*80)
    
    # Print model configuration
    print_model_summary()
    
    # Step 1: Load all data
    print("\n[Step 1/7] Loading Data...")
    print("-"*50)
    loader = DataLoader()
    
    master_data = loader.load_master_dataset()
    intraday_data = loader.load_intraday_data()
    external_data = loader.load_external_data()
    calendar_data = loader.load_calendar_data()
    
    merged_data = loader.merge_all_data()
    loader.validate_data_integrity(merged_data)
    
    # Step 2: Basic Feature Engineering (from v1)
    print("\n[Step 2/7] Engineering Basic Features...")
    print("-"*50)
    basic_engineer = FeatureEngineer(merged_data, intraday_data)
    basic_features = basic_engineer.engineer_all_features()
    
    # Step 3: Enhanced Feature Engineering
    print("\n[Step 3/7] Engineering Enhanced Features...")
    print("-"*50)
    enhanced_engineer = EnhancedFeatureEngineer(basic_features, intraday_data)
    all_features = enhanced_engineer.engineer_all_enhanced_features()
    
    print(f"Total features after enhancement: {all_features.shape[1]}")
    
    # Step 4: Create Targets
    print("\n[Step 4/7] Creating Enhanced Targets...")
    print("-"*50)
    targets_df = create_enhanced_targets(all_features)
    
    # Add intraday targets if available
    intraday_targets = calculate_intraday_targets(all_features, intraday_data)
    if intraday_targets is not None:
        targets_df = pd.merge(targets_df, intraday_targets, on='date', how='left')
    
    print(f"Total targets created: {len(targets_df.columns)}")
    
    # Print target distribution
    print("\nTarget distributions:")
    if 'market_direction_7class' in targets_df.columns:
        print("\n7-Class Market Direction:")
        print(targets_df['market_direction_7class'].value_counts())
    
    # Step 5: Initialize Ensemble Model
    print("\n[Step 5/7] Initializing Ensemble Models...")
    print("-"*50)
    
    ensemble_model = EnsembleMarketPredictor(all_features, targets_df)
    
    # Initialize the models (calling the correct method)
    classification_models, regression_models = ensemble_model.initialize_all_models()
    
    print(f"Number of features for training: {len(ensemble_model.feature_cols)}")
    
    # Step 6: Training with Expanding Window
    print("\n[Step 6/7] Training Models with Expanding Window...")
    print("-"*50)
    
    # Prepare for expanding window training
    initial_size = TRAINING_PARAMS['initial_train_size']
    total_samples = len(targets_df)
    all_predictions = []
    
    print(f"Starting expanding window training from {initial_size} samples...")
    
    for current_end in range(initial_size, total_samples):
  # Limit for demo------update, no limit
        if current_end % 50 == 0:
            print(f"Training up to sample {current_end}/{total_samples}")
        
        # Prepare train/validation split
        train_end = int(current_end * 0.8)
        train_idx = list(range(train_end))
        val_idx = list(range(train_end, current_end))
        test_idx = [current_end]
        
        # Prepare data
        X_train = ensemble_model.features_df.iloc[train_idx][ensemble_model.feature_cols]
        X_val   = ensemble_model.features_df.iloc[val_idx][ensemble_model.feature_cols]
        X_test  = ensemble_model.features_df.iloc[test_idx][ensemble_model.feature_cols]

        # Clean (replace ±Inf with NaN), then impute, then scale
        X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
        X_val_clean   = X_val.replace([np.inf, -np.inf], np.nan)
        X_test_clean  = X_test.replace([np.inf, -np.inf], np.nan)

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_clean)
        X_val_imp   = imputer.transform(X_val_clean)
        X_test_imp  = imputer.transform(X_test_clean)

        # If a column is entirely NaN in this fold, median stays NaN → force to 0
        X_train_imp = np.nan_to_num(X_train_imp, nan=0.0)
        X_val_imp   = np.nan_to_num(X_val_imp,   nan=0.0)
        X_test_imp  = np.nan_to_num(X_test_imp,  nan=0.0)

        # Now scale the imputed arrays
        X_train_scaled = ensemble_model.feature_scaler.fit_transform(X_train_imp)
        X_val_scaled   = ensemble_model.feature_scaler.transform(X_val_imp)
        X_test_scaled  = ensemble_model.feature_scaler.transform(X_test_imp)

        # Safety checks
        assert not np.isnan(X_train_scaled).any(), "NaNs in X_train_scaled"
        assert not np.isnan(X_val_scaled).any(),   "NaNs in X_val_scaled"
        assert not np.isnan(X_test_scaled).any(),  "NaNs in X_test_scaled"
        assert np.isfinite(X_train_scaled).all(),  "Non-finite in X_train_scaled"
        assert np.isfinite(X_val_scaled).all(),    "Non-finite in X_val_scaled"
        assert np.isfinite(X_test_scaled).all(),   "Non-finite in X_test_scaled"
        assert X_test_scaled.shape[0] == 1, f"Expected single test sample, got {X_test_scaled.shape}"


        # Prepare targets
        y_train_dict = {}
        y_val_dict = {}
        y_test_dict = {}
        
        for target_col in ['market_direction_7class', 'direction_3class', 'next_day_return', 
                          'next_week_return', 'volatility_forecast']:
            if target_col in targets_df.columns:
                y_train_dict[target_col] = targets_df[target_col].iloc[train_idx]
                y_val_dict[target_col] = targets_df[target_col].iloc[val_idx]
                y_test_dict[target_col] = targets_df[target_col].iloc[test_idx]
        
        # Train and predict
        if current_end % 20 == 0 or current_end == initial_size:  # Retrain periodically
            predictions = ensemble_model.predict_all_outputs(
                X_train_scaled,  # train
                X_val_scaled,    # val
                X_test_scaled,   # test  (single row)
                y_train_dict,
                y_val_dict,
                y_test_dict
            )

        
        # --- Build one-step-ahead (test) ensemble predictions using fitted models ---
        test_ensembles = {}

        # 1) Classification targets → predict_proba on X_test_scaled, then weight-average
        for target in ['market_direction_7class', 'direction_3class', 'week_outlook']:
            if target in predictions:
                weights = predictions[target]['weights']
                parts = []
                total_w = 0.0
                for name, model in ensemble_model.classification_models.items():
                    if name in weights:
                        try:
                            proba = model.predict_proba(X_test_scaled)  # shape (1, n_classes)
                            parts.append(proba * weights[name])
                            total_w += weights[name]
                        except Exception:
                            pass
                test_ensembles[target] = (None if total_w == 0.0
                                        else (np.sum(parts, axis=0) / total_w))

        # 2) Regression targets → predict on X_test_scaled, then weight-average
        for target in ['next_day_return', 'next_week_return', 'next_month_return',
                    'hour_after_open_return', 'volatility_forecast']:
            if target in predictions:
                weights = predictions[target]['weights']
                parts = []
                total_w = 0.0
                for name, model in ensemble_model.regression_models.items():
                    if name in weights:
                        try:
                            pred = model.predict(X_test_scaled).reshape(-1)  # shape (1,)
                            parts.append(pred * weights[name])
                            total_w += weights[name]
                        except Exception:
                            pass
                test_ensembles[target] = (None if total_w == 0.0
                                        else (np.sum(parts, axis=0) / total_w))

        
        # Make predictions for test point
        prediction_record = {
            'index': current_end,
            'date': targets_df.iloc[current_end]['date'] if 'date' in targets_df else current_end,
            'training_size': len(train_idx)
        }
        
        # Add predictions for each target
        for target in y_test_dict.keys():
            actual_value = y_test_dict[target].iloc[0] if len(y_test_dict[target]) > 0 else np.nan
            prediction_record[f'actual_{target}'] = actual_value

            # pull the freshly-computed test ensemble for THIS step
            te = test_ensembles.get(target, None)
            if te is None:
                continue

            if ('class' in target) or (target == 'week_outlook'):
                # classification → decode to original string labels
                if te is not None and len(te) > 0:
                    class_idx = int(np.argmax(te[0]))
                    le = ensemble_model.label_encoders.get(target, None)
                    pred_label = le.inverse_transform([class_idx])[0] if le is not None else class_idx
                    prediction_record[f'pred_{target}'] = pred_label
            else:
                # regression → scalar
                prediction_record[f'pred_{target}'] = float(np.ravel(te)[0])


        
        all_predictions.append(prediction_record)
        
        # Monitor specific days
        if current_end in MONITOR_DAYS:
            print(f"\n{'='*60}")
            print(f"DETAILED MONITORING - Day {current_end}")
            print(f"{'='*60}")
            print(f"Training samples: {len(train_idx)}")
            print(f"Validation samples: {len(val_idx)}")
            
            if predictions and 'market_direction_7class' in predictions:
                print(f"\nModel ensemble weights:")
                for model, weight in predictions['market_direction_7class']['weights'].items():
                    print(f"  {model}: {weight:.3f}")
    
    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Step 7: Generate Reports and Visualizations
    print("\n[Step 7/7] Generating Reports and Visualizations...")
    print("-"*50)
    
    # Generate text report
    final_report = generate_final_report(predictions_df, targets_df)
    print(final_report)
    
    # Save results
    results_dir = MODEL_PATH / "results"
    results_dir.mkdir(exist_ok=True)
    
 
   # Save predictions (UTF-8 so Excel/Notepad show them cleanly)
    predictions_df.to_csv(results_dir / "predictions_v2.csv", index=False, encoding="utf-8-sig")
    print(f"\nPredictions saved to: {results_dir / 'predictions_v2.csv'}")

    # Save report (UTF-8 so emojis don't crash on Windows)
    with open(results_dir / "performance_report.txt", "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"Report saved to: {results_dir / 'performance_report.txt'}")


    # Save feature importance if available
    feature_importance = ensemble_model.calculate_feature_importance()
    if feature_importance is not None:
        feature_importance.to_csv(results_dir / "feature_importance_v2.csv")
        print(f"Feature importance saved to: {results_dir / 'feature_importance_v2.csv'}")
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string())
    
    # >>> ADD THESE TWO LINES <<<
    print("\n[Step 7/7] Creating visualizations...")
    create_all_visualizations(predictions_df, targets_df, save_dir=results_dir)
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print(f"Model v2 execution completed at: {datetime.now()}")
    
    return predictions_df, targets_df

if __name__ == "__main__":
    try:
        predictions, targets = main()
        print("\n✓ Model v2 execution completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)