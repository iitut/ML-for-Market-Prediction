"""
Main execution script for the ML model pipeline
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v1\\main.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from config import *
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from target_creation import TargetCreator
from model import MarketPredictionModel
from visualization import ModelVisualizer

def main():
    """
    Main execution function
    """
    print("="*70)
    print("MARKET PREDICTION MODEL v1.0")
    print("="*70)
    print(f"Execution started at: {datetime.now()}")
    print(f"Model path: {MODEL_PATH}")
    print("="*70)
    
    # Step 1: Load all data
    print("\n[Step 1/6] Loading Data...")
    print("-"*50)
    loader = DataLoader()
    
    master_data = loader.load_master_dataset()
    intraday_data = loader.load_intraday_data()
    external_data = loader.load_external_data()
    calendar_data = loader.load_calendar_data()
    
    # Merge all data sources
    merged_data = loader.merge_all_data()
    loader.validate_data_integrity(merged_data)
    
    # Step 2: Feature Engineering
    print("\n[Step 2/6] Engineering Features...")
    print("-"*50)
    engineer = FeatureEngineer(merged_data, intraday_data)
    features_df = engineer.engineer_all_features()
    
    # Get feature statistics
    stats, missing = engineer.get_feature_importance_stats()
    print(f"\nFeature matrix shape: {features_df.shape}")
    print(f"Features with missing values: {(missing['missing_count'] > 0).sum()}")
    
    # Step 3: Create Targets
    print("\n[Step 3/6] Creating Target Variables...")
    print("-"*50)
    target_creator = TargetCreator(features_df, gamma=CRASH_BOOM_THRESHOLD)
    targets_df = target_creator.create_all_targets()
    
    # Verify data integrity
    train_idx, test_idx = loader.get_train_test_indices(len(targets_df))
    target_creator.verify_no_leakage(train_idx, test_idx)
    target_creator.get_target_correlations()
    
    # Step 4: Model Training
    print("\n[Step 4/6] Training Models...")
    print("-"*50)
    model = MarketPredictionModel(features_df, targets_df)
    
    print(f"Number of features for training: {len(model.feature_cols)}")
    print(f"Feature names (first 10): {model.feature_cols[:10]}")
    
    # Train with expanding window
    print("\nStarting expanding window training...")
    print("This will train the model progressively, using all historical data")
    print("to predict the next day, mimicking real-world deployment.")
    print("-"*50)
    
    predictions_df = model.train_expanding_window(initial_train_size=252)
    
    # Calculate final metrics
    metrics = model.calculate_metrics()
    
    # Step 5: Visualization
    print("\n[Step 5/6] Creating Visualizations...")
    print("-"*50)
    
    visualizer = ModelVisualizer(predictions_df, targets_df)
    
    # Create monitoring table
    monitoring_table = visualizer.create_monitoring_table()
    
    # Generate plots
    print("\nGenerating performance plots...")
    
    # Save plots to model directory
    plots_dir = MODEL_PATH / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Generate all visualizations
    visualizer.plot_crash_boom_predictions(save_path=plots_dir / "crash_boom_predictions.png")
    visualizer.plot_volatility_predictions(save_path=plots_dir / "volatility_predictions.png")
    visualizer.plot_performance_dashboard(save_path=plots_dir / "performance_dashboard.png")
    
    print(f"Plots saved to: {plots_dir}")
    
    # Step 6: Save Results
    print("\n[Step 6/6] Saving Results...")
    print("-"*50)
    
    # Save predictions
    predictions_df.to_csv(MODEL_PATH / "predictions.csv", index=False)
    print(f"Predictions saved to: {MODEL_PATH / 'predictions.csv'}")
    
    # Save model metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(MODEL_PATH / "metrics.csv", index=False)
    print(f"Metrics saved to: {MODEL_PATH / 'metrics.csv'}")
    
    # Save feature importance if available
    if hasattr(model.crash_boom_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': model.feature_cols,
            'importance': model.crash_boom_model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(MODEL_PATH / "feature_importance.csv", index=False)
        print(f"Feature importance saved to: {MODEL_PATH / 'feature_importance.csv'}")
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
    
    # Final Summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"Total predictions made: {len(predictions_df)}")
    print(f"Date range: {predictions_df['date'].min()} to {predictions_df['date'].max()}" 
          if 'date' in predictions_df else "Date information not available")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Volatility accuracy: {metrics['volatility_accuracy']:.2%}")
    print("="*70)
    
    return predictions_df, metrics

if __name__ == "__main__":
    # Run the main pipeline
    try:
        predictions, metrics = main()
        print("\n✓ Model execution completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)