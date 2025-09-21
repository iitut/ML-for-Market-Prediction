#!/usr/bin/env python
"""
Main training script for VOL-RISK LAB v3.
Orchestrates the complete training pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import mlflow
import json
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.vrisk.io.load_master import load_master_minute
from src.vrisk.calendars.sessions import TradingCalendar
from src.vrisk.labeling.minute_returns import calculate_minute_returns
from src.vrisk.labeling.rv_daily import calculate_daily_rv
from src.vrisk.labeling.crash_boom_labels import create_crash_boom_labels
from src.vrisk.features.vol_rv_bv_jv import VolatilityFeatures
from src.vrisk.features.microstructure_iex import MicrostructureFeatures
from src.vrisk.models.lgbm_cls import LightGBMClassifier
from src.vrisk.training.cv_splitter import TimeSeriesCVSplitter
from src.vrisk.ensembles.stacking_meta_cls import StackingMetaClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main training orchestrator for VOL-RISK LAB."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_dir = Path(config.output.base_path) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        if config.tracking.backend == 'mlflow':
            mlflow.set_experiment(config.tracking.experiment_name)
            mlflow.start_run(run_name=self.run_id)
            mlflow.log_params(OmegaConf.to_container(config))
            
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(OmegaConf.to_container(self.config)).encode()
        ).hexdigest()[:8]
        return f"{timestamp}_{config_hash}"
    
    def run(self):
        """Execute complete training pipeline."""
        logger.info(f"Starting training run: {self.run_id}")
        
        try:
            # 1. Load and validate data
            logger.info("=" * 50)
            logger.info("STEP 1: Loading and validating data")
            minute_df = self._load_data()
            
            # 2. Create labels
            logger.info("=" * 50)
            logger.info("STEP 2: Creating labels")
            daily_df = self._create_labels(minute_df)
            
            # 3. Generate features
            logger.info("=" * 50)
            logger.info("STEP 3: Generating features")
            features_df = self._generate_features(minute_df, daily_df)
            
            # 4. Prepare for modeling
            logger.info("=" * 50)
            logger.info("STEP 4: Preparing data for modeling")
            X, y, dates = self._prepare_modeling_data(features_df)
            
            # 5. Train models
            logger.info("=" * 50)
            logger.info("STEP 5: Training models")
            models = self._train_models(X, y, dates)
            
            # 6. Save outputs
            logger.info("=" * 50)
            logger.info("STEP 6: Saving outputs")
            self._save_outputs(models, features_df)
            
            logger.info(f"Training completed successfully: {self.run_id}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if self.config.tracking.backend == 'mlflow':
                mlflow.end_run()
    
    def _load_data(self) -> pl.DataFrame:
        """Load and validate master minute data."""
        logger.info(f"Loading data from {self.config.data.master_parquet}")
        
        # Load data
        df = load_master_minute(
            self.config.data.master_parquet,
            use_parquet=True,
            validate=True
        )
        
        # Add calendar features
        calendar = TradingCalendar()
        df = calendar.add_session_features(df)
        
        # Log data info
        logger.info(f"Loaded {len(df):,} minute records")
        logger.info(f"Date range: {df['session_date'].min()} to {df['session_date'].max()}")
        
        # Track data hash
        data_hash = hashlib.sha256(
            f"{df.shape}_{df.head(100).to_pandas().values.tobytes()}".encode()
        ).hexdigest()[:16]
        
        if self.config.tracking.backend == 'mlflow':
            mlflow.log_metric("n_minutes", len(df))
            mlflow.log_param("data_hash", data_hash)
            
        return df
    
    def _create_labels(self, minute_df: pl.DataFrame) -> pl.DataFrame:
        """Create classification and regression labels."""
        logger.info("Calculating minute returns")
        
        # Calculate minute returns
        minute_df = calculate_minute_returns(minute_df)
        
        # Calculate daily RV measures
        logger.info("Calculating daily volatility measures")
        rv_df = calculate_daily_rv(minute_df)
        
        # Create crash/boom labels
        logger.info("Creating crash/boom/normal labels")
        daily_df = create_crash_boom_labels(
            minute_df, 
            rv_df,
            gammas=self.config.targets.gammas,
            volatility_estimator='RV'
        )
        
        # Add volatility targets
        daily_df = daily_df.with_columns([
            # Next-day log RV target
            pl.col('log_RV_daily').shift(-1).alias('target_log_rv_1d'),
            
            # Next-day return for quantiles
            pl.col('daily_return').shift(-1).alias('target_return_1d')
        ])
        
        # Log label statistics
        gamma = self.config.targets.default_gamma
        label_col = f'label_gamma_{gamma:.1f}'
        label_counts = daily_df[label_col].value_counts()
        
        logger.info(f"Label distribution (gamma={gamma}):")
        for row in label_counts.to_dicts():
            logger.info(f"  {row[label_col]}: {row['count']}")
            
        if self.config.tracking.backend == 'mlflow':
            mlflow.log_metric("n_days", len(daily_df))
            for row in label_counts.to_dicts():
                mlflow.log_metric(f"n_{row[label_col]}", row['count'])
                
        return daily_df
    
    def _generate_features(self, minute_df: pl.DataFrame, daily_df: pl.DataFrame) -> pl.DataFrame:
        """Generate all feature blocks."""
        logger.info("Generating feature blocks")
        
        # Volatility features
        logger.info("  - Volatility features")
        vol_features = VolatilityFeatures(
            lookback_windows=self.config.features.get('lookback_windows', [5, 10, 20, 60])
        )
        daily_df = vol_features.generate(daily_df)
        
        # Microstructure features
        logger.info("  - Microstructure features")
        micro_features = MicrostructureFeatures(
            lookback_minutes=[30, 60, 120]
        )
        micro_df = micro_features.generate(minute_df)
        
        # Merge features
        features_df = daily_df.join(micro_df, on='session_date', how='left')
        
        # Add more feature blocks here (path, liquidity, macro, etc.)
        # ... (would implement all other feature modules)
        
        # Log feature statistics
        feature_cols = [c for c in features_df.columns 
                       if c.startswith(('vr_', 'iex_', 'liq_', 'path_'))]
        logger.info(f"Generated {len(feature_cols)} features")
        
        if self.config.tracking.backend == 'mlflow':
            mlflow.log_metric("n_features", len(feature_cols))
            
        return features_df
    
    def _prepare_modeling_data(self, features_df: pl.DataFrame) -> tuple:
        """Prepare data for modeling."""
        logger.info("Preparing modeling data")
        
        # Select features and target
        gamma = self.config.targets.default_gamma
        label_col = f'label_gamma_{gamma:.1f}_numeric'
        
        feature_cols = [c for c in features_df.columns 
                       if c.startswith(('vr_', 'iex_', 'liq_', 'path_', 'macro_', 'time_', 'event_'))]
        
        # Remove rows with target NaN (last day)
        clean_df = features_df.filter(pl.col(label_col).is_not_null())
        
        # Convert to numpy
        X = clean_df.select(feature_cols).to_numpy()
        y = clean_df[label_col].to_numpy()
        dates = clean_df['session_date'].to_pandas()
        
        logger.info(f"Modeling data shape: X={X.shape}, y={y.shape}")
        
        # Split train/test
        test_start = pd.Timestamp(self.config.cv.get('test_start', '2025-01-01'))
        train_mask = dates < test_start
        test_mask = dates >= test_start
        
        logger.info(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")
        
        return X, y, dates
    
    def _train_models(self, X: np.ndarray, y: np.ndarray, dates: pd.Series) -> dict:
        """Train all models in the ensemble."""
        logger.info("Training model ensemble")
        
        # Create CV splitter
        cv_splitter = TimeSeriesCVSplitter(
            n_splits=self.config.cv.n_splits,
            min_train_days=self.config.cv.min_train_days,
            test_days=self.config.cv.test_days,
            embargo_days=self.config.cv.embargo_days,
            purge_days=self.config.cv.purge_days,
            anchored=self.config.cv.anchored
        )
        
        # Split train/test
        test_start = pd.Timestamp(self.config.cv.get('test_start', '2025-01-01'))
        train_mask = dates < test_start
        test_mask = dates >= test_start
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        dates_train = dates[train_mask]
        
        # Initialize base models
        base_models = {}
        
        # LightGBM
        if 'lightgbm' in self.config.ensemble.classification_bases:
            base_models['lightgbm'] = LightGBMClassifier(
                params=self.config.models.get('lightgbm', {}),
                class_weight='balanced',
                sample_weight_power=self.config.imbalance.sample_weight_power
            )
            
        # Add other base models here (XGBoost, CatBoost, etc.)
        # ... (would implement other model classes)
        
        # Create stacking ensemble
        logger.info("Training stacking ensemble")
        
        # Define utility matrix
        utility_matrix = np.array([
            [self.config.decision.cost_matrix.crash_crash,
             self.config.decision.cost_matrix.crash_normal,
             self.config.decision.cost_matrix.crash_boom],
            [self.config.decision.cost_matrix.normal_crash,
             self.config.decision.cost_matrix.normal_normal,
             self.config.decision.cost_matrix.normal_boom],
            [self.config.decision.cost_matrix.boom_crash,
             self.config.decision.cost_matrix.boom_normal,
             self.config.decision.cost_matrix.boom_boom]
        ])
        
        ensemble = StackingMetaClassifier(
            base_models=base_models,
            meta_model=self.config.ensemble.meta.classification,
            use_proba=True,
            use_features=False,
            cv_splitter=cv_splitter,
            utility_matrix=utility_matrix
        )
        
        # Fit ensemble
        ensemble.fit(X_train, y_train, dates=dates_train)
        
        # Evaluate on test set
        if test_mask.sum() > 0:
            logger.info("Evaluating on test set")
            test_proba = ensemble.predict_proba(X_test)
            test_pred = ensemble.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import classification_report, confusion_matrix
            
            report = classification_report(y_test, test_pred, 
                                         target_names=['crash', 'normal', 'boom'])
            logger.info(f"Classification Report:\n{report}")
            
            cm = confusion_matrix(y_test, test_pred)
            logger.info(f"Confusion Matrix:\n{cm}")
            
            # Log metrics
            if self.config.tracking.backend == 'mlflow':
                mlflow.log_text(report, "classification_report.txt")
                
        return {'ensemble': ensemble}
    
    def _save_outputs(self, models: dict, features_df: pl.DataFrame):
        """Save all outputs and artifacts."""
        logger.info("Saving outputs")
        
        # Save models
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Save feature data
        features_df.write_parquet(self.output_dir / 'features.parquet')
        
        # Save configuration
        config_path = self.output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            OmegaConf.save(self.config, f)
            
        # Save run metadata
        metadata = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'config_hash': hashlib.md5(
                json.dumps(OmegaConf.to_container(self.config)).encode()
            ).hexdigest(),
            'data_shape': features_df.shape,
            'n_features': len([c for c in features_df.columns 
                             if c.startswith(('vr_', 'iex_', 'liq_'))])
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Outputs saved to {self.output_dir}")
        
        # Log artifacts to MLflow
        if self.config.tracking.backend == 'mlflow':
            mlflow.log_artifacts(str(self.output_dir))


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Run training
    trainer = ModelTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()