#!/usr/bin/env python
"""
Robust training script for VOL-RISK LAB v3.
Handles missing data gracefully and uses all available models.
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

# Import custom modules - use robust versions
from src.vrisk.io.load_master_robust import load_master_minute_robust
from src.vrisk.calendars.sessions import TradingCalendar
from src.vrisk.labeling.minute_returns import calculate_minute_returns
from src.vrisk.labeling.rv_daily import calculate_daily_rv
from src.vrisk.labeling.crash_boom_labels import create_crash_boom_labels
from src.vrisk.features.vol_rv_bv_jv import VolatilityFeatures
from src.vrisk.features.microstructure_iex import MicrostructureFeatures
from src.vrisk.features.path_gaps import PathFeatures

# Import models
from src.vrisk.models.lgbm_cls import LightGBMClassifier
from src.vrisk.models.xgb_cls import XGBoostClassifier

# Import training utilities
from src.vrisk.training.cv_splitter import TimeSeriesCVSplitter
from src.vrisk.ensembles.stacking_meta_cls_fixed import StackingMetaClassifier
from src.vrisk.utils.thresholding import DecisionPolicy
from src.vrisk.evaluation.eval_classify import ClassificationEvaluator
from src.vrisk.reporting.writer import ReportWriter

logger = logging.getLogger(__name__)


class RobustModelTrainer:
    """Robust training orchestrator that handles missing data."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize robust trainer.
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        self.run_id = self._generate_run_id()
        self.output_dir = Path(config.output.base_path) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track missing data handling
        self.missing_data_report = {}
        
        # Initialize MLflow if configured
        if config.tracking.backend == 'mlflow':
            try:
                mlflow.set_experiment(config.tracking.experiment_name)
                mlflow.start_run(run_name=self.run_id)
                mlflow.log_params(OmegaConf.to_container(config))
            except Exception as e:
                logger.warning(f"Failed to generate volatility features: {e}")
            feature_status['volatility'] = f'failed: {e}'
            
        # Microstructure features (may fail if IEX data missing)
        try:
            logger.info("  - Microstructure features")
            micro_features = MicrostructureFeatures(handle_na='forward_fill')
            micro_df = micro_features.generate(minute_df)
            daily_df = daily_df.join(micro_df, on='session_date', how='left')
            feature_status['microstructure'] = 'success'
        except Exception as e:
            logger.warning(f"Failed to generate microstructure features: {e}")
            feature_status['microstructure'] = f'failed: {e}'
            
        # Path features
        try:
            logger.info("  - Path features")
            path_features = PathFeatures()
            daily_df = path_features.generate(minute_df, daily_df)
            feature_status['path'] = 'success'
        except Exception as e:
            logger.warning(f"Failed to generate path features: {e}")
            feature_status['path'] = f'failed: {e}'
            
        # Log feature generation status
        logger.info(f"Feature generation status: {feature_status}")
        
        # Count total features
        feature_cols = [c for c in daily_df.columns 
                       if c.startswith(('vr_', 'iex_', 'path_', 'liq_', 'macro_', 'time_', 'event_'))]
        logger.info(f"Generated {len(feature_cols)} features successfully")
        
        # Track feature statistics
        if self.config.tracking.backend == 'mlflow':
            try:
                mlflow.log_metric("n_features", len(feature_cols))
                mlflow.log_dict(feature_status, "feature_generation_status.json")
            except:
                pass
            
        return daily_df
    
    def _train_models_robust(self, X: np.ndarray, y: np.ndarray, 
                            dates: pd.Series, feature_names: list) -> tuple:
        """Train ensemble with multiple model types."""
        logger.info("Training robust ensemble")
        
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
        dates_test = dates[test_mask]
        
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Initialize base models
        base_models = {}
        
        # LightGBM
        if 'lightgbm' in self.config.ensemble.classification_bases:
            try:
                base_models['lightgbm'] = LightGBMClassifier(
                    class_weight='balanced',
                    sample_weight_power=self.config.imbalance.sample_weight_power,
                    use_gpu=self.config.compute.use_gpu
                )
                logger.info("Added LightGBM to ensemble")
            except Exception as e:
                logger.warning(f"Failed to initialize LightGBM: {e}")
                
        # XGBoost
        if 'xgboost' in self.config.ensemble.classification_bases:
            try:
                base_models['xgboost'] = XGBoostClassifier(
                    class_weight='balanced',
                    sample_weight_power=self.config.imbalance.sample_weight_power,
                    use_gpu=self.config.compute.use_gpu
                )
                logger.info("Added XGBoost to ensemble")
            except Exception as e:
                logger.warning(f"Failed to initialize XGBoost: {e}")
                
        # Random Forest (using sklearn)
        if 'random_forest' in self.config.ensemble.classification_bases:
            try:
                from sklearn.ensemble import RandomForestClassifier
                base_models['random_forest'] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                logger.info("Added Random Forest to ensemble")
            except Exception as e:
                logger.warning(f"Failed to initialize Random Forest: {e}")
                
        if len(base_models) == 0:
            raise ValueError("No base models could be initialized")
            
        logger.info(f"Initialized {len(base_models)} base models")
        
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
        
        # Create stacking ensemble
        ensemble = StackingMetaClassifier(
            base_models=base_models,
            meta_model=self.config.ensemble.meta.classification,
            use_proba=True,
            use_features=False,
            cv_splitter=cv_splitter,
            utility_matrix=utility_matrix
        )
        
        # Fit ensemble
        logger.info("Fitting ensemble on training data")
        ensemble.fit(X_train, y_train, dates=dates_train)
        
        # Evaluate on test set if available
        evaluation_results = {}
        
        if len(X_test) > 0:
            logger.info("Evaluating on test set")
            
            # Get predictions
            test_proba = ensemble.predict_proba(X_test)
            test_pred = ensemble.predict(X_test)
            
            # Run comprehensive evaluation
            evaluator = ClassificationEvaluator(
                utility_matrix=utility_matrix,
                class_names=['crash', 'normal', 'boom']
            )
            
            evaluation_results = evaluator.evaluate(
                y_true=y_test,
                y_pred_proba=test_proba,
                y_pred=test_pred
            )
            
            # Log key metrics
            logger.info(f"Test Results:")
            logger.info(f"  AUCPR (Crash): {evaluation_results['summary']['aucpr_crash']:.3f}")
            logger.info(f"  AUCPR (Boom): {evaluation_results['summary']['aucpr_boom']:.3f}")
            logger.info(f"  Expected Utility: {evaluation_results['summary']['expected_utility']:.3f}")
            logger.info(f"  Accuracy: {evaluation_results['summary']['accuracy']:.1%}")
            
            if self.config.tracking.backend == 'mlflow':
                try:
                    for key, value in evaluation_results['summary'].items():
                        mlflow.log_metric(f"test_{key}", value)
                except:
                    pass
                    
        return {'ensemble': ensemble}, evaluation_results
    
    def _optimize_decision_policy(self, models: dict, X: np.ndarray, 
                                 y: np.ndarray, dates: pd.Series) -> DecisionPolicy:
        """Optimize decision thresholds for utility maximization."""
        logger.info("Optimizing decision policy")
        
        # Get validation set for threshold optimization
        val_start = pd.Timestamp('2024-06-01')
        val_mask = (dates >= val_start) & (dates < pd.Timestamp(self.config.cv.test_start))
        
        if val_mask.sum() < 100:
            logger.warning("Insufficient validation data, using training data for optimization")
            val_mask = dates < pd.Timestamp(self.config.cv.test_start)
            
        X_val = X[val_mask]
        y_val = y[val_mask]
        
        # Get predictions
        ensemble = models['ensemble']
        val_proba = ensemble.predict_proba(X_val)
        
        # Create utility matrix
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
        
        # Initialize and optimize policy
        policy = DecisionPolicy(
            utility_matrix=utility_matrix,
            mode=self.config.decision.mode
        )
        
        # Optimize thresholds
        optimization_results = policy.optimize_thresholds(
            probabilities=val_proba,
            y_true=y_val,
            method='grid',
            n_trials=self.config.decision.threshold_search.n_trials
        )
        
        logger.info(f"Optimal thresholds found:")
        logger.info(f"  τ_crash: {optimization_results['tau_crash']:.3f}")
        logger.info(f"  τ_boom: {optimization_results['tau_boom']:.3f}")
        logger.info(f"  Expected utility: {optimization_results['expected_utility']:.3f}")
        
        # Save policy
        policy_path = self.output_dir / 'policy.json'
        policy.save_policy(str(policy_path))
        
        return policy
    
    def _generate_report(self, models: dict, evaluation_results: dict,
                        features_df: pl.DataFrame, policy: DecisionPolicy):
        """Generate comprehensive evaluation report."""
        logger.info("Generating evaluation report")
        
        # Get feature importance if available
        feature_importance = pd.DataFrame()
        if 'ensemble' in models:
            try:
                feature_importance = models['ensemble'].get_feature_importance()
            except:
                logger.warning(f"MLflow initialization failed: {e}")
                self.config.tracking.backend = 'none'
            
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(
            json.dumps(OmegaConf.to_container(self.config)).encode()
        ).hexdigest()[:8]
        return f"{timestamp}_{config_hash}"
    
    def run(self):
        """Execute complete training pipeline with robust error handling."""
        logger.info(f"Starting robust training run: {self.run_id}")
        
        try:
            # 1. Load and validate data with robust handling
            logger.info("=" * 50)
            logger.info("STEP 1: Loading data with robust NA handling")
            minute_df, data_quality = self._load_data_robust()
            
            # 2. Create labels
            logger.info("=" * 50)
            logger.info("STEP 2: Creating labels")
            daily_df = self._create_labels(minute_df)
            
            # 3. Generate features with NA awareness
            logger.info("=" * 50)
            logger.info("STEP 3: Generating features (NA-aware)")
            features_df = self._generate_features_robust(minute_df, daily_df)
            
            # 4. Prepare for modeling
            logger.info("=" * 50)
            logger.info("STEP 4: Preparing data for modeling")
            X, y, dates, feature_names = self._prepare_modeling_data(features_df)
            
            # 5. Train models
            logger.info("=" * 50)
            logger.info("STEP 5: Training ensemble models")
            models, evaluation_results = self._train_models_robust(X, y, dates, feature_names)
            
            # 6. Optimize decision policy
            logger.info("=" * 50)
            logger.info("STEP 6: Optimizing decision policy")
            policy = self._optimize_decision_policy(models, X, y, dates)
            
            # 7. Generate comprehensive report
            logger.info("=" * 50)
            logger.info("STEP 7: Generating evaluation report")
            self._generate_report(models, evaluation_results, features_df, policy)
            
            # 8. Save outputs
            logger.info("=" * 50)
            logger.info("STEP 8: Saving all outputs")
            self._save_outputs(models, features_df, policy)
            
            logger.info(f"Training completed successfully: {self.run_id}")
            logger.info(f"Outputs saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
        finally:
            if self.config.tracking.backend == 'mlflow':
                try:
                    mlflow.end_run()
                except:
                    pass
    
    def _load_data_robust(self) -> tuple:
        """Load data with robust NA handling."""
        logger.info(f"Loading data from {self.config.data.master_parquet}")
        
        # Use robust loader
        from src.vrisk.io.load_master_robust import RobustMasterDataLoader
        
        loader = RobustMasterDataLoader(
            self.config.data.master_parquet,
            use_parquet=True,
            validate=True,
            handle_missing='fill',
            min_data_fraction=0.7
        )
        
        df = loader.load()
        
        # Get data quality report
        data_quality = loader.missing_columns_report
        self.missing_data_report = data_quality
        
        # Add calendar features
        calendar = TradingCalendar()
        df = calendar.add_session_features(df)
        
        # Log data info
        logger.info(f"Loaded {len(df):,} minute records")
        logger.info(f"Date range: {df['session_date'].min()} to {df['session_date'].max()}")
        
        if data_quality.get('overall_missing_pct', 0) > 0:
            logger.warning(f"Overall missing data: {data_quality.get('overall_missing_pct', 0):.2f}%")
            
        # Get usable date range based on critical columns
        critical_cols = ['ohlcv_open', 'ohlcv_close', 'ohlcv_volume']
        start_date, end_date = loader.get_usable_date_range(critical_cols)
        logger.info(f"Usable date range for critical features: {start_date} to {end_date}")
        
        # Track data hash
        data_hash = loader._data_hash
        
        if self.config.tracking.backend == 'mlflow':
            try:
                mlflow.log_metric("n_minutes", len(df))
                mlflow.log_param("data_hash", data_hash)
                mlflow.log_metric("missing_data_pct", data_quality.get('overall_missing_pct', 0))
            except:
                pass
            
        return df, data_quality
    
    def _generate_features_robust(self, minute_df: pl.DataFrame, daily_df: pl.DataFrame) -> pl.DataFrame:
        """Generate features with proper NA handling."""
        logger.info("Generating features with NA awareness")
        
        # Track which features are generated
        feature_status = {}
        
        # Volatility features (should always work)
        try:
            logger.info("  - Volatility features")
            vol_features = VolatilityFeatures()
            daily_df = vol_features.generate(daily_df)
            feature_status['volatility'] = 'success'
        except Exception as e:
            logger.warning(f"