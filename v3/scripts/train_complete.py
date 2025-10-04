#!/usr/bin/env python
"""
Complete training script for VOL-RISK LAB v3 with all components.
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
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from src.vrisk.io.load_master_robust import load_master_minute_robust
from src.vrisk.calendars.sessions import TradingCalendar
from src.vrisk.labeling.minute_returns import calculate_minute_returns
from src.vrisk.labeling.rv_daily import calculate_daily_rv
from src.vrisk.labeling.crash_boom_labels import create_crash_boom_labels
from src.vrisk.io.load_master_robust import RobustMasterDataLoader

# Import all feature modules
from src.vrisk.features.vol_rv_bv_jv import VolatilityFeatures
from src.vrisk.features.microstructure_iex import MicrostructureFeatures
from src.vrisk.features.path_gaps import PathFeatures
from src.vrisk.features.liquidity import LiquidityFeatures
from src.vrisk.features.macro_time import MacroTimeFeatures
from src.vrisk.features.regime_memory import RegimeMemoryFeatures
from src.vrisk.features.event_flags import EventFlagFeatures

# Import all models
from src.vrisk.models.lgbm_cls import LightGBMClassifier
from src.vrisk.models.xgb_cls import XGBoostClassifier
from src.vrisk.models.catboost_cls import CatBoostClassifier
from src.vrisk.models.volatility_head import VolatilityEnsemble
from src.vrisk.models.quantile_head import QuantilePredictor, ConformalPredictor

# Import training and evaluation
from src.vrisk.training.cv_splitter import TimeSeriesCVSplitter
from src.vrisk.ensembles.stacking_meta_cls_fixed import StackingMetaClassifier
from src.vrisk.utils.thresholding import DecisionPolicy
from src.vrisk.evaluation.eval_classify import ClassificationEvaluator
from src.vrisk.reporting.writer import ReportWriter

logger = logging.getLogger(__name__)


class CompleteModelTrainer:
    """Complete training orchestrator with all VOL-RISK LAB components."""
    
    def __init__(self, config: DictConfig):
        """Initialize complete trainer."""
        self.config = config
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output.base_path) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track component status
        self.component_status = {
            'data_loading': 'pending',
            'labeling': 'pending',
            'features': {},
            'models': {},
            'evaluation': 'pending'
        }

    def _cfg_get(self, dotted: str, default=None):
        """
        Safe getter that supports both 'cv.test_start' and 'test_start'
        depending on how configs are packaged.
        """
        try:
            node = self.config
            for part in dotted.split('.'):
                node = node[part] if isinstance(node, dict) else getattr(node, part)
            return node
        except Exception:
            root_key = dotted.split('.')[-1]
            try:
                return self.config[root_key] if isinstance(self.config, dict) else getattr(self.config, root_key)
            except Exception:
                return default
        
    def run(self):
        """Execute complete training pipeline."""
        logger.info(f"Starting COMPLETE VOL-RISK LAB training: {self.run_id}")
        
        try:
            # 1. Load data with robust handling
            logger.info("="*60)
            logger.info("STEP 1: Loading and validating data")
            minute_df, data_quality = self._load_data()
            self.component_status['data_loading'] = 'complete'
            
            # 2. Create all labels
            logger.info("="*60)
            logger.info("STEP 2: Creating labels (classification + volatility + quantiles)")
            minute_df, daily_df = self._create_all_labels(minute_df)
            self.component_status['labeling'] = 'complete'
            
            # 3. Generate ALL features
            logger.info("="*60)
            logger.info("STEP 3: Generating complete feature set")
            features_df = self._generate_all_features(minute_df, daily_df)
            
            # 4. Prepare modeling data
            logger.info("="*60)
            logger.info("STEP 4: Preparing data for all prediction heads")
            X, y_cls, y_vol, y_ret, dates, feature_names = self._prepare_all_data(features_df)
            
            # 5. Train ALL models
            logger.info("="*60)
            logger.info("STEP 5: Training all model components")
            models = self._train_all_models(X, y_cls, y_vol, y_ret, dates)
            
            # 6. Evaluate everything
            logger.info("="*60)
            logger.info("STEP 6: Comprehensive evaluation")
            evaluation_results = self._evaluate_all(models, X, y_cls, y_vol, y_ret, dates)
            
            # 7. Optimize decision policy
            logger.info("="*60)
            logger.info("STEP 7: Optimizing decision policy")
            policy = self._optimize_policy(models, X, y_cls, dates)
            
            # 8. Generate complete report
            logger.info("="*60)
            logger.info("STEP 8: Generating comprehensive report")
            self._generate_complete_report(models, evaluation_results, features_df, policy)
            
            # 9. Save everything
            logger.info("="*60)
            logger.info("STEP 9: Saving all outputs")
            self._save_all_outputs(models, features_df, policy)
            
            # 10. Print summary
            self._print_summary(evaluation_results)
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Outputs saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            self._save_error_report(str(e))
            raise
    
    def _load_data(self) -> tuple:
        """Load data with complete validation."""
        loader = RobustMasterDataLoader(
            self.config.data.master_parquet,
            use_parquet=True,
            validate=True,
            handle_missing='fill'
        )

        df = loader.load()
        data_quality = loader.missing_columns_report

        # Add calendar features (this adds 'session_date')
        calendar = TradingCalendar()
        df = calendar.add_session_features(df)

        # --- DEBUG: keep only the last N contiguous sessions ---
        dbg = self.config.get('debug', {})
        if dbg and dbg.get('enabled', False):
            n = int(dbg.get('last_n_sessions', 20))
            last_sessions = (
                df.select(pl.col("session_date").unique())
                .to_series()
                .sort()
                .tail(n)
                .to_list()
            )
            df = df.filter(pl.col("session_date").is_in(last_sessions))
            logger.info(
                f"[DEBUG] Using last {n} sessions: "
                f"{df['session_date'].min()} to {df['session_date'].max()} "
                f"({len(last_sessions)} sessions, {len(df):,} minute rows)"
            )

        logger.info(f"Loaded {len(df):,} minute records")
        logger.info(f"Date range: {df['session_date'].min()} to {df['session_date'].max()}")

        return df, data_quality

    
    def _create_all_labels(self, minute_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create all types of labels and return (minute_df_with_returns, daily_df_enriched)."""

        # 1) Minute returns on the minute frame (needed by path/liquidity features later)
        minute_df = calculate_minute_returns(minute_df)  # adds 'minute_return'

        # 2) Daily realized-volatility measures on top of minute_df
        rv_df = calculate_daily_rv(minute_df)

        # 3) Crash/boom labels on daily grain
        daily_df = create_crash_boom_labels(
            minute_df,
            rv_df,
            gammas=self.config.targets.gammas,
            volatility_estimator='RV'
        )

        # 4) Add volatility & return targets (next-day / 5-day)
        daily_df = daily_df.with_columns([
            pl.col('log_RV_daily').shift(-1).alias('target_log_rv_1d'),
            pl.col('daily_return').shift(-1).alias('target_return_1d'),
            pl.col('log_RV_daily').shift(-5).alias('target_log_rv_5d'),
            pl.col('daily_return').shift(-5).rolling_sum(window_size=5).alias('target_return_5d')
        ])

        # 5) Bring session-level flags from minute_df (unique by session_date)
        flags = [
            'is_holiday', 'is_early_close', 'early_close_minutes_local',
            'is_weekly', 'is_monthly_third_friday', 'is_quarterly_eoq', 'is_opx'
        ]
        have_flags = [c for c in flags if c in minute_df.columns]
        if have_flags:
            if 'session_date' not in minute_df.columns:
                raise ValueError("minute_df is missing 'session_date' after calendar augmentation")

            flag_df = (
                minute_df.select(['session_date'] + have_flags)
                         .unique(subset=['session_date'], keep='first')
            )
            daily_df = daily_df.join(flag_df, on='session_date', how='left')

        # 6) Precompute 'event_days_to_opx' if we have 'is_opx'
        if 'is_opx' in daily_df.columns:
            df_pd = daily_df.select(['session_date', 'is_opx']).to_pandas()
            df_pd['session_date'] = pd.to_datetime(df_pd['session_date']).dt.date  # FIX: Convert to date
            df_pd = df_pd.sort_values('session_date').set_index('session_date')

            opx_dates = df_pd.index[df_pd['is_opx'].fillna(False).astype(bool)]
            if len(opx_dates) > 0:
                s_next = pd.Series(opx_dates.values, index=opx_dates)
                next_opx = s_next.reindex(df_pd.index, method='bfill')
                days_to_next = (pd.to_datetime(next_opx.values) - pd.to_datetime(df_pd.index.values)).days.astype('float')
                df_pd['event_days_to_opx'] = days_to_next
            else:
                df_pd['event_days_to_opx'] = np.nan

            tmp = pl.from_pandas(df_pd[['event_days_to_opx']].reset_index())

            # FIX: Ensure session_date is pl.Date type to match daily_df
            if 'session_date' not in tmp.columns:
                for cand in ('index', 'level_0'):
                    if cand in tmp.columns:
                        tmp = tmp.rename({cand: 'session_date'})
                        break
            
            # Convert to pl.Date to match daily_df
            tmp = tmp.with_columns(
                pl.col('session_date').cast(pl.Date)
            )

            daily_df = daily_df.join(tmp, on='session_date', how='left')

        return minute_df, daily_df
    
    def _generate_all_features(self, minute_df: pl.DataFrame, daily_df: pl.DataFrame) -> pl.DataFrame:
        """Generate ALL feature categories."""
        
        logger.info("Generating complete feature set...")
        
        # 1. Volatility features
        try:
            vol_features = VolatilityFeatures()
            daily_df = vol_features.generate(daily_df)
            self.component_status['features']['volatility'] = 'success'
        except Exception as e:
            logger.error(f"Volatility features failed: {e}")
            self.component_status['features']['volatility'] = f'failed: {e}'
            
        # 2. Microstructure features
        try:
            micro_features = MicrostructureFeatures()
            micro_df = micro_features.generate(minute_df)
            daily_df = daily_df.join(micro_df, on='session_date', how='left')
            self.component_status['features']['microstructure'] = 'success'
        except Exception as e:
            logger.error(f"Microstructure features failed: {e}")
            self.component_status['features']['microstructure'] = f'failed: {e}'
            
        # 3. Path features
        try:
            path_features = PathFeatures()
            daily_df = path_features.generate(minute_df, daily_df)
            self.component_status['features']['path'] = 'success'
        except Exception as e:
            logger.error(f"Path features failed: {e}")
            self.component_status['features']['path'] = f'failed: {e}'
            
        # 4. Liquidity features  
        try:
            liq_features = LiquidityFeatures()
            daily_df = liq_features.generate(minute_df, daily_df)
            self.component_status['features']['liquidity'] = 'success'
        except Exception as e:
            logger.error(f"Liquidity features failed: {e}")
            self.component_status['features']['liquidity'] = f'failed: {e}'
            
        # 5. Macro and time features
        try:
            macro_time = MacroTimeFeatures()
            daily_df = macro_time.generate(daily_df)
            self.component_status['features']['macro_time'] = 'success'
        except Exception as e:
            logger.error(f"Macro/time features failed: {e}")
            self.component_status['features']['macro_time'] = f'failed: {e}'
            
        # 6. Regime and memory features
        try:
            regime_features = RegimeMemoryFeatures()
            daily_df = regime_features.generate(daily_df)
            self.component_status['features']['regime'] = 'success'
        except Exception as e:
            logger.error(f"Regime features failed: {e}")
            self.component_status['features']['regime'] = f'failed: {e}'
            
        # 7. Event flags
        try:
            event_features = EventFlagFeatures()
            daily_df = event_features.generate(daily_df)
            self.component_status['features']['events'] = 'success'
        except Exception as e:
            logger.error(f"Event features failed: {e}")
            self.component_status['features']['events'] = f'failed: {e}'
            
        # Count features
        feature_cols = [c for c in daily_df.columns 
                       if c.startswith(('vr_', 'iex_', 'path_', 'liq_', 'macro_', 
                                      'time_', 'reg_', 'event_'))]
        
        logger.info(f"Generated {len(feature_cols)} total features")
        logger.info(f"Feature generation status: {self.component_status['features']}")
        
        return daily_df
    
    def _prepare_all_data(self, features_df: pl.DataFrame) -> tuple:
        """Prepare data for all prediction heads."""
        
        # Select features
        feature_cols = [c for c in features_df.columns 
                    if c.startswith(('vr_', 'iex_', 'path_', 'liq_', 'macro_', 
                                    'time_', 'reg_', 'event_'))]
        
        # Remove rows with missing targets
        gamma = self.config.targets.default_gamma
        label_col = f'label_gamma_{gamma:.1f}_numeric'
        
        clean_df = features_df.filter(
            pl.col(label_col).is_not_null() &
            pl.col('target_log_rv_1d').is_not_null() &
            pl.col('target_return_1d').is_not_null()
        )
        
        # Handle remaining NaNs in features - FIXED VERSION
        for col in feature_cols:
            if col not in clean_df.columns:
                continue
                
            null_count = clean_df[col].null_count()
            if null_count > 0:
                # Get column dtype
                dtype = clean_df[col].dtype
                
                # Handle based on dtype
                if dtype == pl.Boolean:
                    # For boolean columns, fill with False
                    clean_df = clean_df.with_columns(
                        pl.col(col).fill_null(False)
                    )
                elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                    # For integer columns, fill with 0
                    clean_df = clean_df.with_columns(
                        pl.col(col).fill_null(0)
                    )
                else:
                    # For float columns, use median
                    median_val = clean_df[col].median()
                    fill_val = median_val if median_val is not None else 0.0
                    clean_df = clean_df.with_columns(
                        pl.col(col).fill_null(fill_val)
                    )
                    
        # Convert to numpy
        X = clean_df.select(feature_cols).to_numpy()
        y_cls = clean_df[label_col].to_numpy()
        y_vol = clean_df['target_log_rv_1d'].to_numpy()
        y_ret = clean_df['target_return_1d'].to_numpy()
        dates = clean_df['session_date'].to_pandas()
        
        # Final NaN check
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Prepared data shapes:")
        logger.info(f"  X: {X.shape}")
        logger.info(f"  Classification: {y_cls.shape}")
        logger.info(f"  Volatility: {y_vol.shape}")
        logger.info(f"  Returns: {y_ret.shape}")
        
        return X, y_cls, y_vol, y_ret, dates, feature_cols
    
    def _train_all_models(self, X, y_cls, y_vol, y_ret, dates) -> dict:
        """Train all model components."""
        
        models = {}
        
        # Split data (support cv.* or root keys)
        test_start_str = self._cfg_get('cv.test_start', '2025-01-01')
        test_start = pd.Timestamp(test_start_str)
        train_mask = dates < test_start
        test_mask = dates >= test_start
        
        X_train, y_cls_train = X[train_mask], y_cls[train_mask]
        X_test, y_cls_test = X[test_mask], y_cls[test_mask]
        y_vol_train, y_vol_test = y_vol[train_mask], y_vol[test_mask]
        y_ret_train, y_ret_test = y_ret[train_mask], y_ret[test_mask]
        dates_train, dates_test = dates[train_mask], dates[test_mask]
        
        # Create CV splitter
        cv_splitter = TimeSeriesCVSplitter(
            n_splits=int(self._cfg_get('cv.n_splits', 8)),
            min_train_days=int(self._cfg_get('cv.min_train_days', 252)),
            test_days=int(self._cfg_get('cv.test_days', 63)),
            embargo_days=int(self._cfg_get('cv.embargo_days', 1)),
            purge_days=int(self._cfg_get('cv.purge_days', 1)),
            anchored=bool(self._cfg_get('cv.anchored', True)),
        )
        
        # 1. Train Classification Ensemble
        logger.info("Training classification ensemble...")
        
        base_classifiers = {}
        
        # LightGBM
        if 'lightgbm' in self.config.ensemble.classification_bases:
            try:
                base_classifiers['lightgbm'] = LightGBMClassifier(
                    class_weight='balanced',
                    sample_weight_power=self.config.imbalance.sample_weight_power
                )
                self.component_status['models']['lgbm_cls'] = 'success'
            except Exception as e:
                logger.error(f"LightGBM failed: {e}")
                self.component_status['models']['lgbm_cls'] = f'failed: {e}'
                
        # XGBoost
        if 'xgboost' in self.config.ensemble.classification_bases:
            try:
                base_classifiers['xgboost'] = XGBoostClassifier(
                    class_weight='balanced',
                    sample_weight_power=self.config.imbalance.sample_weight_power
                )
                self.component_status['models']['xgb_cls'] = 'success'
            except Exception as e:
                logger.error(f"XGBoost failed: {e}")
                self.component_status['models']['xgb_cls'] = f'failed: {e}'
                
        # CatBoost
        if 'catboost' in self.config.ensemble.classification_bases:
            try:
                base_classifiers['catboost'] = CatBoostClassifier(
                    class_weight='balanced',
                    sample_weight_power=self.config.imbalance.sample_weight_power
                )
                self.component_status['models']['catboost_cls'] = 'success'
            except Exception as e:
                logger.error(f"CatBoost failed: {e}")
                self.component_status['models']['catboost_cls'] = f'failed: {e}'
                
        # Random Forest
        if 'random_forest' in self.config.ensemble.classification_bases:
            try:
                from sklearn.ensemble import RandomForestClassifier
                base_classifiers['random_forest'] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                self.component_status['models']['rf_cls'] = 'success'
            except Exception as e:
                logger.error(f"Random Forest failed: {e}")
                self.component_status['models']['rf_cls'] = f'failed: {e}'
                
        # SVM
        if 'svm_rbf' in self.config.ensemble.classification_bases:
            try:
                from sklearn.svm import SVC
                base_classifiers['svm'] = SVC(
                    kernel='rbf',
                    probability=True,
                    class_weight='balanced',
                    random_state=42
                )
                self.component_status['models']['svm_cls'] = 'success'
            except Exception as e:
                logger.error(f"SVM failed: {e}")
                self.component_status['models']['svm_cls'] = f'failed: {e}'
                
        # Elastic Net Logistic
        if 'elastic_net' in self.config.ensemble.classification_bases:
            try:
                from sklearn.linear_model import LogisticRegression
                base_classifiers['elastic_net'] = LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    l1_ratio=0.5,
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                )
                self.component_status['models']['elastic_cls'] = 'success'
            except Exception as e:
                logger.error(f"Elastic Net failed: {e}")
                self.component_status['models']['elastic_cls'] = f'failed: {e}'
                
        # Create stacking ensemble
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
        
        classification_ensemble = StackingMetaClassifier(
            base_models=base_classifiers,
            meta_model=self.config.ensemble.meta.classification,
            use_proba=True,
            use_features=False,
            cv_splitter=cv_splitter,
            utility_matrix=utility_matrix
        )
        
        classification_ensemble.fit(X_train, y_cls_train, dates=dates_train)
        models['classification'] = classification_ensemble
        
        # 2. Train Volatility Ensemble
        logger.info("Training volatility ensemble...")
        
        try:
            volatility_ensemble = VolatilityEnsemble(
                base_models=self.config.ensemble.volatility_bases,
                meta_model=self.config.ensemble.meta.volatility,
                use_har_baseline=True
            )
            volatility_ensemble.fit(X_train, y_vol_train)
            models['volatility'] = volatility_ensemble
            self.component_status['models']['volatility'] = 'success'
        except Exception as e:
            logger.error(f"Volatility ensemble failed: {e}")
            self.component_status['models']['volatility'] = f'failed: {e}'
            
        # 3. Train Quantile Predictors
        logger.info("Training quantile predictors...")
        
        try:
            quantile_predictor = QuantilePredictor(
                quantiles=self.config.targets.quantiles,
                base_model='lightgbm',
                enforce_monotonicity=True
            )
            quantile_predictor.fit(X_train, y_ret_train)
            models['quantile'] = quantile_predictor
            
            # Add conformal wrapper
            if len(X_test) > 100:
                conformal = ConformalPredictor(quantile_predictor, alpha=0.1)
                # Use part of test as calibration
                cal_size = min(500, len(X_test) // 2)
                conformal.calibrate(X_test[:cal_size], y_ret_test[:cal_size])
                models['conformal'] = conformal
                
            self.component_status['models']['quantile'] = 'success'
        except Exception as e:
            logger.error(f"Quantile predictor failed: {e}")
            self.component_status['models']['quantile'] = f'failed: {e}'
            
        logger.info(f"Model training status: {self.component_status['models']}")
        
        return models
    
    def _evaluate_all(self, models, X, y_cls, y_vol, y_ret, dates) -> dict:
        """Comprehensive evaluation of all models."""
        
        evaluation_results = {}
        
        # Split data for evaluation
        test_start = pd.Timestamp(self._cfg_get('cv.test_start', '2025-01-01'))
        test_mask = dates >= test_start
        
        if test_mask.sum() == 0:
            logger.warning("No test data available")
            return evaluation_results
            
        X_test = X[test_mask]
        y_cls_test = y_cls[test_mask]
        y_vol_test = y_vol[test_mask]
        y_ret_test = y_ret[test_mask]
        
        # 1. Evaluate Classification
        if 'classification' in models:
            logger.info("Evaluating classification...")
            
            cls_proba = models['classification'].predict_proba(X_test)
            cls_pred = models['classification'].predict(X_test)
            
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
            
            evaluator = ClassificationEvaluator(
                utility_matrix=utility_matrix,
                class_names=['crash', 'normal', 'boom']
            )
            
            evaluation_results['classification'] = evaluator.evaluate(
                y_true=y_cls_test,
                y_pred_proba=cls_proba,
                y_pred=cls_pred
            )
            
        # 2. Evaluate Volatility
        if 'volatility' in models:
            logger.info("Evaluating volatility...")
            
            vol_pred = models['volatility'].predict(X_test)
            
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            evaluation_results['volatility'] = {
                'r2': r2_score(y_vol_test, vol_pred),
                'mae': mean_absolute_error(y_vol_test, vol_pred),
                'rmse': np.sqrt(mean_squared_error(y_vol_test, vol_pred)),
                'predictions': vol_pred,
                'y_true': y_vol_test
            }
            
            logger.info(f"Volatility R²: {evaluation_results['volatility']['r2']:.3f}")
            
        # 3. Evaluate Quantiles
        if 'quantile' in models:
            logger.info("Evaluating quantiles...")
            
            qtl_pred = models['quantile'].predict(X_test)
            
            # Calculate coverage
            coverage = {}
            for q, pred in qtl_pred.items():
                empirical_q = (y_ret_test <= pred).mean()
                coverage[q] = empirical_q
                
            evaluation_results['quantiles'] = {
                'coverage': {
                    'nominal': list(qtl_pred.keys()),
                    'empirical': list(coverage.values())
                },
                'predictions': qtl_pred
            }
            
            logger.info(f"Quantile coverage: {coverage}")
            
        return evaluation_results
    
    def _optimize_policy(self, models, X, y_cls, dates):
        """Optimize decision policy."""
        
        if 'classification' not in models:
            logger.warning("No classification model to optimize")
            return None
            
        # Get validation data
        val_start = pd.Timestamp('2024-06-01')
        val_end = pd.Timestamp(self._cfg_get('cv.test_start', '2025-01-01'))
        val_mask = (dates >= val_start) & (dates < val_end)
        
        if val_mask.sum() < 100:
            logger.warning("Insufficient validation data")
            val_mask = dates < pd.Timestamp(self._cfg_get('cv.test_start', '2025-01-01'))
            
        X_val = X[val_mask]
        y_val = y_cls[val_mask]
        
        # Get predictions
        val_proba = models['classification'].predict_proba(X_val)
        
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
        
        # Optimize policy
        policy = DecisionPolicy(
            utility_matrix=utility_matrix,
            mode=self.config.decision.mode
        )
        
        _ = policy.optimize_thresholds(
            probabilities=val_proba,
            y_true=y_val,
            method='grid',
            n_trials=self.config.decision.threshold_search.n_trials
        )
        
        logger.info(f"Optimized policy: τ_crash={policy.tau_crash:.3f}, τ_boom={policy.tau_boom:.3f}")
        
        return policy
    
    def _generate_complete_report(self, models, evaluation_results, features_df, policy):
        """Generate comprehensive report."""
        
        # Get feature importance if available
        feature_importance = pd.DataFrame()
        if 'classification' in models:
            try:
                feature_importance = models['classification'].get_feature_importance()
            except:
                pass
                
        # Create report writer
        writer = ReportWriter(
            output_dir=str(self.output_dir),
            run_id=self.run_id,
            config=OmegaConf.to_container(self.config)
        )
        
        # Generate report
        report_path = writer.generate_report(
            evaluation_results=evaluation_results,
            model_info={
                'component_status': self.component_status,
                'n_features': len([c for c in features_df.columns 
                                  if c.startswith(('vr_', 'iex_', 'path_', 'liq_', 
                                                'macro_', 'time_', 'reg_', 'event_'))])
            },
            feature_importance=feature_importance
        )
        
        logger.info(f"Report generated: {report_path}")
        
    def _save_all_outputs(self, models, features_df, policy):
        """Save all outputs."""
        
        # Save features
        features_df.write_parquet(self.output_dir / 'features.parquet')
        
        # Save configuration
        with open(self.output_dir / 'config.yaml', 'w') as f:
            OmegaConf.save(self.config, f)
            
        # Save component status
        with open(self.output_dir / 'component_status.json', 'w') as f:
            json.dump(self.component_status, f, indent=2, default=str)
            
        # Save policy if exists
        if policy is not None:
            policy.save_policy(str(self.output_dir / 'policy.json'))
            
        # Save metadata
        metadata = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'component_status': self.component_status,
            'data_shape': features_df.shape
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"All outputs saved to {self.output_dir}")
        
    def _print_summary(self, evaluation_results):
        """Print training summary."""
        
        print("\n" + "="*60)
        print("VOL-RISK LAB v3 - TRAINING SUMMARY")
        print("="*60)
        
        print(f"\nRun ID: {self.run_id}")
        print(f"Output Directory: {self.output_dir}")
        
        print("\nComponent Status:")
        for component, status in self.component_status.items():
            if isinstance(status, dict):
                print(f"  {component}:")
                for sub_comp, sub_status in status.items():
                    status_emoji = "✅" if sub_status == 'success' else "❌"
                    print(f"    {status_emoji} {sub_comp}: {sub_status}")
            else:
                status_emoji = "✅" if status == 'complete' else "❌"
                print(f"  {status_emoji} {component}: {status}")
                
        if 'classification' in evaluation_results:
            print("\nClassification Results:")
            summary = evaluation_results['classification']['summary']
            print(f"  AUCPR (Crash): {summary['aucpr_crash']:.3f}")
            print(f"  AUCPR (Boom): {summary['aucpr_boom']:.3f}")
            print(f"  Expected Utility: {summary['expected_utility']:.3f}")
            print(f"  Accuracy: {summary['accuracy']:.1%}")
            
        if 'volatility' in evaluation_results:
            print("\nVolatility Results:")
            print(f"  R²: {evaluation_results['volatility']['r2']:.3f}")
            print(f"  MAE: {evaluation_results['volatility']['mae']:.4f}")
            print(f"  RMSE: {evaluation_results['volatility']['rmse']:.4f}")
            
        print("\n" + "="*60)
        
    def _save_error_report(self, error_msg: str):
        """Save error report if training fails."""
        error_report = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'error': error_msg,
            'component_status': self.component_status
        }
        
        with open(self.output_dir / 'error_report.json', 'w') as f:
            json.dump(error_report, f, indent=2)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    # Override data paths if needed
    if 'paths' in cfg:
        cfg.data.master_parquet = cfg.paths.data.master_parquet
        
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Run complete training
    trainer = CompleteModelTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
