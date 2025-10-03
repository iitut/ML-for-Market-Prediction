"""
Comprehensive evaluation for classification head.
Includes AUCPR, utility metrics, and regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, roc_curve,
    confusion_matrix, classification_report, brier_score_loss,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """Comprehensive evaluation for crash/boom/normal classification."""
    
    def __init__(self,
                 utility_matrix: np.ndarray,
                 class_names: List[str] = ['crash', 'normal', 'boom']):
        """
        Initialize evaluator.
        
        Args:
            utility_matrix: 3x3 utility/cost matrix
            class_names: Names for the 3 classes
        """
        self.utility_matrix = utility_matrix
        self.class_names = class_names
        self.results = {}
        
    def evaluate(self,
                y_true: np.ndarray,
                y_pred_proba: np.ndarray,
                y_pred: Optional[np.ndarray] = None,
                metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            y_true: True labels (0, 1, 2)
            y_pred_proba: Predicted probabilities (n_samples, 3)
            y_pred: Predicted labels (optional)
            metadata: DataFrame with additional info (dates, regimes, etc.)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running classification evaluation")
        
        if y_pred is None:
            y_pred = np.argmax(y_pred_proba, axis=1)
            
        # Core metrics
        self.results['core'] = self._compute_core_metrics(y_true, y_pred, y_pred_proba)
        
        # AUCPR for each class
        self.results['aucpr'] = self._compute_aucpr(y_true, y_pred_proba)
        
        # Utility metrics
        self.results['utility'] = self._compute_utility(y_true, y_pred, y_pred_proba)
        
        # Calibration metrics
        self.results['calibration'] = self._compute_calibration(y_true, y_pred_proba)
        
        # Confusion matrix
        self.results['confusion'] = self._compute_confusion(y_true, y_pred)
        
        # Regime analysis if metadata provided
        if metadata is not None:
            self.results['regime'] = self._analyze_by_regime(
                y_true, y_pred, y_pred_proba, metadata
            )
            
        # Summary statistics
        self.results['summary'] = self._create_summary()
        
        return self.results
    
    def _compute_core_metrics(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_pred_proba: np.ndarray) -> Dict:
        """Compute core classification metrics."""
        metrics = {}
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            metrics[f'precision_{class_name}'] = precision_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
            metrics[f'recall_{class_name}'] = recall_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
            metrics[f'f1_{class_name}'] = f1_score(
                y_true_binary, y_pred_binary, zero_division=0
            )
            
        # Overall metrics
        metrics['accuracy'] = (y_pred == y_true).mean()
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Brier score (multi-class)
        brier_scores = []
        for i in range(3):
            y_true_binary = (y_true == i).astype(int)
            brier = brier_score_loss(y_true_binary, y_pred_proba[:, i])
            brier_scores.append(brier)
            metrics[f'brier_{self.class_names[i]}'] = brier
            
        metrics['brier_mean'] = np.mean(brier_scores)
        
        return metrics
    
    def _compute_aucpr(self, 
                      y_true: np.ndarray,
                      y_pred_proba: np.ndarray) -> Dict:
        """Compute AUCPR for each class."""
        aucpr_results = {}
        
        for i, class_name in enumerate(self.class_names):
            # Binary classification for this class
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_proba[:, i]
            
            # Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(
                y_true_binary, y_score
            )
            
            # AUCPR
            aucpr = auc(recall, precision)
            aucpr_results[f'aucpr_{class_name}'] = aucpr
            
            # Store curves for plotting
            aucpr_results[f'pr_curve_{class_name}'] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }
            
            # ROC-AUC as secondary metric
            try:
                roc_auc = roc_auc_score(y_true_binary, y_score)
                aucpr_results[f'roc_auc_{class_name}'] = roc_auc
                
                # ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_score)
                aucpr_results[f'roc_curve_{class_name}'] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': roc_thresholds
                }
            except:
                aucpr_results[f'roc_auc_{class_name}'] = np.nan
                
        # Focus on extreme classes
        aucpr_results['aucpr_extremes_mean'] = np.mean([
            aucpr_results['aucpr_crash'],
            aucpr_results['aucpr_boom']
        ])
        
        return aucpr_results
    
    def _compute_utility(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_pred_proba: np.ndarray) -> Dict:
        """Compute utility-based metrics."""
        utility_results = {}
        
        # Expected utility for predictions
        total_utility = 0
        n_samples = len(y_true)
        
        for i in range(n_samples):
            true_class = int(y_true[i])
            pred_class = int(y_pred[i])
            utility = self.utility_matrix[pred_class, true_class]
            total_utility += utility
            
        utility_results['total_utility'] = total_utility
        utility_results['expected_utility'] = total_utility / n_samples
        
        # Utility by class
        for true_idx, true_name in enumerate(self.class_names):
            mask = y_true == true_idx
            if mask.sum() > 0:
                class_utility = 0
                for pred_idx in range(3):
                    pred_mask = y_pred[mask] == pred_idx
                    count = pred_mask.sum()
                    utility = self.utility_matrix[pred_idx, true_idx]
                    class_utility += count * utility
                    
                utility_results[f'utility_{true_name}'] = class_utility / mask.sum()
                
        # Optimal utility (perfect predictions)
        optimal_utility = 0
        for i in range(n_samples):
            true_class = int(y_true[i])
            optimal_utility += self.utility_matrix[true_class, true_class]
            
        utility_results['optimal_utility'] = optimal_utility / n_samples
        utility_results['utility_ratio'] = (
            utility_results['expected_utility'] / 
            utility_results['optimal_utility']
        )
        
        # Utility decomposition
        cm = confusion_matrix(y_true, y_pred)
        utility_decomp = {}
        
        for i in range(3):
            for j in range(3):
                pred_name = self.class_names[i]
                true_name = self.class_names[j]
                count = cm[i, j]
                utility_contrib = count * self.utility_matrix[i, j] / n_samples
                utility_decomp[f'{pred_name}_as_{true_name}'] = {
                    'count': count,
                    'utility': self.utility_matrix[i, j],
                    'contribution': utility_contrib
                }
                
        utility_results['decomposition'] = utility_decomp
        
        return utility_results
    
    def _compute_calibration(self,
                           y_true: np.ndarray,
                           y_pred_proba: np.ndarray) -> Dict:
        """Compute calibration metrics."""
        calibration_results = {}
        
        # Expected Calibration Error (ECE)
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_pred_proba[:, i]
            
            # Calibration curve
            fraction_pos, mean_pred = calibration_curve(
                y_true_binary, y_score, n_bins=10, strategy='uniform'
            )
            
            # ECE
            ece = np.abs(fraction_pos - mean_pred).mean()
            calibration_results[f'ece_{class_name}'] = ece
            
            # Store for plotting
            calibration_results[f'calibration_curve_{class_name}'] = {
                'fraction_positive': fraction_pos,
                'mean_predicted': mean_pred
            }
            
        calibration_results['ece_mean'] = np.mean([
            calibration_results[f'ece_{name}'] for name in self.class_names
        ])
        
        return calibration_results
    
    def _compute_confusion(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict:
        """Compute confusion matrix and derived metrics."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize versions
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_normalized_all = cm.astype(float) / cm.sum()
        
        return {
            'matrix': cm,
            'matrix_normalized': cm_normalized,
            'matrix_normalized_all': cm_normalized_all,
            'class_names': self.class_names
        }
    
    def _analyze_by_regime(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_pred_proba: np.ndarray,
                          metadata: pd.DataFrame) -> Dict:
        """Analyze performance by market regime."""
        regime_results = {}
        
        # Analyze by VR quintile if available
        if 'vr_quintile' in metadata.columns:
            for quintile in metadata['vr_quintile'].unique():
                mask = metadata['vr_quintile'] == quintile
                
                if mask.sum() > 10:  # Need minimum samples
                    regime_results[f'vr_{quintile}'] = {
                        'n_samples': mask.sum(),
                        'accuracy': (y_pred[mask] == y_true[mask]).mean(),
                        'utility': self._compute_utility(
                            y_true[mask], y_pred[mask], y_pred_proba[mask]
                        )['expected_utility']
                    }
                    
        # Analyze by options expiration
        if 'is_opx' in metadata.columns:
            for is_opx in [True, False]:
                mask = metadata['is_opx'] == is_opx
                
                if mask.sum() > 10:
                    regime_results[f'opx_{is_opx}'] = {
                        'n_samples': mask.sum(),
                        'accuracy': (y_pred[mask] == y_true[mask]).mean()
                    }
                    
        return regime_results
    
    def _create_summary(self) -> Dict:
        """Create summary of key metrics."""
        summary = {
            'aucpr_crash': self.results['aucpr']['aucpr_crash'],
            'aucpr_boom': self.results['aucpr']['aucpr_boom'],
            'expected_utility': self.results['utility']['expected_utility'],
            'accuracy': self.results['core']['accuracy'],
            'macro_f1': self.results['core']['macro_f1'],
            'brier_mean': self.results['core']['brier_mean'],
            'ece_mean': self.results['calibration']['ece_mean']
        }
        
        return summary