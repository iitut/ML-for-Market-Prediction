"""
Decision policy layer with threshold optimization.
Implements Standard and Forced-Extremes modes with utility maximization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any
from scipy.optimize import minimize_scalar, differential_evolution
import optuna
import logging

logger = logging.getLogger(__name__)


class DecisionPolicy:
    """
    Decision policy for crash/boom/normal classification.
    Supports threshold optimization and forced extremes mode.
    """
    
    def __init__(self,
                 utility_matrix: np.ndarray,
                 mode: str = 'forced',
                 tau_crash: float = 0.33,
                 tau_boom: float = 0.33):
        """
        Initialize decision policy.
        
        Args:
            utility_matrix: 3x3 utility/cost matrix
            mode: 'standard' or 'forced' extremes
            tau_crash: Initial crash threshold
            tau_boom: Initial boom threshold
        """
        self.utility_matrix = utility_matrix
        self.mode = mode
        self.tau_crash = tau_crash
        self.tau_boom = tau_boom
        self.optimization_history = []
        
    def decide(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Make decisions based on probabilities and thresholds.
        
        Args:
            probabilities: Class probabilities (n_samples, 3)
            
        Returns:
            Decisions (0=crash, 1=normal, 2=boom)
        """
        n_samples = probabilities.shape[0]
        decisions = np.ones(n_samples, dtype=int)  # Default to normal
        
        if self.mode == 'standard':
            # Standard mode: use thresholds directly
            decisions = self._standard_decision(probabilities)
            
        elif self.mode == 'forced':
            # Forced extremes: if no threshold met, pick max(crash, boom)
            decisions = self._forced_decision(probabilities)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return decisions
    
    def _standard_decision(self, probabilities: np.ndarray) -> np.ndarray:
        """Standard decision mode with thresholds."""
        decisions = np.ones(len(probabilities), dtype=int)  # Default normal
        
        # Apply thresholds
        crash_mask = probabilities[:, 0] >= self.tau_crash
        boom_mask = probabilities[:, 2] >= self.tau_boom
        
        # Assign decisions
        decisions[crash_mask] = 0  # Crash
        decisions[boom_mask] = 2   # Boom
        
        # Handle conflicts (both thresholds met)
        conflict_mask = crash_mask & boom_mask
        if conflict_mask.any():
            # Choose higher probability
            decisions[conflict_mask] = np.where(
                probabilities[conflict_mask, 0] > probabilities[conflict_mask, 2],
                0, 2
            )
            
        return decisions
    
    def _forced_decision(self, probabilities: np.ndarray) -> np.ndarray:
        """Forced extremes mode - always pick crash or boom if thresholds not met."""
        decisions = self._standard_decision(probabilities)
        
        # Find samples that were classified as normal
        normal_mask = decisions == 1
        
        if normal_mask.any():
            # For normal predictions, force to extreme with higher probability
            crash_probs = probabilities[normal_mask, 0]
            boom_probs = probabilities[normal_mask, 2]
            
            # Pick crash or boom based on which has higher probability
            forced_decisions = np.where(crash_probs > boom_probs, 0, 2)
            decisions[normal_mask] = forced_decisions
            
        return decisions
    
    def optimize_thresholds(self,
                           probabilities: np.ndarray,
                           y_true: np.ndarray,
                           method: str = 'grid',
                           n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize thresholds to maximize expected utility.
        
        Args:
            probabilities: Predicted probabilities
            y_true: True labels
            method: Optimization method ('grid', 'bayesian', 'differential_evolution')
            n_trials: Number of trials for optimization
            
        Returns:
            Optimization results
        """
        logger.info(f"Optimizing thresholds using {method} method")
        
        if method == 'grid':
            results = self._grid_search(probabilities, y_true, n_trials)
            
        elif method == 'bayesian':
            results = self._bayesian_optimization(probabilities, y_true, n_trials)
            
        elif method == 'differential_evolution':
            results = self._differential_evolution(probabilities, y_true)
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
        # Update thresholds
        self.tau_crash = results['tau_crash']
        self.tau_boom = results['tau_boom']
        
        logger.info(f"Optimal thresholds: tau_crash={self.tau_crash:.3f}, "
                   f"tau_boom={self.tau_boom:.3f}")
        logger.info(f"Expected utility: {results['expected_utility']:.4f}")
        
        return results
    
    def _calculate_utility(self,
                          decisions: np.ndarray,
                          y_true: np.ndarray) -> float:
        """Calculate expected utility for decisions."""
        n_samples = len(y_true)
        total_utility = 0
        
        for i in range(n_samples):
            true_class = int(y_true[i])
            pred_class = int(decisions[i])
            utility = self.utility_matrix[pred_class, true_class]
            total_utility += utility
            
        return total_utility / n_samples
    
    def _grid_search(self,
                    probabilities: np.ndarray,
                    y_true: np.ndarray,
                    n_points: int) -> Dict[str, Any]:
        """Grid search for optimal thresholds."""
        # Create grid
        tau_range = np.linspace(0.2, 0.8, n_points)
        
        best_utility = -np.inf
        best_params = None
        results_grid = []
        
        for tau_c in tau_range:
            for tau_b in tau_range:
                # Set thresholds temporarily
                self.tau_crash = tau_c
                self.tau_boom = tau_b
                
                # Make decisions
                decisions = self.decide(probabilities)
                
                # Calculate utility
                utility = self._calculate_utility(decisions, y_true)
                
                results_grid.append({
                    'tau_crash': tau_c,
                    'tau_boom': tau_b,
                    'utility': utility
                })
                
                if utility > best_utility:
                    best_utility = utility
                    best_params = {'tau_crash': tau_c, 'tau_boom': tau_b}
                    
        return {
            **best_params,
            'expected_utility': best_utility,
            'grid_results': pd.DataFrame(results_grid),
            'method': 'grid'
        }
    
    def _bayesian_optimization(self,
                              probabilities: np.ndarray,
                              y_true: np.ndarray,
                              n_trials: int) -> Dict[str, Any]:
        """Bayesian optimization using Optuna."""
        
        def objective(trial):
            # Sample thresholds
            tau_c = trial.suggest_float('tau_crash', 0.15, 0.85)
            tau_b = trial.suggest_float('tau_boom', 0.15, 0.85)
            
            # Set thresholds
            self.tau_crash = tau_c
            self.tau_boom = tau_b
            
            # Make decisions
            decisions = self.decide(probabilities)
            
            # Calculate utility (negative for minimization)
            utility = self._calculate_utility(decisions, y_true)
            
            return -utility  # Minimize negative utility
        
        # Run optimization
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Get best parameters
        best_params = study.best_params
        best_utility = -study.best_value
        
        return {
            'tau_crash': best_params['tau_crash'],
            'tau_boom': best_params['tau_boom'],
            'expected_utility': best_utility,
            'study': study,
            'method': 'bayesian'
        }
    
    def _differential_evolution(self,
                              probabilities: np.ndarray,
                              y_true: np.ndarray) -> Dict[str, Any]:
        """Differential evolution optimization."""
        
        def objective(params):
            tau_c, tau_b = params
            self.tau_crash = tau_c
            self.tau_boom = tau_b
            
            decisions = self.decide(probabilities)
            utility = self._calculate_utility(decisions, y_true)
            
            return -utility  # Minimize negative utility
        
        # Run optimization
        bounds = [(0.15, 0.85), (0.15, 0.85)]
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15
        )
        
        return {
            'tau_crash': result.x[0],
            'tau_boom': result.x[1],
            'expected_utility': -result.fun,
            'convergence': result.success,
            'method': 'differential_evolution'
        }
    
    def create_decision_audit(self,
                            probabilities: np.ndarray,
                            y_true: np.ndarray,
                            metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create detailed audit table of decisions.
        
        Args:
            probabilities: Predicted probabilities
            y_true: True labels
            metadata: Additional metadata (dates, returns, etc.)
            
        Returns:
            DataFrame with decision audit information
        """
        decisions = self.decide(probabilities)
        
        audit_df = pd.DataFrame({
            'p_crash': probabilities[:, 0],
            'p_normal': probabilities[:, 1],
            'p_boom': probabilities[:, 2],
            'decision': decisions,
            'decision_label': pd.Categorical.from_codes(
                decisions, categories=['crash', 'normal', 'boom']
            ),
            'true_label': pd.Categorical.from_codes(
                y_true, categories=['crash', 'normal', 'boom']
            ),
            'correct': decisions == y_true,
            'tau_crash': self.tau_crash,
            'tau_boom': self.tau_boom,
            'mode': self.mode
        })
        
        # Calculate utility contribution
        utility_contrib = []
        for i in range(len(y_true)):
            true_class = int(y_true[i])
            pred_class = int(decisions[i])
            utility = self.utility_matrix[pred_class, true_class]
            utility_contrib.append(utility)
            
        audit_df['utility_contribution'] = utility_contrib
        
        # Add metadata if provided
        if metadata is not None:
            for col in metadata.columns:
                audit_df[col] = metadata[col].values
                
        return audit_df
    
    def save_policy(self, path: str):
        """Save policy parameters to JSON."""
        import json
        
        policy_dict = {
            'mode': self.mode,
            'tau_crash': float(self.tau_crash),
            'tau_boom': float(self.tau_boom),
            'utility_matrix': self.utility_matrix.tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(policy_dict, f, indent=2)
            
        logger.info(f"Policy saved to {path}")
        
    @classmethod
    def load_policy(cls, path: str) -> 'DecisionPolicy':
        """Load policy from JSON."""
        import json
        
        with open(path, 'r') as f:
            policy_dict = json.load(f)
            
        return cls(
            utility_matrix=np.array(policy_dict['utility_matrix']),
            mode=policy_dict['mode'],
            tau_crash=policy_dict['tau_crash'],
            tau_boom=policy_dict['tau_boom']
        )