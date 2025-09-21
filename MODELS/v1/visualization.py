"""
Visualization module for model results and monitoring
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v1\\visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from config import *

class ModelVisualizer:
    def __init__(self, predictions_df, targets_df):
        """
        Initialize visualizer with predictions and actual targets
        """
        self.predictions_df = predictions_df
        self.targets_df = targets_df
        
        # Set style
        plt.style.use(PLOT_STYLE)
        sns.set_palette("husl")
        
    def plot_crash_boom_predictions(self, save_path=None):
        """
        Plot crash/boom predictions vs actual
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # 1. Time series of predictions
        ax1 = fig.add_subplot(gs[0, :])
        
        # Map labels to numeric values for plotting
        label_map = {'crash': -1, 'normal': 0, 'boom': 1}
        actual_numeric = self.predictions_df['actual_label'].map(label_map)
        predicted_numeric = self.predictions_df['predicted_label'].map(label_map)
        
        ax1.plot(self.predictions_df.index, actual_numeric, 
                label='Actual', alpha=0.7, linewidth=1)
        ax1.scatter(self.predictions_df.index, predicted_numeric, 
                   c='red', s=10, alpha=0.5, label='Predicted')
        
        ax1.set_ylabel('Crash (-1) / Normal (0) / Boom (1)')
        ax1.set_title('Crash/Boom Predictions Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Probability distributions
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist([self.predictions_df['crash_prob'], 
                 self.predictions_df['normal_prob'],
                 self.predictions_df['boom_prob']], 
                label=['Crash', 'Normal', 'Boom'], 
                alpha=0.6, bins=30)
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.legend()
        
        # 3. Confusion matrix
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        labels = ['crash', 'normal', 'boom']
        cm = confusion_matrix(self.predictions_df['actual_label'], 
                            self.predictions_df['predicted_label'], 
                            labels=labels)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax3.imshow(cm_normalized, cmap='Blues', aspect='auto')
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels)
        ax3.set_yticklabels(labels)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix (Normalized)')
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax3.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})',
                              ha="center", va="center", color="black" if cm_normalized[i, j] < 0.5 else "white")
        
        plt.colorbar(im, ax=ax3)
        
        # 4. Accuracy over time
        ax4 = fig.add_subplot(gs[2, :])
        
        # Calculate rolling accuracy
        window = 50
        rolling_accuracy = pd.Series(
            (self.predictions_df['actual_label'] == self.predictions_df['predicted_label']).astype(int)
        ).rolling(window=window, min_periods=10).mean()
        
        ax4.plot(self.predictions_df.index, rolling_accuracy, label=f'{window}-day Rolling Accuracy')
        ax4.axhline(y=rolling_accuracy.mean(), color='r', linestyle='--', 
                   label=f'Mean: {rolling_accuracy.mean():.2%}')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Model Accuracy Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
        
    def plot_volatility_predictions(self, save_path=None):
        """
        Plot volatility predictions vs actual
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # 1. Actual vs Predicted Log RV
        ax1 = axes[0]
        ax1.plot(self.predictions_df.index, self.predictions_df['actual_log_rv'], 
                label='Actual Log RV', alpha=0.7)
        ax1.plot(self.predictions_df.index, self.predictions_df['predicted_log_rv'], 
                label='Predicted Log RV', alpha=0.7)
        ax1.set_ylabel('Log Realized Variance')
        ax1.set_title('Volatility Predictions: Log Realized Variance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Prediction Error
        ax2 = axes[1]
        error = self.predictions_df['predicted_log_rv'] - self.predictions_df['actual_log_rv']
        ax2.plot(self.predictions_df.index, error, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.fill_between(self.predictions_df.index, 0, error, alpha=0.3)
        ax2.set_ylabel('Prediction Error')
        ax2.set_title(f'Volatility Prediction Error (MAE: {np.abs(error).mean():.4f})')
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot with R²
        ax3 = axes[2]
        ax3.scatter(self.predictions_df['actual_log_rv'], 
                   self.predictions_df['predicted_log_rv'], 
                   alpha=0.5, s=10)
        
        # Add diagonal line
        min_val = min(self.predictions_df['actual_log_rv'].min(), 
                     self.predictions_df['predicted_log_rv'].min())
        max_val = max(self.predictions_df['actual_log_rv'].max(), 
                     self.predictions_df['predicted_log_rv'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(self.predictions_df['actual_log_rv'], 
                     self.predictions_df['predicted_log_rv'])
        
        ax3.set_xlabel('Actual Log RV')
        ax3.set_ylabel('Predicted Log RV')
        ax3.set_title(f'Actual vs Predicted (R² = {r2:.3f})')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
        
    def plot_performance_dashboard(self, save_path=None):
        """
        Create comprehensive performance dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 3, figure=fig)
        
        # 1. Overall accuracy metrics
        ax1 = fig.add_subplot(gs[0, :])
        
        # Calculate metrics
        crash_acc = (self.predictions_df[self.predictions_df['actual_label'] == 'crash']['predicted_label'] == 'crash').mean()
        boom_acc = (self.predictions_df[self.predictions_df['actual_label'] == 'boom']['predicted_label'] == 'boom').mean()
        normal_acc = (self.predictions_df[self.predictions_df['actual_label'] == 'normal']['predicted_label'] == 'normal').mean()
        overall_acc = (self.predictions_df['actual_label'] == self.predictions_df['predicted_label']).mean()
        
        metrics = ['Overall', 'Crash', 'Boom', 'Normal']
        accuracies = [overall_acc, crash_acc, boom_acc, normal_acc]
        
        bars = ax1.bar(metrics, accuracies, color=['blue', 'red', 'green', 'gray'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy by Category')
        ax1.set_ylim([0, 1])
        
        # Add percentage labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1%}', ha='center', va='bottom')
        
        # 2. Probability calibration
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Bin predictions and calculate actual frequencies
        n_bins = 10
        for label, prob_col, color in [('Crash', 'crash_prob', 'red'), 
                                       ('Boom', 'boom_prob', 'green')]:
            prob_bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
            
            actual_freq = []
            predicted_freq = []
            
            for i in range(n_bins):
                mask = (self.predictions_df[prob_col] >= prob_bins[i]) & \
                      (self.predictions_df[prob_col] < prob_bins[i+1])
                if mask.sum() > 0:
                    actual = (self.predictions_df[mask]['actual_label'] == label.lower()).mean()
                    predicted = self.predictions_df[mask][prob_col].mean()
                    actual_freq.append(actual)
                    predicted_freq.append(predicted)
                else:
                    actual_freq.append(0)
                    predicted_freq.append(bin_centers[i])
            
            ax2.plot(predicted_freq, actual_freq, 'o-', label=label, color=color, alpha=0.7)
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Actual Frequency')
        ax2.set_title('Probability Calibration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature importance (if available)
        ax3 = fig.add_subplot(gs[1, 1:])
        ax3.text(0.5, 0.5, 'Feature Importance\n(Updated during training)', 
                ha='center', va='center', fontsize=12)
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        # 4. Training size over time
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(self.predictions_df.index, self.predictions_df['training_size'])
        ax4.set_xlabel('Prediction Index')
        ax4.set_ylabel('Training Samples Used')
        ax4.set_title('Expanding Window: Training Set Size')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance by market condition
        ax5 = fig.add_subplot(gs[3, 0])
        
        # Calculate performance in different volatility regimes
        vol_percentiles = self.predictions_df['actual_log_rv'].quantile([0.33, 0.67])
        
        low_vol_mask = self.predictions_df['actual_log_rv'] <= vol_percentiles.iloc[0]
        mid_vol_mask = (self.predictions_df['actual_log_rv'] > vol_percentiles.iloc[0]) & \
                      (self.predictions_df['actual_log_rv'] <= vol_percentiles.iloc[1])
        high_vol_mask = self.predictions_df['actual_log_rv'] > vol_percentiles.iloc[1]
        
        regime_acc = [
            (self.predictions_df[low_vol_mask]['actual_label'] == 
             self.predictions_df[low_vol_mask]['predicted_label']).mean(),
            (self.predictions_df[mid_vol_mask]['actual_label'] == 
             self.predictions_df[mid_vol_mask]['predicted_label']).mean(),
            (self.predictions_df[high_vol_mask]['actual_label'] == 
             self.predictions_df[high_vol_mask]['predicted_label']).mean()
        ]
        
        ax5.bar(['Low Vol', 'Mid Vol', 'High Vol'], regime_acc, 
               color=['lightblue', 'blue', 'darkblue'])
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Accuracy by Volatility Regime')
        ax5.set_ylim([0, 1])
        
        # 6. Cumulative accuracy
        ax6 = fig.add_subplot(gs[3, 1:])
        
        cumulative_correct = (self.predictions_df['actual_label'] == 
                             self.predictions_df['predicted_label']).cumsum()
        cumulative_total = np.arange(1, len(self.predictions_df) + 1)
        cumulative_accuracy = cumulative_correct / cumulative_total
        
        ax6.plot(self.predictions_df.index, cumulative_accuracy)
        ax6.axhline(y=cumulative_accuracy.iloc[-1], color='r', linestyle='--', 
                   label=f'Final: {cumulative_accuracy.iloc[-1]:.2%}')
        ax6.set_xlabel('Sample Index')
        ax6.set_ylabel('Cumulative Accuracy')
        ax6.set_title('Cumulative Model Accuracy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
        
    def create_monitoring_table(self):
        """
        Create a monitoring table with key metrics
        """
        # Calculate key metrics
        metrics = {
            'Total Predictions': len(self.predictions_df),
            'Overall Accuracy': f"{(self.predictions_df['actual_label'] == self.predictions_df['predicted_label']).mean():.2%}",
            'Crash Detection Rate': f"{(self.predictions_df[self.predictions_df['actual_label'] == 'crash']['predicted_label'] == 'crash').mean():.2%}",
            'Boom Detection Rate': f"{(self.predictions_df[self.predictions_df['actual_label'] == 'boom']['predicted_label'] == 'boom').mean():.2%}",
            'Volatility MAE': f"{np.abs(self.predictions_df['predicted_log_rv'] - self.predictions_df['actual_log_rv']).mean():.4f}",
            'Volatility R²': f"{r2_score(self.predictions_df['actual_log_rv'], self.predictions_df['predicted_log_rv']):.3f}",
            'Avg Training Size': f"{self.predictions_df['training_size'].mean():.0f}",
            'Date Range': f"{self.predictions_df['date'].min()} to {self.predictions_df['date'].max()}" if 'date' in self.predictions_df else "N/A"
        }
        
        # Create DataFrame
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        
        print("\n" + "="*50)
        print("MODEL MONITORING TABLE")
        print("="*50)
        print(metrics_df.to_string(index=False))
        print("="*50)
        
        return metrics_df

# Import r2_score for the module
from sklearn.metrics import r2_score

# Test visualization
if __name__ == "__main__":
    print("Run main.py to generate visualizations")