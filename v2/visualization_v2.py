"""
Enhanced visualization module for Model v2
Path: C:\\Users\\shyto\\Downloads\\Closing-Auction Variance Measures\\ML_MP\\MODELS\\v2\\visualization_v2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from config import *

class EnhancedVisualizer:
    def __init__(self, predictions_df, targets_df):
        """Initialize enhanced visualizer"""
        self.predictions_df = predictions_df
        self.targets_df = targets_df
        
        plt.style.use(PLOT_STYLE)
        sns.set_palette("husl")
        
    def plot_7class_performance(self, save_path=None):
        """Plot 7-class market direction performance"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Time series of 7-class predictions
        ax1 = fig.add_subplot(gs[0, :])
        
        if 'actual_market_direction_7class' in self.predictions_df.columns:
            # Map classes to numeric values
            class_map = {
                'extreme_crash': -3, 'crash': -2, 'mild_down': -1,
                'normal': 0, 'mild_up': 1, 'boom': 2, 'extreme_boom': 3
            }
            
            actual = self.predictions_df['actual_market_direction_7class'].map(class_map)
            
            # Create color map for actual values
            colors = []
            for val in self.predictions_df['actual_market_direction_7class']:
                colors.append(COLORS.get(val, 'gray'))
            
            ax1.scatter(self.predictions_df.index, actual, c=colors, alpha=0.6, s=20)
            ax1.set_ylabel('Market Direction')
            ax1.set_title('7-Class Market Direction Over Time')
            ax1.set_yticks(list(class_map.values()))
            ax1.set_yticklabels(list(class_map.keys()))
            ax1.grid(True, alpha=0.3)
        
        # 2. Confusion matrix for 7 classes
        ax2 = fig.add_subplot(gs[1, :2])
        
        if 'actual_market_direction_7class' in self.predictions_df.columns and \
           'pred_market_direction_7class' in self.predictions_df.columns:
            
            from sklearn.metrics import confusion_matrix
            
            classes = ['extreme_crash', 'crash', 'mild_down', 'normal', 'mild_up', 'boom', 'extreme_boom']
            cm = confusion_matrix(
                self.predictions_df['actual_market_direction_7class'],
                self.predictions_df['pred_market_direction_7class'],
                labels=classes
            )
            
            # Normalize
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=classes, yticklabels=classes, ax=ax2)
            ax2.set_title('7-Class Confusion Matrix (Normalized)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
        
        # 3. Class distribution
        ax3 = fig.add_subplot(gs[1, 2])
        
        if 'actual_market_direction_7class' in self.predictions_df.columns:
            class_counts = self.predictions_df['actual_market_direction_7class'].value_counts()
            
            bars = ax3.bar(range(len(class_counts)), class_counts.values)
            ax3.set_xticks(range(len(class_counts)))
            ax3.set_xticklabels(class_counts.index, rotation=45, ha='right')
            ax3.set_title('Actual Class Distribution')
            ax3.set_ylabel('Count')
            
            # Color bars
            for i, (label, bar) in enumerate(zip(class_counts.index, bars)):
                bar.set_color(COLORS.get(label, 'gray'))
        
        # 4. Return predictions scatter
        ax4 = fig.add_subplot(gs[2, 0])
        
        if 'actual_next_day_return' in self.predictions_df.columns and \
           'pred_next_day_return' in self.predictions_df.columns:
            
            ax4.scatter(self.predictions_df['actual_next_day_return'] * 100,
                       self.predictions_df['pred_next_day_return'] * 100,
                       alpha=0.5, s=10)
            
            # Add diagonal line
            lims = [
                np.min([ax4.get_xlim(), ax4.get_ylim()]),
                np.max([ax4.get_xlim(), ax4.get_ylim()])
            ]
            ax4.plot(lims, lims, 'r--', alpha=0.5)
            
            # Calculate R²
            from sklearn.metrics import r2_score
            r2 = r2_score(self.predictions_df['actual_next_day_return'].dropna(),
                         self.predictions_df['pred_next_day_return'].dropna())
            
            ax4.set_xlabel('Actual Next Day Return (%)')
            ax4.set_ylabel('Predicted Next Day Return (%)')
            ax4.set_title(f'Next Day Return Predictions (R² = {r2:.3f})')
            ax4.grid(True, alpha=0.3)
        
        # 5. Weekly return predictions
        ax5 = fig.add_subplot(gs[2, 1])
        
        if 'actual_next_week_return' in self.predictions_df.columns and \
           'pred_next_week_return' in self.predictions_df.columns:
            
            ax5.scatter(self.predictions_df['actual_next_week_return'] * 100,
                       self.predictions_df['pred_next_week_return'] * 100,
                       alpha=0.5, s=10, color='green')
            
            lims = [
                np.min([ax5.get_xlim(), ax5.get_ylim()]),
                np.max([ax5.get_xlim(), ax5.get_ylim()])
            ]
            ax5.plot(lims, lims, 'r--', alpha=0.5)
            
            r2_week = r2_score(self.predictions_df['actual_next_week_return'].dropna(),
                             self.predictions_df['pred_next_week_return'].dropna())
            
            ax5.set_xlabel('Actual Weekly Return (%)')
            ax5.set_ylabel('Predicted Weekly Return (%)')
            ax5.set_title(f'Weekly Return Predictions (R² = {r2_week:.3f})')
            ax5.grid(True, alpha=0.3)
        
        # 6. Monthly return predictions
        ax6 = fig.add_subplot(gs[2, 2])
        
        if 'actual_next_month_return' in self.predictions_df.columns and \
           'pred_next_month_return' in self.predictions_df.columns:
            
            ax6.scatter(self.predictions_df['actual_next_month_return'] * 100,
                       self.predictions_df['pred_next_month_return'] * 100,
                       alpha=0.5, s=10, color='purple')
            
            lims = [
                np.min([ax6.get_xlim(), ax6.get_ylim()]),
                np.max([ax6.get_xlim(), ax6.get_ylim()])
            ]
            ax6.plot(lims, lims, 'r--', alpha=0.5)
            
            r2_month = r2_score(self.predictions_df['actual_next_month_return'].dropna(),
                              self.predictions_df['pred_next_month_return'].dropna())
            
            ax6.set_xlabel('Actual Monthly Return (%)')
            ax6.set_ylabel('Predicted Monthly Return (%)')
            ax6.set_title(f'Monthly Return Predictions (R² = {r2_month:.3f})')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Enhanced Model v2 - Multi-Timeframe Performance', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_probability_analysis(self, save_path=None):
        """Plot probability predictions and calibration"""
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # 1. Crash probability time series
        ax1 = fig.add_subplot(gs[0, :])
        
        if 'pred_crash_probability_1d' in self.predictions_df.columns:
            ax1.plot(self.predictions_df.index, 
                    self.predictions_df['pred_crash_probability_1d'] * 100,
                    label='1-Day Crash Probability', color='red', alpha=0.7)
        
        if 'pred_crash_probability_1w' in self.predictions_df.columns:
            ax1.plot(self.predictions_df.index,
                    self.predictions_df['pred_crash_probability_1w'] * 100,
                    label='1-Week Crash Probability', color='darkred', alpha=0.7)
        
        if 'pred_boom_probability_1d' in self.predictions_df.columns:
            ax1.plot(self.predictions_df.index,
                    self.predictions_df['pred_boom_probability_1d'] * 100,
                    label='1-Day Boom Probability', color='green', alpha=0.7)
        
        if 'pred_boom_probability_1w' in self.predictions_df.columns:
            ax1.plot(self.predictions_df.index,
                    self.predictions_df['pred_boom_probability_1w'] * 100,
                    label='1-Week Boom Probability', color='darkgreen', alpha=0.7)
        
        ax1.set_ylabel('Probability (%)')
        ax1.set_title('Crash/Boom Probabilities Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Daily crash probability calibration
        ax2 = fig.add_subplot(gs[1, 0])
        
        if 'actual_crash_probability_1d' in self.predictions_df.columns and \
           'pred_crash_probability_1d' in self.predictions_df.columns:
            
            self._plot_calibration(
                self.predictions_df['pred_crash_probability_1d'],
                self.predictions_df['actual_crash_probability_1d'],
                ax2, 'Daily Crash Probability Calibration'
            )
        
        # 3. Daily boom probability calibration
        ax3 = fig.add_subplot(gs[1, 1])
        
        if 'actual_boom_probability_1d' in self.predictions_df.columns and \
           'pred_boom_probability_1d' in self.predictions_df.columns:
            
            self._plot_calibration(
                self.predictions_df['pred_boom_probability_1d'],
                self.predictions_df['actual_boom_probability_1d'],
                ax3, 'Daily Boom Probability Calibration'
            )
        
        # 4. Probability distribution
        ax4 = fig.add_subplot(gs[1, 2])
        
        prob_cols = [col for col in self.predictions_df.columns if 'pred_' in col and 'probability' in col]
        if prob_cols:
            for col in prob_cols:
                label = col.replace('pred_', '').replace('_', ' ').title()
                ax4.hist(self.predictions_df[col], bins=30, alpha=0.5, label=label)
            
            ax4.set_xlabel('Probability')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Predicted Probabilities')
            ax4.legend()
        
        plt.suptitle('Probability Analysis and Calibration', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def _plot_calibration(self, predicted, actual, ax, title):
        """Helper function to plot calibration curve"""
        # Create calibration bins
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        actual_freq = []
        predicted_freq = []
        
        for i in range(n_bins):
            mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i+1])
            if mask.sum() > 0:
                actual_freq.append(actual[mask].mean())
                predicted_freq.append(predicted[mask].mean())
            else:
                actual_freq.append(0)
                predicted_freq.append(bin_centers[i])
        
        ax.plot(predicted_freq, actual_freq, 'o-', color='blue', alpha=0.7)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_comprehensive_dashboard(self, save_path=None):
        """Create comprehensive dashboard with all metrics"""
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig)
        
        # Title section
        fig.text(0.5, 0.98, 'Model v2 - Comprehensive Performance Dashboard', 
                fontsize=20, ha='center', weight='bold')
        
        # 1. Overall accuracy by timeframe
        ax1 = fig.add_subplot(gs[0, :2])
        
        timeframes = ['Day', 'Week', 'Month']
        accuracies = []
        
        for tf, col in zip(timeframes, ['next_day_return', 'next_week_return', 'next_month_return']):
            if f'actual_{col}' in self.predictions_df.columns and f'pred_{col}' in self.predictions_df.columns:
                actual = self.predictions_df[f'actual_{col}']
                pred = self.predictions_df[f'pred_{col}']
                
                # Calculate directional accuracy
                acc = ((actual > 0) == (pred > 0)).mean()
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        bars = ax1.bar(timeframes, accuracies, color=['blue', 'green', 'purple'])
        ax1.set_ylabel('Directional Accuracy')
        ax1.set_title('Prediction Accuracy by Timeframe')
        ax1.set_ylim([0, 1])
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1%}', ha='center', va='bottom')
        
        # 2. Returns distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        
        if 'actual_next_day_return' in self.predictions_df.columns:
            returns_data = [
                self.predictions_df['actual_next_day_return'] * 100,
            ]
            labels = ['Actual']
            
            if 'pred_next_day_return' in self.predictions_df.columns:
                returns_data.append(self.predictions_df['pred_next_day_return'] * 100)
                labels.append('Predicted')
            
            ax2.violinplot(returns_data, positions=range(len(returns_data)))
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels)
            ax2.set_ylabel('Daily Return (%)')
            ax2.set_title('Return Distribution Comparison')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Cumulative returns
        ax3 = fig.add_subplot(gs[1, :])
        
        if 'actual_next_day_return' in self.predictions_df.columns:
            cum_actual = (1 + self.predictions_df['actual_next_day_return']).cumprod()
            ax3.plot(self.predictions_df.index, (cum_actual - 1) * 100, 
                    label='Actual Cumulative Return', linewidth=2)
            
            if 'pred_next_day_return' in self.predictions_df.columns:
                cum_pred = (1 + self.predictions_df['pred_next_day_return']).cumprod()
                ax3.plot(self.predictions_df.index, (cum_pred - 1) * 100,
                        label='Predicted Cumulative Return', linewidth=2, alpha=0.7)
            
            ax3.set_ylabel('Cumulative Return (%)')
            ax3.set_title('Cumulative Returns Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Error analysis
        ax4 = fig.add_subplot(gs[2, 0])
        
        if 'actual_next_day_return' in self.predictions_df.columns and \
           'pred_next_day_return' in self.predictions_df.columns:
            
            error = (self.predictions_df['pred_next_day_return'] - 
                    self.predictions_df['actual_next_day_return']) * 100
            
            ax4.hist(error, bins=50, alpha=0.7, color='red', edgecolor='black')
            ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Prediction Error (%)')
            ax4.set_ylabel('Frequency')
            ax4.set_title(f'Error Distribution (Mean: {error.mean():.3f}%)')
        
        # 5. Model performance over time
        ax5 = fig.add_subplot(gs[2, 1:])
        
        if 'training_size' in self.predictions_df.columns:
            ax5_twin = ax5.twinx()
            
            # Plot training size
            ax5.plot(self.predictions_df.index, self.predictions_df['training_size'],
                    color='gray', alpha=0.5, label='Training Size')
            ax5.set_ylabel('Training Size', color='gray')
            
            # Calculate rolling accuracy
            if 'actual_market_direction_7class' in self.predictions_df.columns and \
               'pred_market_direction_7class' in self.predictions_df.columns:
                
                rolling_acc = (self.predictions_df['actual_market_direction_7class'] == 
                             self.predictions_df['pred_market_direction_7class']).rolling(50).mean()
                
                ax5_twin.plot(self.predictions_df.index, rolling_acc,
                            color='blue', label='50-day Rolling Accuracy')
                ax5_twin.set_ylabel('Accuracy', color='blue')
            
            ax5.set_title('Model Performance Evolution')
            ax5.grid(True, alpha=0.3)
        
        # 6. Metrics summary table
        ax6 = fig.add_subplot(gs[3, :2])
        ax6.axis('tight')
        ax6.axis('off')
        
        # Calculate metrics
        metrics_data = []
        
        if 'actual_next_day_return' in self.predictions_df.columns and \
           'pred_next_day_return' in self.predictions_df.columns:
            
            mae = np.abs(self.predictions_df['actual_next_day_return'] - 
                        self.predictions_df['pred_next_day_return']).mean()
            rmse = np.sqrt(((self.predictions_df['actual_next_day_return'] - 
                           self.predictions_df['pred_next_day_return'])**2).mean())
            
            metrics_data.append(['Daily MAE', f'{mae*100:.3f}%'])
            metrics_data.append(['Daily RMSE', f'{rmse*100:.3f}%'])
        
        if metrics_data:
            table = ax6.table(cellText=metrics_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        
        # 7. Class balance
        ax7 = fig.add_subplot(gs[3, 2:])
        
        if 'actual_market_direction_7class' in self.predictions_df.columns:
            class_counts = self.predictions_df['actual_market_direction_7class'].value_counts()
            
            # Create pie chart
            colors_list = [COLORS.get(label, 'gray') for label in class_counts.index]
            ax7.pie(class_counts.values, labels=class_counts.index, colors=colors_list,
                   autopct='%1.1f%%', startangle=90)
            ax7.set_title('Actual Class Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()

# Standalone function to create all visualizations
def create_all_visualizations(predictions_df, targets_df, save_dir=None):
    """Create all visualization plots"""
    visualizer = EnhancedVisualizer(predictions_df, targets_df)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    # Create all plots
    print("Creating 7-class performance plot...")
    visualizer.plot_7class_performance(
        save_path=save_dir / "7class_performance.png" if save_dir else None
    )
    
    print("Creating probability analysis plot...")
    visualizer.plot_probability_analysis(
        save_path=save_dir / "probability_analysis.png" if save_dir else None
    )
    
    print("Creating comprehensive dashboard...")
    visualizer.plot_comprehensive_dashboard(
        save_path=save_dir / "comprehensive_dashboard.png" if save_dir else None
    )
    
    print("All visualizations created successfully!")