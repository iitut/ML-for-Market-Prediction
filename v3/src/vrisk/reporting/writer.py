"""
Report generation module for VOL-RISK LAB.
Creates comprehensive HTML/PDF reports with all required visualizations.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


class ReportWriter:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self,
                 output_dir: str,
                 run_id: str,
                 config: Dict[str, Any]):
        """
        Initialize report writer.
        
        Args:
            output_dir: Directory for outputs
            run_id: Unique run identifier
            config: Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.config = config
        
        # Create directories
        self.reports_dir = self.output_dir / 'reports' / run_id
        self.assets_dir = self.reports_dir / 'assets'
        self.tables_dir = self.reports_dir / 'tables'
        
        for dir_path in [self.reports_dir, self.assets_dir, self.tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def generate_report(self,
                       evaluation_results: Dict[str, Any],
                       model_info: Dict[str, Any],
                       feature_importance: pd.DataFrame) -> str:
        """
        Generate complete HTML report.
        
        Args:
            evaluation_results: Evaluation metrics and results
            model_info: Model configuration and metadata
            feature_importance: Feature importance scores
            
        Returns:
            Path to generated HTML report
        """
        logger.info(f"Generating report for run {self.run_id}")
        
        # Generate all visualizations
        self._create_classification_plots(evaluation_results['classification'])
        self._create_volatility_plots(evaluation_results.get('volatility', {}))
        self._create_quantile_plots(evaluation_results.get('quantiles', {}))
        self._create_feature_plots(feature_importance)
        
        # Save tables
        self._save_tables(evaluation_results)
        
        # Generate HTML
        html_content = self._generate_html(
            evaluation_results,
            model_info,
            feature_importance
        )
        
        # Save HTML
        html_path = self.reports_dir / 'index.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Report saved to {html_path}")
        
        # Generate technical appendix
        self._generate_appendix(model_info, feature_importance)
        
        # Generate model card
        self._generate_model_card(evaluation_results, model_info)
        
        return str(html_path)
    
    def _create_classification_plots(self, results: Dict[str, Any]):
        """Create all classification visualizations."""
        
        # 1. PR Curves for Crash and Boom
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, class_name in enumerate(['crash', 'boom']):
            ax = axes[idx]
            pr_data = results['aucpr'][f'pr_curve_{class_name}']
            
            ax.plot(pr_data['recall'], pr_data['precision'], 
                   label=f"AUCPR = {results['aucpr'][f'aucpr_{class_name}']:.3f}")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve: {class_name.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.assets_dir / 'cls_pr_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = results['confusion']['matrix_normalized']
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=results['confusion']['class_names'],
                   yticklabels=results['confusion']['class_names'],
                   ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Normalized Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(self.assets_dir / 'cls_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Reliability Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, class_name in enumerate(['crash', 'normal', 'boom']):
            ax = axes[idx]
            cal_data = results['calibration'][f'calibration_curve_{class_name}']
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
            
            # Actual calibration
            ax.plot(cal_data['mean_predicted'], cal_data['fraction_positive'],
                   marker='o', label=f"ECE = {results['calibration'][f'ece_{class_name}']:.3f}")
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration: {class_name.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.assets_dir / 'cls_calibration.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Utility Heatmap
        self._create_utility_heatmap(results['utility'])
        
    def _create_utility_heatmap(self, utility_results: Dict[str, Any]):
        """Create utility optimization heatmap."""
        if 'grid_results' not in utility_results:
            return
            
        grid_df = utility_results['grid_results']
        
        # Pivot for heatmap
        pivot = grid_df.pivot(index='tau_crash', columns='tau_boom', values='utility')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(pivot, annot=False, cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Expected Utility'}, ax=ax)
        
        # Mark optimal point
        optimal_tau_c = utility_results['tau_crash']
        optimal_tau_b = utility_results['tau_boom']
        
        ax.scatter(optimal_tau_b, optimal_tau_c, marker='*', 
                  s=500, c='blue', label='Optimal')
        
        ax.set_xlabel('Boom Threshold (τ_B)')
        ax.set_ylabel('Crash Threshold (τ_C)')
        ax.set_title('Expected Utility Surface')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.assets_dir / 'cls_utility_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _create_volatility_plots(self, results: Dict[str, Any]):
        """Create volatility prediction visualizations."""
        if not results:
            return
            
        # 1. Predicted vs Realized Scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if 'predictions' in results:
            y_true = results['y_true']
            y_pred = results['predictions']
            
            ax.scatter(y_true, y_pred, alpha=0.5, s=10)
            ax.plot([y_true.min(), y_true.max()], 
                   [y_true.min(), y_true.max()], 
                   'r--', alpha=0.7, label='45° line')
            
            ax.set_xlabel('Realized log(RV)')
            ax.set_ylabel('Predicted log(RV)')
            ax.set_title(f"Volatility Predictions (R² = {results.get('r2', 0):.3f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.assets_dir / 'vol_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _create_quantile_plots(self, results: Dict[str, Any]):
        """Create quantile prediction visualizations."""
        if not results:
            return
            
        # Coverage plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'coverage' in results:
            quantiles = [0.05, 0.50, 0.95]
            empirical = results['coverage']['empirical']
            
            ax.bar(range(len(quantiles)), empirical, alpha=0.7, label='Empirical')
            ax.plot(range(len(quantiles)), quantiles, 'r--', marker='o', label='Nominal')
            
            ax.set_xticks(range(len(quantiles)))
            ax.set_xticklabels([f'{q:.0%}' for q in quantiles])
            ax.set_ylabel('Coverage')
            ax.set_title('Quantile Coverage Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.assets_dir / 'qtl_coverage.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _create_feature_plots(self, feature_importance: pd.DataFrame):
        """Create feature importance visualizations."""
        if feature_importance.empty:
            return
            
        # Top 20 features
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = feature_importance.head(20)
        
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Feature Importance')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.assets_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
    def generate_pdf_report(self, 
                        evaluation_results: Dict[str, Any],
                        model_info: Dict[str, Any]) -> str:
        """
        Generate PDF report with all visualizations.
        
        Args:
            evaluation_results: Evaluation metrics
            model_info: Model information
            
        Returns:
            Path to PDF file
        """
        pdf_path = self.reports_dir / 'report.pdf'
        
        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary metrics
            fig, axes = plt.subplots(3, 2, figsize=(11, 14))
            fig.suptitle(f'VOL-RISK LAB Report - {self.run_id}', fontsize=16, fontweight='bold')
            
            summary = evaluation_results['classification']['summary']
            
            # Create text summaries
            axes[0, 0].axis('off')
            axes[0, 0].text(0.1, 0.9, 'Classification Metrics', fontsize=14, fontweight='bold')
            axes[0, 0].text(0.1, 0.7, f"AUCPR (Crash): {summary['aucpr_crash']:.3f}", fontsize=12)
            axes[0, 0].text(0.1, 0.6, f"AUCPR (Boom): {summary['aucpr_boom']:.3f}", fontsize=12)
            axes[0, 0].text(0.1, 0.5, f"Accuracy: {summary['accuracy']:.1%}", fontsize=12)
            axes[0, 0].text(0.1, 0.4, f"Expected Utility: {summary['expected_utility']:.3f}", fontsize=12)
            
            # Include existing plots
            # Copy PR curves
            if (self.assets_dir / 'cls_pr_curves.png').exists():
                img = plt.imread(self.assets_dir / 'cls_pr_curves.png')
                axes[1, 0].imshow(img)
                axes[1, 0].axis('off')
                axes[1, 0].set_title('PR Curves')
            
            # Copy confusion matrix
            if (self.assets_dir / 'cls_confusion_matrix.png').exists():
                img = plt.imread(self.assets_dir / 'cls_confusion_matrix.png')
                axes[1, 1].imshow(img)
                axes[1, 1].axis('off')
                axes[1, 1].set_title('Confusion Matrix')
            
            plt.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close()
            
            # Add more pages with existing visualizations
            for img_file in sorted(self.assets_dir.glob('*.png')):
                fig = plt.figure(figsize=(11, 8.5))
                img = plt.imread(img_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(img_file.stem.replace('_', ' ').title())
                plt.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close()
        
        logger.info(f"PDF report saved to {pdf_path}")
        return str(pdf_path)

    # Update generate_report method to call PDF generation
    def generate_report(self,
                    evaluation_results: Dict[str, Any],
                    model_info: Dict[str, Any],
                    feature_importance: pd.DataFrame) -> str:
        """Generate complete HTML report."""
        logger.info(f"Generating report for run {self.run_id}")
        
        # ... existing code ...
        
        # Generate PDF if requested
        if self.config.get('reporting', {}).get('generate_pdf', False):
            try:
                self.generate_pdf_report(evaluation_results, model_info)
            except Exception as e:
                logger.error(f"PDF generation failed: {e}")
        
        return str(html_path)
    def _save_tables(self, results: Dict[str, Any]):
        """Save all tables as CSV files."""
        
        # Classification metrics
        if 'classification' in results:
            metrics_df = pd.DataFrame([results['classification']['core']])
            metrics_df.to_csv(self.tables_dir / 'classification_metrics.csv', index=False)
            
            # Confusion matrix
            cm_df = pd.DataFrame(
                results['classification']['confusion']['matrix'],
                index=results['classification']['confusion']['class_names'],
                columns=results['classification']['confusion']['class_names']
            )
            cm_df.to_csv(self.tables_dir / 'confusion_matrix.csv')
            
            # Utility decomposition
            if 'decomposition' in results['classification']['utility']:
                decomp_df = pd.DataFrame(
                    results['classification']['utility']['decomposition']
                ).T
                decomp_df.to_csv(self.tables_dir / 'utility_decomposition.csv')
                
    def _generate_html(self,
                      results: Dict[str, Any],
                      model_info: Dict[str, Any],
                      feature_importance: pd.DataFrame) -> str:
        """Generate HTML report content."""
        
        # Get summary metrics
        summary = results['classification']['summary']
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VOL-RISK LAB Report - {self.run_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        img {{
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 20px 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>VOL-RISK LAB Evaluation Report</h1>
    <p class="timestamp">Run ID: {self.run_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Executive Summary</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">AUCPR (Crash)</div>
            <div class="metric-value">{summary['aucpr_crash']:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">AUCPR (Boom)</div>
            <div class="metric-value">{summary['aucpr_boom']:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Expected Utility</div>
            <div class="metric-value">{summary['expected_utility']:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{summary['accuracy']:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Macro F1</div>
            <div class="metric-value">{summary['macro_f1']:.3f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Brier Score</div>
            <div class="metric-value">{summary['brier_mean']:.3f}</div>
        </div>
    </div>
    
    <h2>Classification Performance</h2>
    
    <h3>Precision-Recall Curves</h3>
    <div class="image-grid">
        <img src="assets/cls_pr_curves.png" alt="PR Curves">
    </div>
    
    <h3>Confusion Matrix</h3>
    <div class="image-grid">
        <img src="assets/cls_confusion_matrix.png" alt="Confusion Matrix">
    </div>
    
    <h3>Calibration Analysis</h3>
    <div class="image-grid">
        <img src="assets/cls_calibration.png" alt="Calibration">
    </div>
    
    <h3>Utility Optimization</h3>
    <div class="image-grid">
        <img src="assets/cls_utility_heatmap.png" alt="Utility Heatmap">
    </div>
    
    <h2>Feature Importance</h2>
    <div class="image-grid">
        <img src="assets/feature_importance.png" alt="Feature Importance">
    </div>
    
    <h2>Model Configuration</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Gamma Threshold</td><td>{self.config.get('targets', {}).get('default_gamma', 2.0)}</td></tr>
        <tr><td>Decision Mode</td><td>{self.config.get('decision', {}).get('mode', 'forced')}</td></tr>
        <tr><td>CV Folds</td><td>{self.config.get('cv', {}).get('n_splits', 8)}</td></tr>
        <tr><td>Base Models</td><td>{len(self.config.get('ensemble', {}).get('classification_bases', []))}</td></tr>
    </table>
    
    <h2>Additional Resources</h2>
    <ul>
        <li><a href="tables/classification_metrics.csv">Download Metrics Table</a></li>
        <li><a href="tables/confusion_matrix.csv">Download Confusion Matrix</a></li>
        <li><a href="tables/utility_decomposition.csv">Download Utility Decomposition</a></li>
        <li><a href="../model_card.md">Model Card</a></li>
        <li><a href="../appendix_features_formulas.md">Technical Appendix</a></li>
    </ul>
    
</body>
</html>
"""
        return html
    
    def _generate_appendix(self, model_info: Dict[str, Any], feature_importance: pd.DataFrame):
        """Generate technical appendix with feature formulas."""
        
        appendix_content = """# Technical Appendix - Feature Formulas and Definitions

## Feature Dictionary

This document provides complete specifications for all features used in the VOL-RISK LAB model.

### Volatility Features (prefix: vr_)

| Feature | Formula | Units | Becomes Known At | Description |
|---------|---------|-------|------------------|-------------|
| vr_rv | Σ(r_t,i)² | variance | Day t close | Realized variance |
| vr_bv | (π/2) × Σ\|r_i\|\|r_{i-1}\| | variance | Day t close | Bipower variation |
| vr_jv | max(RV - BV, 0) | variance | Day t close | Jump variation |
| vr_ewma_rv | EWMA(RV, λ=0.94) | variance | Day t close | Exponentially weighted RV |
| vr_vr | RV_t / EWMA(RV)_{t-1} | ratio | Day t close | Volatility ratio |

### Microstructure Features (prefix: iex_)

| Feature | Formula | Units | Becomes Known At | Description |
|---------|---------|-------|------------------|-------------|
| iex_spread_rel | (ask - bid) / mid | ratio | Minute close | Relative bid-ask spread |
| iex_depth_imbalance | (ask_size - bid_size) / (ask_size + bid_size) | ratio | Minute close | Order book imbalance |
| iex_mid_return | log(mid_t / mid_{t-1}) | return | Minute close | Mid-price return |

### Path Features (prefix: path_)

| Feature | Formula | Units | Becomes Known At | Description |
|---------|---------|-------|------------------|-------------|
| path_gap | log(open_t / close_{t-1}) | return | Day t open | Overnight gap |
| path_range | (high - low) / close | ratio | Day t close | Daily range |

## Data Availability Timeline

- **Minute OHLCV**: Available from 2020-07-27
- **IEX Microstructure**: Available from 2020-08-28 (NA before)
- **Daily EOD**: Available from 2019-01-02
- **Macro Data**: Monthly with 3-day lag applied

## Anti-Leakage Controls

1. **Purging**: 1 full trading day removed before each test fold
2. **Embargo**: 1 full trading day after each test fold
3. **Known-by-close**: Daily features use only data available at market close
4. **Macro lag**: Monthly data appears only after release + 3 business days

"""
        
        appendix_path = self.reports_dir / 'appendix_features_formulas.md'
        with open(appendix_path, 'w') as f:
            f.write(appendix_content)
            
    def _generate_model_card(self, results: Dict[str, Any], model_info: Dict[str, Any]):
        """Generate model card with key information."""
        
        model_card = f"""# Model Card - VOL-RISK LAB v3

## Model Details
- **Name**: VOL-RISK LAB Ensemble
- **Version**: 3.0.0
- **Run ID**: {self.run_id}
- **Date**: {datetime.now().strftime('%Y-%m-%d')}

## Intended Use
- **Primary**: Predict next-day crash/boom/normal market regimes
- **Secondary**: Forecast next-day log realized variance
- **Tertiary**: Estimate return quantiles (5%, 50%, 95%)

## Performance Metrics
- AUCPR (Crash): {results['classification']['summary']['aucpr_crash']:.3f}
- AUCPR (Boom): {results['classification']['summary']['aucpr_boom']:.3f}
- Expected Utility: {results['classification']['summary']['expected_utility']:.3f}
- Accuracy: {results['classification']['summary']['accuracy']:.1%}

## Training Data
- **Asset**: QQQ (Nasdaq-100 ETF)
- **Period**: 2020-07-27 to 2024-12-31 (training)
- **Frequency**: 1-minute bars, regular session only
- **Features**: {len(model_info.get('feature_names', []))} engineered features

## Limitations
- Model trained on QQQ only, may not generalize to other assets
- Requires complete minute-level data
- IEX microstructure features NA before 2020-08-28
- Assumes stationary utility function

## Ethical Considerations
- Model outputs are probabilistic and should not be sole basis for trading
- Past performance does not guarantee future results
- Users should understand financial risks

## Change Log
- v3.0.0: Initial production release
- Removed auction proxy features per specification
- Implemented forced-extremes decision mode
"""
        
        model_card_path = self.reports_dir / 'model_card.md'
        with open(model_card_path, 'w') as f:
            f.write(model_card)