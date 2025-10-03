# VOL-RISK LAB v3 - Production ML System for Volatility Risk Prediction

A production-grade, leakage-safe, regime-aware super-ensemble for predicting market crashes, booms, and volatility.

## 🎯 Project Overview

VOL-RISK LAB is a comprehensive machine learning system designed to predict extreme market movements and volatility for the QQQ ETF (Nasdaq-100). The system implements three prediction heads:

1. **Classification Head (Primary)**: Predicts next-day Crash/Boom/Normal regimes
2. **Volatility Head (Secondary)**: Forecasts next-day log realized variance
3. **Quantile Head (Tertiary)**: Estimates return quantiles (5%, 50%, 95%)

## 🚀 Key Features

- **Zero Data Leakage**: Strict temporal controls with purging and embargo
- **Regime-Aware**: Adapts to different volatility regimes and market conditions
- **Utility-Optimized**: Decision policy maximizes expected utility rather than accuracy
- **Production-Ready**: Complete pipeline from data ingestion to report generation
- **Comprehensive Evaluation**: AUCPR, utility metrics, calibration analysis, and regime breakdowns

## 📁 Project Structure

```
vol-risk-lab/
├── configs/                 # Hydra configuration files
│   ├── default.yaml        # Main configuration
│   ├── cv.yaml            # Cross-validation settings
│   └── models/            # Model-specific configs
├── scripts/                # Entry point scripts
│   ├── train.py           # Main training orchestrator
│   ├── evaluate.py        # Evaluation script
│   └── report.py          # Report generation
├── src/vrisk/              # Core library code
│   ├── io/                # Data loading and validation
│   ├── calendars/         # Trading calendar utilities
│   ├── labeling/          # Target label generation
│   ├── features/          # Feature engineering modules
│   ├── models/            # Model implementations
│   ├── ensembles/         # Stacking and ensemble methods
│   ├── training/          # Training orchestration
│   ├── evaluation/        # Metrics and evaluation
│   └── reporting/         # Report generation
└── tests/                  # Unit and integration tests
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional, for faster training)
- 64GB+ RAM recommended
- NVMe SSD for data storage

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd vol-risk-lab
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up data paths in `configs/default.yaml`:
```yaml
data:
  base_path: "C:/Users/shyto/Downloads/Closing-Auction Variance Measures/ML_MP"
  master_parquet: "${data.base_path}/master_minute.parquet"
```

## 📊 Data Requirements

The system expects a master minute dataset with the following structure:

### Required Columns
- **Minute Market Data**: `ohlcv_open`, `ohlcv_high`, `ohlcv_low`, `ohlcv_close`, `ohlcv_volume`, `ohlcv_vwap`
- **IEX Microstructure**: `iex_bid_price`, `iex_bid_size`, `iex_ask_price`, `iex_ask_size`
- **Daily EOD**: `eod_Open`, `eod_High`, `eod_Low`, `eod_Close`, `eod_Volume`
- **Calendar Flags**: `is_holiday`, `is_early_close`, `is_opx`, etc.
- **Macro Data**: `umcsi`, `us_policy_uncertainty`

### Data Specifications
- **Granularity**: 1-minute bars for regular session only (9:30-16:00 ET)
- **Timezone**: UTC timestamps with NY session_date
- **Coverage**: 2020-07-27 to 2025-08-29
- **Format**: Parquet (recommended) or CSV

## 🏃 Running the System

### Training a Model

Basic training:
```bash
python scripts/train.py
```

With custom configuration:
```bash
python scripts/train.py \
  targets.default_gamma=2.5 \
  cv.n_splits=10 \
  decision.mode=forced
```

### Configuration Options

Key parameters in `configs/default.yaml`:

```yaml
targets:
  gammas: [1.5, 2.0, 2.5, 3.0]  # Crash/boom thresholds
  
decision:
  mode: "forced"  # or "standard"
  cost_matrix:   # Utility values for each outcome
    crash_crash: 2.0
    crash_normal: -0.5
    # ... etc
    
cv:
  n_splits: 8
  embargo_days: 1
  purge_days: 1
```

## 📈 Model Components

### Base Learners
- LightGBM (gradient boosting)
- XGBoost (gradient boosting)
- CatBoost (gradient boosting)
- Random Forest
- SVM with RBF kernel
- Elastic Net Logistic
- Neural Networks (MLP, TCN)

### Ensemble Strategy
- Out-of-fold (OOF) predictions from base models
- Stacking with LightGBM meta-classifier
- Utility-aware optimization
- Temperature calibration

### Feature Categories
- **Volatility**: RV, BV, JV, Parkinson, Garman-Klass
- **Microstructure**: Spread, depth imbalance, quote intensity
- **Path**: Gaps, ranges, cumulative returns
- **Liquidity**: Dollar volume, volume volatility
- **Regime**: EWMA, volatility ratios, regime indicators
- **Macro**: Consumer sentiment, policy uncertainty (with lag)
- **Calendar**: Options expiration, holidays, early closes

## 📊 Evaluation Metrics

### Classification
- **AUCPR**: Area under Precision-Recall curve (focus on extremes)
- **Expected Utility**: Based on configurable cost matrix
- **Calibration**: ECE, reliability plots
- **Confusion Matrix**: By regime and event type

### Volatility
- **R²**: Coefficient of determination
- **MAE**: Mean absolute error
- **DM Test**: vs HAR-RV and EWMA baselines

### Quantiles
- **Coverage**: Empirical vs nominal
- **Pinball Loss**: Quantile-specific loss
- **Interval Width**: After conformal adjustment

## 📝 Output Artifacts

Each training run produces:

```
outputs/<run_id>/
├── models/              # Trained model files
├── reports/
│   ├── index.html      # Main evaluation report
│   ├── assets/         # Charts and visualizations
│   └── tables/         # CSV metrics tables
├── features.parquet     # Engineered features
├── config.yaml         # Configuration used
├── decision_audit.csv  # Detailed decision log
└── metadata.json       # Run metadata
```

## 🔒 Anti-Leakage Controls

1. **Temporal Separation**: 
   - 1-day purging before test
   - 1-day embargo after test
   
2. **Feature Alignment**:
   - Daily data available only at close
   - Macro data with 3-day lag
   - No forward-looking information

3. **Cross-Validation**:
   - Anchored expanding windows
   - Proper time series splits
   - No data shuffling

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run integration test:
```bash
pytest tests/integration/test_pipeline.py
```

## 📚 References

The system implements techniques from:
- Realized Variance estimation (Andersen et al.)
- Bipower Variation (Barndorff-Nielsen & Shephard)
- HAR-RV model (Corsi)
- Stacking ensembles (Wolpert)
- Conformal prediction (Vovk et al.)

## ⚠️ Important Notes

1. **Data Quality**: System expects clean minute data with no gaps
2. **Memory Usage**: Full pipeline requires ~64GB RAM
3. **Training Time**: ~2-4 hours on 32-core CPU
4. **IEX Limitation**: Microstructure features NA before 2020-08-28

## 📧 Support

For issues or questions:
1. Check the technical appendix in reports
2. Review configuration options
3. Examine the decision audit CSV

## 📄 License

[Specify your license here]

---

**Version**: 3.0.0  
**Last Updated**: 2025 
**Status**: In development