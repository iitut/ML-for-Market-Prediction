# Market Prediction Model v1.0

## Overview
This is a comprehensive machine learning system for predicting market crashes, booms, and volatility using QQQ data with advanced feature engineering.

## Project Structure
```
C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\MODELS\v1\
│
├── config.py                 # Configuration and constants
├── data_loader.py           # Data loading and merging module
├── feature_engineering.py   # Feature calculation module
├── target_creation.py       # Target variable creation
├── model.py                 # ML model implementation
├── visualization.py         # Plotting and monitoring
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
└── plots/                   # Generated visualizations
    ├── crash_boom_predictions.png
    ├── volatility_predictions.png
    └── performance_dashboard.png
```

## Features Implemented

### Volatility Features
- **Realized Variance (RV)**: Calculated from minute-level returns
- **Parkinson Estimator**: Using daily high/low prices
- **Garman-Klass**: OHLC-based volatility
- **Rogers-Satchell**: Drift-free estimator
- **Bipower Variation**: Robust to jumps
- **Jump Variation**: Detecting price jumps
- **Realized Skewness/Kurtosis**: Higher moments
- **EWMA of RV**: Exponentially weighted moving average
- **Volatility Ratio**: Current vs historical volatility

### Trend/Path Features
- **Gap**: Overnight return
- **Intraday Range Ratio**: Normalized daily range
- **Last-hour Return**: Final hour momentum

### Liquidity Features
- **Dollar Volume**: Log-transformed dollar volume
- **Volume Volatility**: Intraday volume variation

## Prediction Targets

### 1. Crash/Boom Classification (Primary)
- **Crash**: Standardized return ≤ -2σ
- **Boom**: Standardized return ≥ +2σ
- **Normal**: Otherwise
- Output: Probabilities for each class

### 2. Next-day Volatility (Secondary)
- Target: Log realized variance for next day
- Used for risk management and position sizing

### 3. Return Quantiles (Tertiary)
- 5%, 50%, and 95% quantiles of returns
- Provides full distribution forecast

## Model Architecture

### Classification Model (XGBoost)
- 200 trees, max depth 6
- Learning rate: 0.01
- Subsample: 0.8
- Handles class imbalance

### Volatility Model (Neural Network)
- 3 hidden layers (128, 64, 32 neurons)
- Batch normalization and dropout
- Adam optimizer with MSE loss

## Training Strategy

### Expanding Window Approach
- Initial training: 252 days (1 year)
- Progressive expansion: adds each new day
- No future data leakage guaranteed
- Periodic model retraining

### Data Integrity Checks
- Temporal ordering validation
- Train/test separation verification
- Missing value handling
- Feature scaling without leakage

## Monitoring Features

### Detailed Monitoring Points
- First prediction (day 0)
- Every 100th day
- Shows:
  - Training data used
  - Feature importance
  - Prediction probabilities
  - Actual vs predicted values

### Performance Metrics
- Overall accuracy
- Class-specific precision
- Volatility MAE and R²
- Rolling accuracy trends
- Regime-based performance

## Installation

1. **Install Python 3.8+**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify data files exist in:**
```
C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\Data\
```

## Usage

### Quick Start
```python
# Run the complete pipeline
python main.py
```

### Custom Usage
```python
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model import MarketPredictionModel

# Load data
loader = DataLoader()
data = loader.load_master_dataset()

# Engineer features
engineer = FeatureEngineer(data)
features = engineer.engineer_all_features()

# Train model
model = MarketPredictionModel(features, targets)
predictions = model.train_expanding_window()
```

## Output Files

### Predictions
- `predictions.csv`: All predictions with probabilities
- `metrics.csv`: Model performance metrics
- `feature_importance.csv`: Feature rankings

### Visualizations
- Crash/boom predictions over time
- Volatility forecast accuracy
- Performance dashboard
- Confusion matrices

## Performance Expectations

### Classification
- Overall accuracy: 60-70%
- Crash detection: 30-50%
- Boom detection: 30-50%

### Volatility
- R²: 0.3-0.5
- MAE: < 0.5 (log scale)

## Key Innovations

1. **No Look-Ahead Bias**: Strict temporal separation
2. **Multi-Target Learning**: Simultaneous prediction of multiple objectives
3. **Adaptive Training**: Model updates with new data
4. **Comprehensive Features**: 30+ engineered features
5. **Real-time Monitoring**: Track model performance continuously

## Troubleshooting

### Common Issues

**Missing Data Files:**
- Verify all CSV files are in the Data folder
- Check file names match config.py

**Memory Issues:**
- Reduce `n_estimators` in XGBoost
- Use smaller batch sizes for neural network
- Process intraday data in chunks

**Slow Training:**
- Set `tree_method='hist'` in XGBoost
- Reduce neural network epochs
- Use GPU if available (`device='cuda'`)

## Future Improvements

1. **Additional Features:**
   - Order book imbalance
   - Options flow indicators
   - Sentiment analysis

2. **Model Enhancements:**
   - Ensemble methods
   - LSTM for sequence modeling
   - Attention mechanisms

3. **Risk Management:**
   - Portfolio optimization
   - Position sizing algorithms
   - Stop-loss optimization

## Contact
Ivan Shytov
Ardévaz College, Sion, Switzerland

## License
This project is for educational purposes.

---

*Last Updated: 2025*