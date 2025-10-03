# Model v2 - Enhanced Market Prediction System

## 📊 **COMPLETE SYSTEM OVERVIEW**

### **Location**: `C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\MODELS\v2\`

---

## 🎯 **KEY IMPROVEMENTS FROM V1**

### **1. Classification Thresholds (More Sensitive)**
- **Previous (v1)**: Only 3 classes with ±2σ threshold (~5% events)
- **New (v2)**: 7 classes with percentage-based thresholds:
  - **Extreme Crash**: < -3% daily return
  - **Crash**: -3% to -1.5% 
  - **Mild Down**: -1.5% to -0.5%
  - **Normal**: -0.5% to +0.5%
  - **Mild Up**: +0.5% to +1.5%
  - **Boom**: +1.5% to +3%
  - **Extreme Boom**: > +3%

### **2. Multiple Time Horizons**
- **Intraday**: 1 hour after market open
- **Daily**: Next day predictions
- **Weekly**: 5-day outlook
- **Monthly**: 21-day outlook

### **3. Ensemble of 6 Models**
1. **XGBoost** - Gradient boosting
2. **LightGBM** - Fast gradient boosting
3. **CatBoost** - Categorical-friendly boosting
4. **Random Forest** - Bagging ensemble
5. **Gradient Boosting** - Traditional boosting
6. **Neural Network** - Deep learning

### **4. Enhanced Features (70+ total)**
- **Volatility** (12 features): Yang-Zhang, Close-to-Close, Parkinson, Garman-Klass, etc.
- **Technical** (10 features): RSI, MACD, Bollinger Bands, Stochastic
- **Moving Averages** (15 features): SMA (5,10,20,50,100,200), EMA (12,26,50)
- **Volume** (8 features): OBV, MFI, Volume ROC, A/D Line
- **Momentum** (7 features): Momentum, ROC, Williams %R
- **Microstructure** (5 features): Amihud illiquidity, Kyle's lambda
- **Support/Resistance** (4 features): Dynamic levels
- **Regime** (3 features): Trend strength, volatility regime

---

## 📈 **PREDICTION OUTPUTS**

### **Classification Outputs**
1. **7-Class Market Direction**: Extreme crash to extreme boom
2. **3-Class Direction**: Down/Neutral/Up
3. **Week Outlook**: Weekly crash/normal/boom

### **Regression Outputs**
1. **Next Day Return**: Continuous percentage
2. **Next Week Return**: 5-day return
3. **Next Month Return**: 21-day return
4. **Hour After Open**: Intraday movement
5. **Volatility Forecast**: Log realized variance

### **Probabilistic Outputs**
1. **Daily Crash Probability**: P(return < -1.5%)
2. **Daily Boom Probability**: P(return > +1.5%)
3. **Weekly Crash Probability**: P(week return < -3%)
4. **Weekly Boom Probability**: P(week return > +3%)

---

## 🛠️ **TECHNICAL IMPLEMENTATION**

### **Methods Used**

#### **Machine Learning Algorithms**
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Deep Learning**: Multi-layer Neural Network (256-128-64-32 architecture)
- **Ensemble Weighting**: Dynamic weighted averaging based on validation performance

#### **Feature Engineering Formulas**

**Volatility Estimators:**
- Yang-Zhang: `σ²_YZ = σ²_overnight + k×σ²_open-to-close + (1-k)×σ²_rogers-satchell`
- Parkinson: `σ²_P = (1/4ln2) × [ln(H/L)]²`
- Garman-Klass: `σ²_GK = 0.5×[ln(H/L)]² - (2ln2-1)×[ln(C/O)]²`

**Technical Indicators:**
- RSI: `RSI = 100 - (100/(1 + RS))` where RS = avg gain/avg loss
- MACD: `MACD = EMA(12) - EMA(26)`
- Bollinger Position: `(Price - Lower Band)/(Upper Band - Lower Band)`

**Microstructure:**
- Amihud Illiquidity: `ILLIQ = |Return|/(Dollar Volume) × 10⁶`
- Kyle's Lambda: Regression coefficient of price impact

#### **Training Strategy**
- **Expanding Window**: Starts with 252 days, progressively adds data
- **No Look-Ahead Bias**: Strict temporal separation
- **Cross-Validation**: 5-fold time series split
- **Ensemble Weighting**: Performance-based dynamic weights

---

## 📁 **FILE STRUCTURE**

```
v2/
├── config.py                    # Enhanced configuration
├── enhanced_features.py         # 70+ feature calculations
├── ensemble_model.py           # 6-model ensemble system
├── main_v2.py                  # Main execution
├── visualization_v2.py         # Enhanced plotting
│
└── results/
    ├── predictions_v2.csv      # All predictions
    ├── feature_importance_v2.csv # Feature rankings
    ├── performance_report.txt  # Text summary
    └── plots/
        ├── 7class_performance.png
        ├── probability_analysis.png
        └── comprehensive_dashboard.png
```

---

## 💻 **HOW TO RUN**

### **Installation**
```bash
cd "C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\MODELS\v2"

# Install additional packages for v2
pip install lightgbm catboost
```

### **Execution**
```bash
python main_v2.py
```

---

## 📊 **EXPECTED PERFORMANCE**

### **Classification Metrics**
- **7-Class Accuracy**: 40-50% (much harder than 3-class)
- **Crash Detection**: 40-60% (more sensitive thresholds)
- **Boom Detection**: 40-60% (more events to learn from)
- **Direction Accuracy**: 55-65%

### **Regression Metrics**
- **Daily Return R²**: 0.1-0.3
- **Weekly Return R²**: 0.2-0.4
- **Volatility R²**: 0.3-0.5

### **Probability Calibration**
- Well-calibrated probabilities (diagonal on calibration plot)
- Meaningful risk assessments

---

## 🔍 **MONITORING OUTPUTS**

### **Console Output**
```
7-Class Market Direction:
  extreme_crash   :   2.1% (n=35)
  crash          :   8.3% (n=139)
  mild_down      :  18.7% (n=313)
  normal         :  41.2% (n=690)
  mild_up        :  19.1% (n=320)
  boom           :   8.8% (n=147)
  extreme_boom   :   1.8% (n=30)

Model ensemble weights:
  xgboost: 0.235
  lightgbm: 0.218
  catboost: 0.203
  random_forest: 0.152
  gradient_boosting: 0.192
```

### **Visualizations**
1. **7-Class Performance**: Confusion matrix, time series, distribution
2. **Probability Analysis**: Calibration curves, probability time series
3. **Comprehensive Dashboard**: All metrics in one view

### **Text Report**
Complete performance metrics saved to `performance_report.txt`

---

## 🚀 **KEY ADVANTAGES**

1. **More Sensitive Detection**: Catches more market movements
2. **Multi-Timeframe**: From intraday to monthly predictions
3. **Ensemble Robustness**: 6 models reduce overfitting
4. **Rich Feature Set**: 70+ engineered features
5. **Comprehensive Outputs**: 15+ different predictions
6. **No Data Leakage**: Verified temporal integrity
7. **Dynamic Adaptation**: Models update with new data

---

## 📈 **FUTURE ENHANCEMENTS**

1. **Add Sentiment Analysis**: News/social media features
2. **Options Data**: Implied volatility, put/call ratios
3. **Order Book Features**: Bid-ask spread, depth
4. **Attention Mechanisms**: Transformer models
5. **Online Learning**: Real-time model updates
6. **Portfolio Optimization**: Position sizing based on predictions

---

## ⚠️ **IMPORTANT NOTES**

- **Independent Outputs**: Each prediction is made independently to avoid cascading errors
- **Temporal Integrity**: Future data never used for current predictions
- **Ensemble Diversity**: Different algorithms capture different patterns
- **Balanced Thresholds**: More events in each class for better learning

---

## 📞 **SUPPORT**

Created by: Ivan Shytov
Institution: Ardévaz College, Sion, Switzerland
Purpose: Educational/Research

---

*Model v2 represents a significant upgrade with more sensitive detection, multiple timeframes, and ensemble robustness.*