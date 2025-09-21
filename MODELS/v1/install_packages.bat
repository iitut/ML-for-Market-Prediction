@echo off
echo ========================================
echo Installing ML Model v1 Dependencies
echo ========================================
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing required packages...
echo.

REM Core data processing
pip install pandas>=1.5.0
pip install numpy>=1.23.0
pip install scipy>=1.9.0

REM Machine Learning packages
pip install scikit-learn>=1.1.0
pip install xgboost>=1.7.0
pip install tensorflow>=2.10.0

REM Visualization
pip install matplotlib>=3.5.0
pip install seaborn>=0.12.0

REM Additional utilities
pip install tqdm>=4.64.0
pip install joblib>=1.2.0

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo You can now run: python main.py
echo.
pause