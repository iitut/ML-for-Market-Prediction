@echo off
REM Batch script to run VOL-RISK LAB training

REM Activate virtual environment
call "C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\env_for_models\v3_env\Scripts\activate.bat"

REM Navigate to project folder
cd "C:\Users\shyto\Downloads\Closing-Auction Variance Measures\ML_MP\MODELS\v3"

REM Run training
python scripts/train_complete.py

REM Keep window open to see results
pause