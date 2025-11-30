#!/bin/bash
echo "Running XGBoost..."
python saransh-experiments/train/train_xgboost.py

echo -e "\nRunning CatBoost..."
python saransh-experiments/train/train_catboost.py

echo -e "\nRunning LightGBM..."
python saransh-experiments/train/train_lightgbm.py
