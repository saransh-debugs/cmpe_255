#!/bin/bash
echo "=========================================="
echo "      RUNNING FULL MODEL BENCHMARKS       "
echo "=========================================="

# Use the specific conda environment python
PYTHON_CMD="/opt/anaconda3/envs/myenv/bin/python"

# Ensure imports from the current directory work
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo -e "\n--- 1. Logistic Regression (Baseline) ---"
$PYTHON_CMD saransh-experiments/train/train_logistic_regression.py

echo -e "\n--- 2. Linear SGD (Fast Linear) ---"
$PYTHON_CMD saransh-experiments/train/train_linear_sgd.py

echo -e "\n--- 3. Random Forest ---"
$PYTHON_CMD saransh-experiments/train/train_randomforest.py

echo -e "\n--- 4. Decision Tree ---"
$PYTHON_CMD saransh-experiments/train/train_decision_tree.py

echo -e "\n--- 5. XGBoost ---"
$PYTHON_CMD saransh-experiments/train/train_xgboost.py

echo -e "\n--- 6. CatBoost ---"
$PYTHON_CMD saransh-experiments/train/train_catboost.py

echo -e "\n--- 7. LightGBM ---"
$PYTHON_CMD saransh-experiments/train/train_lightgbm.py

echo -e "\n--- 8. Leakage Analysis ---"
$PYTHON_CMD saransh-experiments/train/train_leakage_analysis.py

echo -e "\n--- 9. Group Robustness ---"
$PYTHON_CMD saransh-experiments/train/experiment_group_robustness.py

echo -e "\n--- 10. Imbalance Handling ---"
$PYTHON_CMD saransh-experiments/train/experiment_imbalance.py

echo -e "\n=========================================="
echo "           BENCHMARK COMPLETE             "
echo "=========================================="
