#!/bin/bash
# Use the specific conda environment python
PYTHON_CMD="/opt/anaconda3/envs/myenv/bin/python"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "------------------------------------------"
echo "STEP 2: Imbalance Handling (SMOTE, Weights)"
echo "------------------------------------------"
$PYTHON_CMD train/experiment_imbalance.py

echo -e "\n------------------------------------------"
echo "STEP 3: Group Robustness (ServiceArea)"
echo "------------------------------------------"
$PYTHON_CMD train/experiment_group_robustness.py

