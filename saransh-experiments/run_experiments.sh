#!/bin/bash
echo "------------------------------------------"
echo "STEP 2: Imbalance Handling (SMOTE, Weights)"
echo "------------------------------------------"
python saransh-experiments/train/experiment_imbalance.py

echo -e "\n------------------------------------------"
echo "STEP 3: Group Robustness (ServiceArea)"
echo "------------------------------------------"
python saransh-experiments/train/experiment_group_robustness.py

