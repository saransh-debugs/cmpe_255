# cmpe_255

## Customer Churn Prediction - Model Benchmarking Project

This project implements and benchmarks various machine learning models for predicting customer churn using the Cell2Cell dataset.

## Project Structure

```
cmpe_255/
├── train/                          # Training scripts for all models
│   ├── train_logistic_regression.py
│   ├── train_linear_sgd.py
│   ├── train_randomforest.py
│   ├── train_decision_tree.py
│   ├── train_xgboost.py
│   ├── train_catboost.py
│   ├── train_lightgbm.py
│   ├── train_leakage_analysis.py
│   ├── experiment_group_robustness.py
│   ├── experiment_imbalance.py
│   └── ...
├── processed_data_folds_strict/    # Processed data (no leakage)
├── processed_data_folds_permissive/# Processed data (with leakage cols)
├── shap_plots/                     # SHAP explainability plots
├── run_benchmarks.sh               # Main benchmark runner
├── run_experiments.sh              # Additional experiments runner
├── churn_pipeline_saransh.py       # Data preprocessing pipeline
├── cell2celltrain.csv              # Training data
├── cell2cellholdout.csv            # Holdout data
└── README.md
```

## Quick Start

```bash
# Run all benchmarks
./run_benchmarks.sh

# Run additional experiments (imbalance handling, group robustness)
./run_experiments.sh
```

---

## Benchmark Results

### Model Performance Summary (5-Fold Cross-Validation, Strict Mode - No Leakage)

| Model | ROC-AUC | PR-AUC | Brier Score | Recall@20% |
|-------|---------|--------|-------------|------------|
| **CatBoost** | **0.6740 ± 0.0036** | **0.4508 ± 0.0041** | **0.1893 ± 0.0007** | **0.3327 ± 0.0057** |
| LightGBM | 0.6688 ± 0.0048 | 0.4430 ± 0.0038 | 0.1901 ± 0.0008 | 0.3252 ± 0.0031 |
| XGBoost | 0.6560 ± 0.0065 | 0.4288 ± 0.0035 | 0.1963 ± 0.0014 | 0.3135 ± 0.0033 |
| Random Forest | 0.6517 ± 0.0054 | 0.4172 ± 0.0058 | 0.2067 ± 0.0010 | 0.3037 ± 0.0077 |
| Decision Tree | 0.6185 ± 0.0040 | 0.3731 ± 0.0011 | 0.2382 ± 0.0010 | 0.2754 ± 0.0057 |
| Logistic Regression | 0.6143 ± 0.0031 | 0.3813 ± 0.0046 | 0.2395 ± 0.0003 | 0.2777 ± 0.0040 |
| Linear SGD | 0.5891 ± 0.0048 | 0.3571 ± 0.0078 | 0.2504 ± 0.0036 | 0.2642 ± 0.0064 |

**Key Finding:** CatBoost achieves the best performance across all metrics, with XGBoost and LightGBM close behind.

---

### Detailed Model Results

#### 1. Logistic Regression (Baseline)
```
ROC-AUC:      0.6143 ± 0.0031
PR-AUC:       0.3813 ± 0.0046
Brier Score:  0.2395 ± 0.0003
Recall@20%:   0.2777 ± 0.0040

Aggregated Confusion Matrix (Threshold 0.5):
[[22042  14294]
 [6511  8200]]
```

#### 2. Linear SGD (Fast Linear)
```
ROC-AUC:      0.5891 ± 0.0048
PR-AUC:       0.3571 ± 0.0078
Brier Score:  0.2504 ± 0.0036
Recall@20%:   0.2642 ± 0.0064

Aggregated Confusion Matrix (Threshold 0.5):
[[21184  15152]
 [6701  8010]]
```

#### 3. Random Forest
```
ROC-AUC:      0.6517 ± 0.0054
PR-AUC:       0.4172 ± 0.0058
Brier Score:  0.2067 ± 0.0010
Recall@20%:   0.3037 ± 0.0077

Aggregated Confusion Matrix (Threshold 0.5):
[[29885  6451]
 [9761  4950]]
```

#### 4. Decision Tree
```
ROC-AUC:      0.6185 ± 0.0040
PR-AUC:       0.3731 ± 0.0011
Brier Score:  0.2382 ± 0.0010
Recall@20%:   0.2754 ± 0.0057

Aggregated Confusion Matrix (Threshold 0.5):
[[18597  17739]
 [4949  9762]]
```

#### 5. XGBoost
```
ROC-AUC:      0.6560 ± 0.0065
PR-AUC:       0.4288 ± 0.0035
Brier Score:  0.1963 ± 0.0014
Recall@20%:   0.3135 ± 0.0033

Aggregated Confusion Matrix (Threshold 0.5):
[[33183  3153]
 [11632  3079]]
```

#### 6. CatBoost (Best Model)
```
ROC-AUC:      0.6740 ± 0.0036
PR-AUC:       0.4508 ± 0.0041
Brier Score:  0.1893 ± 0.0007
Recall@20%:   0.3327 ± 0.0057

Aggregated Confusion Matrix (Threshold 0.5):
[[34444  1892]
 [12386  2325]]
```

#### 7. LightGBM
```
ROC-AUC:      0.6688 ± 0.0048
PR-AUC:       0.4430 ± 0.0038
Brier Score:  0.1901 ± 0.0008
Recall@20%:   0.3252 ± 0.0031

Aggregated Confusion Matrix (Threshold 0.5):
[[34752  1584]
 [12733  1978]]
```

---

### Leakage Impact Analysis

Comparison of models trained with and without potential leakage columns (RetentionCalls, RetentionOffersAccepted, etc.):

| Model | View | ROC-AUC | PR-AUC | Lift (ROC) |
|-------|------|---------|--------|------------|
| LogisticRegression | Strict | 0.6139 | 0.3816 | - |
| | Permissive | 0.6198 | 0.3880 | +0.0059 |
| RandomForest | Strict | 0.6488 | 0.4155 | - |
| | Permissive | 0.6536 | 0.4227 | +0.0048 |
| XGBoost | Strict | 0.6560 | 0.4288 | - |
| | Permissive | 0.6549 | 0.4267 | -0.0011 |

**Key Finding:** Leakage columns provide minimal performance boost (~0.5% AUC), suggesting the strict preprocessing is appropriate and models are not overfitting to retention-related signals.

---

### Group Robustness (ServiceArea Split)

Testing model generalization across different geographic regions using StratifiedGroupKFold:

```
============================================================
GROUP ROBUSTNESS RESULTS (Split by ServiceArea)
============================================================
Average AUC: 0.6627
Average PR-AUC: 0.4390

Comparison to Random Split (Baseline XGBoost ~0.6560):
Delta: +0.0067

>> SUCCESS: Model generalizes well across different regions.
```

---

### Imbalance Handling Strategies

Comparison of different approaches to handle class imbalance:

| Strategy | ROC-AUC | PR-AUC |
|----------|---------|--------|
| Baseline (Unweighted) | 0.6560 | 0.4288 |
| Class Weighted | 0.6515 | 0.4267 |
| SMOTE | 0.6557 | 0.4281 |
| SMOTE+Tomek | 0.6547 | 0.4256 |

**Key Finding:** Standard unweighted training performs on par or better than explicit imbalance handling techniques for this dataset.

---

## Conclusions

1. **Best Model:** CatBoost achieves the highest performance (ROC-AUC: 0.674, Recall@20%: 33.3%)
2. **Gradient Boosting dominates:** All three gradient boosting methods (CatBoost, LightGBM, XGBoost) outperform traditional ML methods
3. **No significant leakage:** Strict preprocessing removes potential leakage without significant performance loss
4. **Good generalization:** Models generalize well across different geographic regions
5. **Class imbalance:** Built-in handling by gradient boosting models is sufficient; explicit resampling doesn't improve results

---

## Requirements

See `requirements.txt` for dependencies. Key packages:
- scikit-learn
- xgboost
- catboost
- lightgbm
- pandas
- numpy
- imbalanced-learn (for SMOTE experiments)
