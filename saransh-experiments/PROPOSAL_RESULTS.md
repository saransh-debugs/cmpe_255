# Churn Prediction Proposal Implementation: Results Summary

## 1. Leakage Control Analysis (Strict vs. Permissive)
We compared models trained on **Strict** data (excluding retention variables) vs. **Permissive** data (including them) to quantify leakage.

| Model | View | ROC-AUC | Lift |
|-------|------|---------|------|
| **Logistic Regression** | Strict | 0.6139 | - |
| | Permissive | 0.6198 | +0.0059 |
| **Random Forest** | Strict | 0.6488 | - |
| | Permissive | 0.6536 | +0.0048 |
| **XGBoost** | Strict | 0.6560 | - |
| | Permissive | 0.6549 | -0.0011 |

**Conclusion:** The inclusion of retention-related variables (`MadeCallToRetentionTeam`, etc.) has **negligible impact** on model performance. There is no significant data leakage inflating the scores.

## 2. Explainability & Feature Drivers
Using the **Strict** XGBoost model (AUC ~0.65), we identified the key drivers of churn.

### Top Features (SHAP)
1.  **CurrentEquipmentDays** (Lifecycle): Older equipment increases churn risk.
2.  **ServiceArea** (Demographic/Ops): Location significantly impacts retention.
3.  **MonthsInService** (Lifecycle): Tenure is a strong predictor.
4.  **PercChangeMinutes** (Usage): Volatility in usage signals churn.
5.  **MonthlyMinutes** (Usage): High/Low usage levels correlate with risk.

### Feature Family Ablation
We removed entire families of features to measure their unique contribution.

| Feature Family | AUC Impact (Drop) | Status |
|----------------|-------------------|--------|
| **Lifecycle** | **-0.0445** | 🚨 **Critical** |
| **Usage & Revenue** | **-0.0299** | ⚠️ **High Value** |
| Friction | +0.0045 (Improved) | Noise / Redundant |
| Demographics | +0.0043 (Improved) | Noise / Redundant |

**Insight:** The model relies almost entirely on **Lifecycle** (tenure/device age) and **Usage Dynamics**. "Friction" (dropped calls) and static "Demographics" add noise.

## 3. Calibration
*   **Brier Score:** 0.1981
*   **Reliability:** The model tends to be **over-confident** at high probabilities (e.g., predicting 93% risk when actual risk is 83%). Calibration (Isotonic/Platt) is recommended before using scores for financial decisions.

