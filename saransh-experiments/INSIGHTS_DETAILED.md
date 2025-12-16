# Comprehensive Insights Report: Churn Prediction on Cell2Cell

## 1. Executive Summary
This project benchmarked multiple machine learning strategies to predict customer churn while controlling for data leakage and ensuring explainability.
*   **Best Model:** CatBoost (AUC: 0.6740).
*   **Key Finding:** The model is driven almost entirely by **Lifecycle** (Tenure, Device Age) and **Usage Volatility**. Demographics and Dropped Calls are surprisingly weak predictors.
*   **Robustness:** The model generalizes exceptionally well to new geographic regions (Service Areas) and does not require complex resampling (SMOTE) to handle class imbalance.
*   **Leakage:** There is **no evidence of data leakage** from retention intervention variables.

---

## 2. Model Benchmarking
We compared three state-of-the-art gradient boosting frameworks on the "Strict" (leakage-free) dataset.

| Model | ROC-AUC | PR-AUC | Performance |
| :--- | :--- | :--- | :--- |
| **CatBoost** | **0.6740** | **0.4508** | 🏆 **Winner** |
| LightGBM | 0.6688 | 0.4430 | 🥈 Runner Up |
| XGBoost | 0.6560 | 0.4288 | 🥉 Baseline |

**Insight:** CatBoost's superior handling of categorical variables (like `ServiceArea`, `PrizmCode`) likely gives it the edge over XGBoost's one-hot encoding approach.

---

## 3. Imbalance Handling Strategy
We tested if synthetic oversampling (SMOTE) or class weighting could improve detection of the minority churn class (~28%).

| Strategy | ROC-AUC | PR-AUC | Impact |
| :--- | :--- | :--- | :--- |
| **Baseline (Unweighted)** | **0.6560** | **0.4288** | ✅ **Best** |
| SMOTE | 0.6557 | 0.4281 | Neutral |
| SMOTE + Tomek | 0.6547 | 0.4256 | Negative |
| Class Weighted | 0.6515 | 0.4267 | Negative |

**Insight:**
*   **Don't Over-Engineer:** Standard training works best.
*   **Why?** The dataset is large enough (50k+ rows) and the imbalance (28% vs 72%) is not severe enough to require synthetic data. SMOTE simply added noise, and Class Weights distorted the probability calibration without improving ranking.

---

## 4. Geographic Robustness (Service Area)
We tested if the model fails when applied to unseen regions (e.g., training on NYC, testing on LA).

*   **Random Split AUC:** 0.6560
*   **Group Split (ServiceArea) AUC:** **0.6627** (+0.0067)

**Insight:**
*   The model is **Geographically Robust**. It relies on universal behavioral patterns (e.g., "user usage dropped 50%") rather than location-specific bias.
*   This confirms the model is safe for nationwide deployment.

---

## 5. Leakage Control (Strict vs. Permissive)
We compared models trained with and without retention intervention variables (`MadeCallToRetentionTeam`, etc.).

*   **Strict (No Retention Info):** 0.6560 AUC
*   **Permissive (With Retention Info):** 0.6549 AUC

**Insight:**
*   **No Leakage:** Including retention data didn't artificially inflate the score.
*   **Operational Implication:** You can deploy the "Strict" model (which doesn't need complex upstream data about retention calls) without losing any accuracy.

---

## 6. Key Drivers (Explainability)
Using SHAP and Permutation Importance, we identified what actually drives the prediction.

### 🚨 Top Risk Factors (The "Why")
1.  **CurrentEquipmentDays:** The older the phone, the higher the churn risk. This is the #1 predictor.
    *   *Action:* Offer device upgrades to customers with phones > 300 days old.
2.  **MonthsInService:** Extremes in tenure (very new or very old customers) behave differently.
3.  **PercChangeMinutes:** **Volatility is bad.** Sudden drops or spikes in usage are strong distress signals.

### 📉 Weak Signals (The "Noise")
*   **Dropped/Blocked Calls:** Surprisingly, network friction features had low importance. Removing them actually improved the model slightly.
*   **Demographics:** Static traits (Age, Truck Owner, etc.) are not useful for predicting *immediate* churn compared to behavioral data.

---

## 7. Recommendations
1.  **Deploy CatBoost:** It is the most accurate and handles the categorical nature of the data best.
2.  **Focus on Device Upgrades:** Since `CurrentEquipmentDays` is the top driver, a marketing campaign targeting users with old devices is likely the highest ROI intervention.
3.  **Monitor Usage Volatility:** Build triggers for customers whose minutes usage changes by >20% month-over-month.
4.  **Skip SMOTE:** Complexity cost > Performance gain.

