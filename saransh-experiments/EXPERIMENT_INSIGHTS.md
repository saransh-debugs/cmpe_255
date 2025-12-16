# Experiment Summary & Insights: Leakage-Aware Churn Prediction on Cell2Cell

## Experiment Overview
We designed and executed a **leakage-aware, explainable churn prediction pipeline** on the Cell2Cell dataset to assess model performance, stability, and the impact of intervention-related features.

*   **Objective:** Benchmark churn prediction models while controlling for data leakage from retention interventions.
*   **Models:** Logistic Regression, Random Forest, XGBoost.
*   **Validation Strategy:** 5-Fold Stratified Cross-Validation.
*   **Feature Engineering:** Target Encoding (High Cardinality), Log1p (Skewed Usage), Winsorization (Outliers).
*   **Leakage Control:**
    *   **Strict View:** Excluded post-risk intervention features (`MadeCallToRetentionTeam`, `RetentionCalls`, `RetentionOffersAccepted`).
    *   **Permissive View:** Included these features to establish an upper-bound on performance.

---

## Key Results & Insights

### 1. Leakage Analysis (Strict vs. Permissive)
*   **Finding:** There is **negligible performance difference** between Strict and Permissive models.
    *   **XGBoost Lift:** -0.0011 AUC (Permissive actually slightly lower, likely due to noise).
    *   **Random Forest Lift:** +0.0048 AUC (Marginal gain).
*   **Insight:** Retention variables (`MadeCallToRetentionTeam`, etc.) do not artificially inflate predictive power in this dataset. The "Strict" model is robust and safe for deployment without sacrificing accuracy.

### 2. Model Performance Benchmarks (Strict View)
*   **Best Performer:** **XGBoost** (AUC: **0.6560**).
*   **Runner Up:** Random Forest (AUC: 0.6488).
*   **Baseline:** Logistic Regression (AUC: 0.6139).
*   **Insight:** Tree-based models significantly outperform linear baselines, capturing non-linear interactions in customer behavior (e.g., usage volatility vs. tenure).

### 3. Explainability & Drivers (SHAP & Permutation)
*   **Dominant Drivers (Lifecycle):**
    *   `CurrentEquipmentDays` (Device Age): The strongest predictor. Older devices strongly correlate with churn.
    *   `MonthsInService` (Tenure): Long-time customers show distinct stability patterns.
*   **Secondary Drivers (Usage Dynamics):**
    *   `PercChangeMinutes`: Volatility in usage (sudden drops/spikes) is a critical risk signal.
    *   `MonthlyMinutes`: Absolute usage levels matter.
*   **Surprising Non-Factors:**
    *   **Friction:** `DroppedCalls` and `BlockedCalls` were **not** primary drivers.
    *   **Demographics:** Static attributes (Age, Occupation) contributed little signal compared to behavioral metrics.

### 4. Feature Family Ablation (What Matters?)
We removed entire groups of features to test their necessity.
*   **Lifecycle Features:** Removing these caused a **massive drop (-0.0445 AUC)**. The model collapses without tenure/device data.
*   **Usage & Revenue:** Removing these caused a significant drop (-0.0299 AUC).
*   **Friction & Demographics:** Removing these actually **improved** the model slightly (+0.0045 AUC).
*   **Insight:** The model is effectively a "Lifecycle & Usage" predictor. Collecting cleaner friction data or more granular demographic data might not yield ROI compared to better device/tenure tracking.

### 5. Calibration & Reliability
*   **Brier Score:** 0.1981 (Lower is better).
*   **Curve Analysis:** The model is **over-confident at high probabilities**.
    *   *Example:* When predicting **93%** risk, the actual observed churn rate was only **83%**.
*   **Action:** Raw probabilities should be calibrated (e.g., Isotonic Regression) before being used in financial "Expected Value" calculations to avoid over-spending on retention for false alarms.

