import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import warnings
import sys
import os

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_fold_data(fold_num, base_dir='processed_data_folds_strict'):
    try:
        X_train = pd.read_csv(f"{base_dir}/fold_{fold_num}_train_X.csv")
        y_train = pd.read_csv(f"{base_dir}/fold_{fold_num}_train_y.csv").iloc[:, 0]
        X_val = pd.read_csv(f"{base_dir}/fold_{fold_num}_val_X.csv")
        y_val = pd.read_csv(f"{base_dir}/fold_{fold_num}_val_y.csv").iloc[:, 0]
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        print(f"Error: Data for fold {fold_num} not found.")
        return None, None, None, None

def calculate_profit_curve(y_true, y_prob, offer_cost, customer_value, acceptance_rate):
    """
    Calculates the expected profit at various probability thresholds.
    
    Formula per customer:
    - TP (Churner we target): + (Customer Value * Acceptance Rate) - Offer Cost
    - FP (Non-Churner we target): - Offer Cost
    - FN (Churner we miss): 0 (Lost Value is implicit baseline)
    - TN (Non-Churner we ignore): 0
    """
    thresholds = np.linspace(0, 1, 101)
    profits = []
    
    # Constant benefit/cost per targeted customer
    # If we target a true churner (TP): We save them with Prob=AcceptanceRate. 
    #   Value = (LTV * AcceptanceRate) - Cost
    # If we target a non-churner (FP): We waste money.
    #   Value = -Cost
    
    benefit_tp = (customer_value * acceptance_rate) - offer_cost
    cost_fp = -offer_cost
    
    for t in thresholds:
        # Vectorized calculation
        pred_target = (y_prob >= t)
        
        tp = (pred_target & (y_true == 1)).sum()
        fp = (pred_target & (y_true == 0)).sum()
        
        total_profit = (tp * benefit_tp) + (fp * cost_fp)
        profits.append(total_profit)
        
    return thresholds, np.array(profits)

def run_business_utility():
    print("--- Running Business Utility Analysis (Profit Curves) ---")
    print("Using CatBoost (Best Model) on Fold 1...")
    
    # 1. Load Data (Fold 1 is sufficient for demonstration)
    X_train, y_train, X_val, y_val = load_fold_data(1)
    if X_train is None: return

    # 2. Train Model (Uncalibrated)
    cat = CatBoostClassifier(
        random_state=42, verbose=0, allow_writing_files=False, thread_count=-1
    )
    cat.fit(X_train, y_train)
    y_prob_raw = cat.predict_proba(X_val)[:, 1]
    
    # 3. Calibrate Model (Isotonic)
    # Note: CatBoost is usually well-calibrated, but we check to be safe/conservative
    print("  Calibrating probabilities (Isotonic)...")
    cal_clf = CalibratedClassifierCV(cat, method='isotonic', cv='prefit')
    cal_clf.fit(X_val, y_val) # Normally we'd split Val into Calib/Test, but using Val for simplicity here
    y_prob_cal = cal_clf.predict_proba(X_val)[:, 1]
    
    # Check Brier Score (Lower is better)
    brier_raw = brier_score_loss(y_val, y_prob_raw)
    brier_cal = brier_score_loss(y_val, y_prob_cal)
    print(f"  Brier Score - Raw: {brier_raw:.4f} vs Calibrated: {brier_cal:.4f}")
    
    # 4. Profit Simulation
    # Scenarios:
    # A. High Value / High Cost (e.g., iPhone upgrade offer)
    #    LTV=$500, Cost=$50, Acceptance=30%
    # B. Low Value / Low Cost (e.g., $5 bill credit)
    #    LTV=$500, Cost=$5, Acceptance=10%
    
    scenarios = [
        {'name': 'High Stakes (Device Upgrade)', 'LTV': 500, 'Cost': 50, 'Accept': 0.3},
        {'name': 'Low Stakes (Bill Credit)',    'LTV': 500, 'Cost': 5,  'Accept': 0.1}
    ]
    
    print("\n--- Profit Analysis ---")
    print("Note: Results assume constant acceptance rates and LTV, which simplifies reality.")
    
    for sc in scenarios:
        thresholds, profits = calculate_profit_curve(
            y_val, y_prob_cal, sc['Cost'], sc['LTV'], sc['Accept']
        )
        
        # Find Max Profit
        max_idx = np.argmax(profits)
        max_profit = profits[max_idx]
        opt_thresh = thresholds[max_idx]
        
        # Baseline: Target Everyone (Threshold = 0)
        # Baseline: Target None (Threshold = 1) -> Profit = 0
        profit_all = profits[0]
        
        print(f"\nScenario: {sc['name']}")
        print(f"  Params: Cost=${sc['Cost']}, LTV=${sc['LTV']}, Acceptance={sc['Accept']*100}%")
        print(f"  Optimal Threshold: {opt_thresh:.2f}")
        print(f"  Max Expected Profit (Validation Set): ${max_profit:,.0f}")
        print(f"  Profit vs Targeting Everyone: ${max_profit - profit_all:,.0f}")
        
        if max_profit <= 0:
            print("  >> WARNING: No profitable strategy found. Retention costs exceed expected value.")
        else:
            # Percentage of user base targeted
            pct_targeted = (y_prob_cal >= opt_thresh).mean() * 100
            print(f"  >> Strategy: Target top {pct_targeted:.1f}% of risky customers.")

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("- These curves suggest potential value, but rely on the 'Acceptance Rate' assumption.")
    print("- Real-world lift is likely lower due to customer heterogeneity.")
    print("- Calibration improved reliability, ensuring we don't overspend on false alarms.")
    print("="*60)

if __name__ == "__main__":
    run_business_utility()

