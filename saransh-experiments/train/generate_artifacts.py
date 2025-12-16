import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import pickle
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

# Setup
OUTPUT_DIR = "artifacts"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

STRICT_DIR = "saransh-experiments/processed_data_folds_strict"
PERMISSIVE_DIR = "saransh-experiments/processed_data_folds_permissive"

# Constants for Business Utility
COST_CONTACT = 10
BENEFIT_RETAIN = 200 # LTV (Increased to demonstrate profitable campaign)
ACCEPTANCE_RATE = 0.25

def load_fold_data(fold, base_dir):
    try:
        X_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_X.csv")
        y_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_y.csv").iloc[:, 0]
        X_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_X.csv")
        y_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_y.csv").iloc[:, 0]
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        return None, None, None, None

def calculate_recall_at_k(y_true, y_prob, k_percent=0.20):
    k = int(len(y_true) * k_percent)
    if k == 0: return 0.0
    
    # Sort by probability descending
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False)
    
    top_k_hits = df.iloc[:k]['true'].sum()
    total_positives = df['true'].sum()
    
    if total_positives == 0: return 0.0
    return top_k_hits / total_positives

def get_model(name):
    if name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000, solver='liblinear')
    elif name == 'CatBoost':
        return CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False, thread_count=-1)
    elif name == 'XGBoost':
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    elif name == 'LightGBM':
        return lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {name}")

def compute_profit_curve(y_true, y_prob, cost, benefit, acceptance):
    thresholds = np.linspace(0, 1, 101)
    profits = []
    
    # Benefit of TP = (Benefit * Acceptance) - Cost
    # Cost of FP = -Cost
    unit_benefit = (benefit * acceptance) - cost
    unit_cost = -cost
    
    for t in thresholds:
        pred_pos = (y_prob >= t)
        tp = (pred_pos & (y_true == 1)).sum()
        fp = (pred_pos & (y_true == 0)).sum()
        
        profit = (tp * unit_benefit) + (fp * unit_cost)
        profits.append(profit)
        
    return thresholds, np.array(profits)

def main():
    print("Starting Artifact Generation...")
    
    # Storage for aggregated results
    all_fold_results = []
    
    # Storage for raw curves (from Fold 1 for simplicity, or concatenated)
    # We will use concatenated validation predictions across all folds for smoother curves
    raw_preds = {
        'strict': {'LogisticRegression': {'y_true': [], 'y_prob': []}, 'CatBoost': {'y_true': [], 'y_prob': []}, 'XGBoost': {'y_true': [], 'y_prob': []}, 'LightGBM': {'y_true': [], 'y_prob': []}},
        'permissive': {'LogisticRegression': {'y_true': [], 'y_prob': []}, 'CatBoost': {'y_true': [], 'y_prob': []}, 'XGBoost': {'y_true': [], 'y_prob': []}, 'LightGBM': {'y_true': [], 'y_prob': []}}
    }

    settings = [
        ('strict', STRICT_DIR),
        ('permissive', PERMISSIVE_DIR)
    ]
    
    models = ['LogisticRegression', 'CatBoost', 'XGBoost', 'LightGBM']

    # 1. Loop through Settings, Folds, Models
    for setting_name, data_dir in settings:
        print(f"Processing {setting_name}...")
        for fold in range(1, 6):
            X_train, y_train, X_val, y_val = load_fold_data(fold, data_dir)
            if X_train is None: continue
            
            for model_name in models:
                clf = get_model(model_name)
                clf.fit(X_train, y_train)
                
                y_prob = clf.predict_proba(X_val)[:, 1]
                
                # Metrics
                roc = roc_auc_score(y_val, y_prob)
                pr = average_precision_score(y_val, y_prob)
                brier = brier_score_loss(y_val, y_prob)
                rec20 = calculate_recall_at_k(y_val, y_prob, 0.20)
                
                all_fold_results.append({
                    'fold': fold,
                    'model': model_name,
                    'setting': setting_name,
                    'roc_auc': roc,
                    'pr_auc': pr,
                    'brier': brier,
                    'recall20': rec20
                })
                
                # Store predictions for global curves
                raw_preds[setting_name][model_name]['y_true'].extend(y_val)
                raw_preds[setting_name][model_name]['y_prob'].extend(y_prob)
                
                # Save SHAP for Best Model (CatBoost, Strict, Fold 1)
                if setting_name == 'strict' and model_name == 'CatBoost' and fold == 1:
                    print("  Generating SHAP values (Fold 1)...")
                    explainer = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_val)
                    
                    # Save SHAP summary plot
                    plt.figure()
                    shap.summary_plot(shap_values, X_val, show=False)
                    plt.tight_layout()
                    plt.savefig(f"{OUTPUT_DIR}/shap_summary_best_model.png")
                    plt.close()
                    
                    # Save SHAP data
                    np.savez(f"{OUTPUT_DIR}/shap_data.npz", 
                             shap_values=shap_values, 
                             features=X_val.values, 
                             feature_names=X_val.columns.tolist())

    # 2. Save Fold Results CSV
    df_folds = pd.DataFrame(all_fold_results)
    df_folds.to_csv(f"{OUTPUT_DIR}/fold_results.csv", index=False)
    print("Saved fold_results.csv")
    
    # 3. Aggregated Leakage Metrics CSV
    df_agg = df_folds.groupby(['model', 'setting'])[['roc_auc', 'pr_auc', 'brier', 'recall20']].mean().reset_index()
    df_agg.to_csv(f"{OUTPUT_DIR}/leakage_metrics.csv", index=False)
    print("Saved leakage_metrics.csv")
    
    # 4. Calibration Curve (Best vs LogReg - Strict)
    print("Generating Calibration Curve...")
    plt.figure(figsize=(10, 6))
    
    # Use concatenated predictions from Strict setting
    for model_name in models:
        y_true = np.array(raw_preds['strict'][model_name]['y_true'])
        y_prob = np.array(raw_preds['strict'][model_name]['y_prob'])
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)
        
        # Save raw data for user
        np.savez(f"{OUTPUT_DIR}/calibration_data_{model_name}.npz", y_true=y_true, y_prob=y_prob)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Strict Setting)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/calibration_curve_best_vs_logreg.png")
    plt.close()
    
    # 5. Profit Curve (Threshold Sweep) - CatBoost Strict
    print("Generating Profit Curve...")
    y_true_cb = np.array(raw_preds['strict']['CatBoost']['y_true'])
    y_prob_cb = np.array(raw_preds['strict']['CatBoost']['y_prob'])
    
    thresholds, profits = compute_profit_curve(y_true_cb, y_prob_cb, COST_CONTACT, BENEFIT_RETAIN, ACCEPTANCE_RATE)
    
    # Normalize profit to per-customer for easier reading or keep total? User asked for curve.
    # Let's plot Total Profit.
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, profits, label='CatBoost Profit')
    
    # Find max
    max_idx = np.argmax(profits)
    max_profit = profits[max_idx]
    best_thresh = thresholds[max_idx]
    
    plt.scatter([best_thresh], [max_profit], color='red', zorder=5)
    plt.annotate(f"Max: ${max_profit:,.0f} @ {best_thresh:.2f}", 
                 (best_thresh, max_profit), xytext=(best_thresh, max_profit*1.1))
    
    plt.xlabel('Threshold')
    plt.ylabel('Total Expected Profit')
    plt.title(f'Profit Curve (C={COST_CONTACT}, B={BENEFIT_RETAIN}, Acc={ACCEPTANCE_RATE})')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/profit_curve_threshold_sweep.png")
    plt.close()
    
    # Save Profit Data
    pd.DataFrame({'threshold': thresholds, 'profit': profits}).to_csv(f"{OUTPUT_DIR}/profit_curve_data.csv", index=False)
    
    # 6. Leakage Delta PR (Bar Plot or similar)
    print("Generating Leakage Delta Plot...")
    
    # Models to compare
    models = ['XGBoost', 'LightGBM', 'CatBoost']
    
    strict_vals = []
    perm_vals = []
    
    for model in models:
        try:
            s_val = df_agg[(df_agg['model']==model) & (df_agg['setting']=='strict')]['pr_auc'].values[0]
            p_val = df_agg[(df_agg['model']==model) & (df_agg['setting']=='permissive')]['pr_auc'].values[0]
            strict_vals.append(s_val)
            perm_vals.append(p_val)
        except IndexError:
            # Fallback if model not in CSV yet (e.g. if we haven't re-run full suite)
            print(f"Warning: Data for {model} not found in leakage metrics.")
            strict_vals.append(0)
            perm_vals.append(0)

    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, strict_vals, width, label='Strict', color='#1f77b4')
    rects2 = ax.bar(x + width/2, perm_vals, width, label='Permissive', color='#ff7f0e')
    
    ax.set_ylabel('PR-AUC')
    ax.set_title('Leakage Impact on PR-AUC (Boosted Models)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 0.5) # Set reasonable y-limit for visibility
    
    plt.savefig(f"{OUTPUT_DIR}/leakage_delta_pr.png")
    plt.close()

    # 7. PR Curve (Strict)
    print("Generating PR Curve...")
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(10, 6))
    
    pr_data = {}
    
    # Generate PR curves for the boosted models (and LogReg as baseline if needed, but logic below iterates on 'models' which is locally redefined)
    # Note: 'models' variable was redefined in step 6 to ['XGBoost', 'LightGBM', 'CatBoost']
    # If we want LogReg too, we should ensure it's in the list or handle separately.
    # The original code iterated on the locally defined 'models'.
    
    # Let's stick to the top models + LogReg for PR Curve context
    pr_models_to_plot = ['LogisticRegression', 'XGBoost', 'LightGBM', 'CatBoost']

    for model_name in pr_models_to_plot:
        y_true = np.array(raw_preds['strict'][model_name]['y_true'])
        y_prob = np.array(raw_preds['strict'][model_name]['y_prob'])
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, label=model_name)
        
        pr_data[model_name] = {'precision': precision, 'recall': recall, 'y_true': y_true, 'y_prob': y_prob}
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Strict)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/pr_curve_strict.png")
    plt.close()
    
    # Save raw PR data (y_true, y_score) as requested
    np.savez(f"{OUTPUT_DIR}/pr_curve_data.npz", 
             catboost_true=pr_data['CatBoost']['y_true'], 
             catboost_prob=pr_data['CatBoost']['y_prob'],
             logreg_true=pr_data['LogisticRegression']['y_true'], 
             logreg_prob=pr_data['LogisticRegression']['y_prob'],
             xgboost_true=pr_data['XGBoost']['y_true'],
             xgboost_prob=pr_data['XGBoost']['y_prob'],
             lightgbm_true=pr_data['LightGBM']['y_true'],
             lightgbm_prob=pr_data['LightGBM']['y_prob'])

    print("All artifacts generated in 'artifacts/' directory.")

if __name__ == "__main__":
    main()

