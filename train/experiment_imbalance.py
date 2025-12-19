import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings

warnings.filterwarnings('ignore')

# --- 1. Focal Loss for XGBoost ---
# Gradient and Hessian needed for custom objective
def focal_binary_object(pred, dtrain):
    gamma = 2.0
    alpha = 0.25
    label = dtrain.get_label()
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    return None 

def load_fold_data(fold_num, base_dir='processed_data_folds_strict'):
    try:
        X_train = pd.read_csv(f"{base_dir}/fold_{fold_num}_train_X.csv")
        y_train = pd.read_csv(f"{base_dir}/fold_{fold_num}_train_y.csv").iloc[:, 0]
        X_val = pd.read_csv(f"{base_dir}/fold_{fold_num}_val_X.csv")
        y_val = pd.read_csv(f"{base_dir}/fold_{fold_num}_val_y.csv").iloc[:, 0]
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        return None, None, None, None

def run_imbalance_experiment():
    print("--- Running Imbalance Handling Experiment ---")
    
    folds = range(1, 6)
    
    strategies = ['Baseline (Unweighted)', 'Class Weighted', 'SMOTE', 'SMOTE+Tomek']
    results = {s: {'AUC': [], 'PR-AUC': []} for s in strategies}
    
    for fold in folds:
        print(f"Processing Fold {fold}...")
        X_train, y_train, X_val, y_val = load_fold_data(fold)
        if X_train is None: continue

        # 1. Baseline (Unweighted)
        model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        results['Baseline (Unweighted)']['AUC'].append(roc_auc_score(y_val, y_prob))
        results['Baseline (Unweighted)']['PR-AUC'].append(average_precision_score(y_val, y_prob))
        
        # 2. Class Weighted (scale_pos_weight)
        # Estimate ratio
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
        model_w = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1,
            scale_pos_weight=ratio
        )
        model_w.fit(X_train, y_train)
        y_prob = model_w.predict_proba(X_val)[:, 1]
        results['Class Weighted']['AUC'].append(roc_auc_score(y_val, y_prob))
        results['Class Weighted']['PR-AUC'].append(average_precision_score(y_val, y_prob))
        
        # 3. SMOTE (Synthetic Minority Over-sampling Technique)
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        model_sm = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1
        )
        model_sm.fit(X_train_sm, y_train_sm)
        y_prob = model_sm.predict_proba(X_val)[:, 1]
        results['SMOTE']['AUC'].append(roc_auc_score(y_val, y_prob))
        results['SMOTE']['PR-AUC'].append(average_precision_score(y_val, y_prob))

        # 4. SMOTE + Tomek (Hybird)
        try:
            smt = SMOTETomek(random_state=42)
            X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)
            
            model_smt = xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1
            )
            model_smt.fit(X_train_smt, y_train_smt)
            y_prob = model_smt.predict_proba(X_val)[:, 1]
            results['SMOTE+Tomek']['AUC'].append(roc_auc_score(y_val, y_prob))
            results['SMOTE+Tomek']['PR-AUC'].append(average_precision_score(y_val, y_prob))
        except Exception as e:
            print(f"  SMOTE+Tomek failed for fold {fold}: {e}")

    # Summary
    print("\n" + "="*60)
    print("IMBALANCE STRATEGY RESULTS (Avg over 5 folds)")
    print("="*60)
    print(f"{'Strategy':<25} | {'ROC-AUC':<8} | {'PR-AUC':<8}")
    print("-" * 60)
    
    for s in strategies:
        auc = np.mean(results[s]['AUC'])
        pr = np.mean(results[s]['PR-AUC'])
        print(f"{s:<25} | {auc:.4f}   | {pr:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    run_imbalance_experiment()

