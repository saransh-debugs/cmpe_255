import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings('ignore')

def load_fold_data(fold_num, base_dir):
    """Loads X and y for train and val for a specific fold."""
    try:
        X_train = pd.read_csv(f"{base_dir}/fold_{fold_num}_train_X.csv")
        y_train = pd.read_csv(f"{base_dir}/fold_{fold_num}_train_y.csv").iloc[:, 0]
        X_val = pd.read_csv(f"{base_dir}/fold_{fold_num}_val_X.csv")
        y_val = pd.read_csv(f"{base_dir}/fold_{fold_num}_val_y.csv").iloc[:, 0]
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        return None, None, None, None

def train_evaluate(model, X_train, y_train, X_val, y_val):
    """Trains a model and returns probas and metrics."""
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    pr_auc = average_precision_score(y_val, y_pred_prob)
    
    return roc_auc, pr_auc

def run_experiment(data_dir, setting_name):
    print(f"\nRunning Experiment: {setting_name}")
    print(f"Data Source: {data_dir}")
    
    folds = range(1, 6)
    results = []
    
    for fold in folds:
        X_train, y_train, X_val, y_val = load_fold_data(fold, data_dir)
        if X_train is None:
            continue
            
        # Models to benchmark
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, solver='liblinear'),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        }
        
        fold_res = {'Fold': fold}
        
        for name, model in models.items():
            roc, pr = train_evaluate(model, X_train, y_train, X_val, y_val)
            fold_res[f'{name}_ROC'] = roc
            fold_res[f'{name}_PR'] = pr
            
        results.append(fold_res)
        print(f"  Fold {fold} processed.")
        
    # Aggregating
    df_res = pd.DataFrame(results)
    avg_res = df_res.mean(numeric_only=True)
    
    print(f"\n--- Results ({setting_name}) ---")
    for name in ['LogisticRegression', 'RandomForest', 'XGBoost']:
        print(f"{name}: ROC-AUC={avg_res[f'{name}_ROC']:.4f}, PR-AUC={avg_res[f'{name}_PR']:.4f}")
        
    return avg_res

def main():
    # 1. Strict View (No Leakage)
    res_strict = run_experiment('processed_data_folds_strict', 'STRICT (No Leakage)')
    
    # 2. Permissive View (With Leakage)
    res_permissive = run_experiment('processed_data_folds_permissive', 'PERMISSIVE (With Leakage)')
    
    # 3. Comparison
    print("\n" + "="*60)
    print("LEAKAGE IMPACT ANALYSIS (Strict vs Permissive)")
    print("="*60)
    print(f"{'Model':<20} | {'View':<12} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Lift (ROC)':<10}")
    print("-" * 75)
    
    for name in ['LogisticRegression', 'RandomForest', 'XGBoost']:
        roc_strict = res_strict[f'{name}_ROC']
        roc_perm = res_permissive[f'{name}_ROC']
        lift = roc_perm - roc_strict
        
        print(f"{name:<20} | {'Strict':<12} | {roc_strict:.4f}   | {res_strict[f'{name}_PR']:.4f}   | -")
        print(f"{'':<20} | {'Permissive':<12} | {roc_perm:.4f}   | {res_permissive[f'{name}_PR']:.4f}   | {lift:+.4f}")
        print("-" * 75)

if __name__ == "__main__":
    main()
