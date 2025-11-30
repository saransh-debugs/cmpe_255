import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import warnings
import sys
import os

warnings.filterwarnings('ignore')

def load_data():
    # Load Fold 1 only for speed during optimization
    base_dir = 'saransh-experiments/processed_data_folds_strict'
    try:
        X_train = pd.read_csv(f"{base_dir}/fold_1_train_X.csv")
        y_train = pd.read_csv(f"{base_dir}/fold_1_train_y.csv").iloc[:, 0]
        X_val = pd.read_csv(f"{base_dir}/fold_1_val_X.csv")
        y_val = pd.read_csv(f"{base_dir}/fold_1_val_y.csv").iloc[:, 0]
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        print("Data not found.")
        return None, None, None, None

def objective(trial):
    X_train, y_train, X_val, y_val = load_data()
    if X_train is None: return 0

    # Hyperparameter Search Space
    params = {
        'iterations': 500, # Fixed for speed, can be higher in final
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 4.0), # Handle Imbalance
        'verbose': 0,
        'thread_count': -1,
        'random_state': 42,
        'allow_writing_files': False
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Optimization Target: Recall @ Top 20%
    sorted_indices = np.argsort(y_prob)[::-1]
    k = int(len(y_val) * 0.2)
    top_k_indices = sorted_indices[:k]
    
    # Captured churners in top k
    captured = y_val.iloc[top_k_indices].sum()
    total = y_val.sum()
    recall_top20 = captured / total
    
    return recall_top20

def run_optimization():
    print("--- Running CatBoost Hyperparameter Optimization (Optuna) ---")
    print("Target Metric: Recall @ Top 20% (Lift)")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # 20 trials for speed
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Recall @ Top 20%: {study.best_value:.4f} (Baseline was ~0.33)")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    # Retrain with best params on Fold 1 to confirm
    print("\nVerifying on Fold 1...")
    X_train, y_train, X_val, y_val = load_data()
    
    best_params = study.best_params
    best_params.update({
        'iterations': 1000, # Boost iterations for final run
        'verbose': 0,
        'thread_count': -1, 
        'random_state': 42,
        'allow_writing_files': False
    })
    
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Calc Final Metrics
    auc = roc_auc_score(y_val, y_prob)
    
    sorted_indices = np.argsort(y_prob)[::-1]
    k = int(len(y_val) * 0.2)
    recall_top20 = y_val.iloc[sorted_indices[:k]].sum() / y_val.sum()
    
    print(f"Final Verification -> AUC: {auc:.4f}, Recall@Top20%: {recall_top20:.4f}")

if __name__ == "__main__":
    run_optimization()

