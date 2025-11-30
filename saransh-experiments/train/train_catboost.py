import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def run_catboost():
    print(f"--- Training CatBoost ---")
    base_dir = 'saransh-experiments/processed_data_folds_strict'
    folds = range(1, 6)
    
    metrics = {
        'AUC': [], 'PR': [],
        'Recall_0_Default': [], 'Recall_1_Default': [],
        'Recall_0_Top20': [], 'Recall_1_Top20': []
    }

    for fold in folds:
        try:
            X_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_X.csv")
            y_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_y.csv").iloc[:, 0]
            X_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_X.csv")
            y_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_y.csv").iloc[:, 0]

            # CatBoost
            model = CatBoostClassifier(
                random_state=42, verbose=0, allow_writing_files=False, thread_count=-1
            )
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred_default = model.predict(X_val) # Threshold 0.5
            
            # Top 20% Threshold
            sorted_indices = np.argsort(y_prob)[::-1]
            k = int(len(y_val) * 0.2)
            top_k_indices = sorted_indices[:k]
            y_pred_top20 = np.zeros_like(y_val)
            y_pred_top20[top_k_indices] = 1
            
            # Metrics
            auc = roc_auc_score(y_val, y_prob)
            pr = average_precision_score(y_val, y_prob)
            
            # Confusion Matrix for Default (0.5)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_default).ravel()
            rec_0_def = tn / (tn + fp) # Specificity
            rec_1_def = tp / (tp + fn) # Sensitivity
            
            # Confusion Matrix for Top 20%
            tn2, fp2, fn2, tp2 = confusion_matrix(y_val, y_pred_top20).ravel()
            rec_0_top20 = tn2 / (tn2 + fp2)
            rec_1_top20 = tp2 / (tp2 + fn2)
            
            metrics['AUC'].append(auc)
            metrics['PR'].append(pr)
            metrics['Recall_0_Default'].append(rec_0_def)
            metrics['Recall_1_Default'].append(rec_1_def)
            metrics['Recall_0_Top20'].append(rec_0_top20)
            metrics['Recall_1_Top20'].append(rec_1_top20)
            
            print(f"  Fold {fold}: AUC={auc:.4f}")
            
        except FileNotFoundError:
            print(f"  Fold {fold}: Data not found.")

    # Summary
    print("\n" + "="*60)
    print("CATBOOST DETAILED RECALL (Avg over 5 Folds)")
    print("="*60)
    print(f"{'Metric':<30} | {'Value':<10}")
    print("-" * 60)
    print(f"AUC                            | {np.mean(metrics['AUC']):.4f}")
    print(f"PR-AUC                         | {np.mean(metrics['PR']):.4f}")
    print("-" * 60)
    print("SCENARIO 1: Default Threshold (0.5) -- Conservative")
    print(f"  Recall Class 0 (Stayers)     | {np.mean(metrics['Recall_0_Default']):.2%}")
    print(f"  Recall Class 1 (Churners)    | {np.mean(metrics['Recall_1_Default']):.2%}")
    print("-" * 60)
    print("SCENARIO 2: Target Top 20% -- Aggressive Marketing")
    print(f"  Recall Class 0 (Stayers)     | {np.mean(metrics['Recall_0_Top20']):.2%}")
    print(f"  Recall Class 1 (Churners)    | {np.mean(metrics['Recall_1_Top20']):.2%}")
    print("-" * 60)

if __name__ == "__main__":
    run_catboost()
