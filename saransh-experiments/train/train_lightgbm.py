import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
import warnings

warnings.filterwarnings('ignore')

def run_lightgbm():
    print(f"--- Training LightGBM ---")
    # Adjusted path to point to parent directory
    base_dir = 'saransh-experiments/processed_data_folds_strict'
    folds = range(1, 6)
    auc_scores = []
    pr_scores = []
    recall_scores = []

    for fold in folds:
        try:
            X_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_X.csv")
            y_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_y.csv").iloc[:, 0]
            X_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_X.csv")
            y_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_y.csv").iloc[:, 0]

            # LightGBM
            model = lgb.LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            auc = roc_auc_score(y_val, y_prob)
            pr = average_precision_score(y_val, y_prob)
            # Recall at Top 20%
            sorted_indices = np.argsort(y_prob)[::-1]
            y_val_sorted = y_val.iloc[sorted_indices].values
            k = int(len(y_val) * 0.2)
            total_churners = y_val.sum()
            captured_churners = y_val_sorted[:k].sum()
            recall_top20 = captured_churners / total_churners
            
            auc_scores.append(auc)
            pr_scores.append(pr)
            recall_scores.append(recall_top20)
            print(f"  Fold {fold}: AUC={auc:.4f}, PR={pr:.4f}, Recall@Top20%={recall_top20:.4f}")
            
        except FileNotFoundError:
            print(f"  Fold {fold}: Data not found at {base_dir}")

    if auc_scores:
        print(f"LightGBM Average: AUC={np.mean(auc_scores):.4f}, PR={np.mean(pr_scores):.4f}, Recall={np.mean(recall_scores):.4f}")
        return np.mean(auc_scores)
    return 0

if __name__ == "__main__":
    run_lightgbm()
