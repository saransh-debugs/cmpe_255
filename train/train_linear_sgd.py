import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings("ignore")

def run_linear_sgd():
    print(f"--- Training Linear SGD (Calibrated) ---")

    base_dir = "processed_data_folds_strict"
    folds = range(1, 6)

    auc_scores = []
    pr_scores = []
    brier_scores = []
    recall_top20_scores = []
    
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

    for fold in folds:
        try:
            X_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_X.csv")
            y_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_y.csv").iloc[:, 0]
            X_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_X.csv")
            y_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_y.csv").iloc[:, 0]

            # SGD itself doesn't output probabilities nicely unless loss='log_loss', 
            # but usually needs calibration for best Brier score
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SGDClassifier(
                    loss="log_loss", 
                    max_iter=5000,
                    alpha=1e-4,
                    penalty="l2",
                    class_weight="balanced",
                    random_state=42
                ))
            ])
            
            # Fit pipeline
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            # Metrics
            auc = roc_auc_score(y_val, y_prob)
            pr = average_precision_score(y_val, y_prob)
            brier = brier_score_loss(y_val, y_prob)
            
            k = int(len(y_val) * 0.2)
            sorted_idx = np.argsort(y_prob)[::-1]
            top_k_churners = y_val.iloc[sorted_idx[:k]].sum()
            total_churners = y_val.sum()
            recall_top20 = top_k_churners / total_churners if total_churners > 0 else 0

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            total_tp += tp; total_fp += fp; total_tn += tn; total_fn += fn

            auc_scores.append(auc)
            pr_scores.append(pr)
            brier_scores.append(brier)
            recall_top20_scores.append(recall_top20)

            print(f"  Fold {fold}: AUC={auc:.4f}, PR={pr:.4f}, Brier={brier:.4f}, Recall@20%={recall_top20:.4f}")

        except FileNotFoundError:
            print(f"  Fold {fold}: Missing data files.")

    if auc_scores:
        print("\n=== Linear SGD Report ===")
        print(f"ROC-AUC:      {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
        print(f"PR-AUC:       {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f}")
        print(f"Brier Score:  {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")
        print(f"Recall@20%:   {np.mean(recall_top20_scores):.4f} ± {np.std(recall_top20_scores):.4f}")
        print("\nAggregated Confusion Matrix (Threshold 0.5):")
        print(f"[[{total_tn}  {total_fp}]\n [{total_fn}  {total_tp}]]")

if __name__ == "__main__":
    run_linear_sgd()
