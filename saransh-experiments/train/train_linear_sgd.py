import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

warnings.filterwarnings("ignore")


def run_linear_sgd():
    print(f"--- Training Linear SGD ---")

    base_dir = "../processed_data_folds_strict"
    folds = range(1, 6)

    auc_scores = []
    pr_scores = []
    recall_scores = []

    for fold in folds:
        try:
            # Load data
            X_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_X.csv")
            y_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_y.csv").iloc[:, 0]

            X_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_X.csv")
            y_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_y.csv").iloc[:, 0]

            # Linear SGD Pipeline
            pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SGDClassifier(
                    loss="log_loss",
                    max_iter=5000,
                    alpha=1e-4,
                    early_stopping=True,
                    n_iter_no_change=5,
                    penalty="l2",
                    class_weight="balanced",
                    random_state=42
                ))
            ])

            # Train model
            pipeline.fit(X_train, y_train)

            # Predict probabilities
            y_prob = pipeline.predict_proba(X_val)[:, 1]

            # Metrics
            auc = roc_auc_score(y_val, y_prob)
            pr = average_precision_score(y_val, y_prob)

            # Recall@Top20%
            sorted_idx = np.argsort(y_prob)[::-1]
            y_sorted = y_val.iloc[sorted_idx].values
            k = int(len(y_val) * 0.2)

            total_churners = y_val.sum()
            captured_churners = y_sorted[:k].sum()

            recall_top20 = captured_churners / total_churners if total_churners > 0 else 0

            auc_scores.append(auc)
            pr_scores.append(pr)
            recall_scores.append(recall_top20)

            print(
                f"  Fold {fold}: "
                f"AUC={auc:.4f}, "
                f"PR={pr:.4f}, "
                f"Recall@Top20%={recall_top20:.4f}"
            )

        except FileNotFoundError:
            print(f"  Fold {fold}: Missing data files.")

    # Final averages
    if auc_scores:
        print("\n--- Linear SGD Final Averages (Strict Folds) ---")
        print(f"AUC:          {np.mean(auc_scores):.4f}")
        print(f"PR AUC:       {np.mean(pr_scores):.4f}")
        print(f"Recall@20%:   {np.mean(recall_scores):.4f}")

    return np.mean(auc_scores) if auc_scores else 0


if __name__ == "__main__":
    run_linear_sgd()