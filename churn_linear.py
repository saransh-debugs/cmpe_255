import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import inspect

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from sklearn.model_selection import train_test_split

# ---------------------------
# Paths
# ---------------------------
TRAIN_PATH = "cell2celltrain.csv"
HOLDOUT_PATH = "cell2cellholdout.csv"
PRED_OUT = "churn_predictions_holdout.csv"
COEF_OUT = "sgd_logreg_feature_importance.csv"

# ---------------------------
# Utilities
# ---------------------------
def find_target_column(df):
    candidates = ['Churn', 'churn', 'CHURN']
    for c in df.columns:
        if c in candidates:
            return c
    for c in df.columns:
        if 'churn' in c.lower():
            return c
    binary_like = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    return binary_like[0] if binary_like else df.columns[-1]

def to_binary(series: pd.Series) -> pd.Series:
    if series.dtype.name == "category":
        series = series.astype(str)
    if series.dtype == bool:
        return series.astype(int)
    s = series.astype(str).str.strip().str.lower()
    true_vals = {'yes', 'true', '1', 'churned', 'y', 't'}
    return s.isin(true_vals).astype(int)

def build_preprocessor(X: pd.DataFrame):
    # Split columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Numeric: impute + scale (dense)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ohe_kwargs = {"handle_unknown": "ignore"}
    params = inspect.signature(OneHotEncoder).parameters

    if "sparse_output" in params:
        ohe_kwargs["sparse_output"] = True
    elif "sparse" in params:
        ohe_kwargs["sparse"] = True

    if "min_frequency" in params:
        ohe_kwargs["min_frequency"] = 0.01 

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**ohe_kwargs)),
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    return preprocess, numeric_cols, categorical_cols

# ---------------------------
# Main flow
# ---------------------------
def main(use_exact_logreg=False):
    train_df = pd.read_csv(TRAIN_PATH)
    holdout_df = pd.read_csv(HOLDOUT_PATH)

    target_col = find_target_column(train_df)
    y = to_binary(train_df[target_col])

    id_like = [c for c in train_df.columns
               if any(tok in c.lower() for tok in ['id', 'customer', 'account'])]
    feature_cols = [c for c in train_df.columns if c not in {target_col, *id_like}]
    X = train_df[feature_cols].copy()

    holdout_feature_cols = [c for c in holdout_df.columns if c in feature_cols]
    X_holdout = holdout_df[holdout_feature_cols].copy()

    preprocess, num_cols, cat_cols = build_preprocessor(X)

    if use_exact_logreg:
        clf = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=5000,
            n_jobs=-1
        )
    else:
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=5000,
            early_stopping=True,
            n_iter_no_change=5,
            class_weight="balanced",
            random_state=42
        )

    model = Pipeline([("preprocess", preprocess), ("clf", clf)])

    # Train/valid split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )   

    # Fit
    model.fit(X_tr, y_tr)

    # Validation metrics
    yva_proba = model.predict_proba(X_va)[:, 1]
    yva_pred = (yva_proba >= 0.5).astype(int)

    valid_auc = roc_auc_score(y_va, yva_proba)
    valid_acc = accuracy_score(y_va, yva_pred)
    valid_cm = confusion_matrix(y_va, yva_pred)
    print("\n=== Validation Results ===")
    print(f"AUC:       {valid_auc:.4f}")
    print(f"Accuracy:  {valid_acc:.4f}")
    print("Confusion matrix:\n", valid_cm)
    print("\nClassification report:\n", classification_report(y_va, yva_pred))

    # Plot ROC (validation)
    try:
        RocCurveDisplay.from_predictions(y_va, yva_proba)
        plt.title("Validation ROC Curve")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Skipping ROC plot:", e)

    # Holdout predictions (labels may be missing or single-class)
    yhold_proba = model.predict_proba(X_holdout)[:, 1]
    yhold_pred = (yhold_proba >= 0.5).astype(int)

    has_holdout_label = False
    if target_col in holdout_df.columns:
        y_hold = to_binary(holdout_df[target_col])
        if y_hold.nunique(dropna=True) >= 2:
            has_holdout_label = True

    if has_holdout_label:
        hold_auc = roc_auc_score(y_hold, yhold_proba)
        hold_acc = accuracy_score(y_hold, yhold_pred)
        hold_cm = confusion_matrix(y_hold, yhold_pred)
        print("\n=== Holdout Results ===")
        print(f"AUC:       {hold_auc:.4f}")
        print(f"Accuracy:  {hold_acc:.4f}")
        print("Confusion matrix:\n", hold_cm)
        print("\nClassification report:\n", classification_report(y_hold, yhold_pred))
        try:
            RocCurveDisplay.from_predictions(y_hold, yhold_proba)
            plt.title("Holdout ROC Curve")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print("Skipping holdout ROC plot:", e)
    else:
        print("\n(Info) Holdout labels are missing or single-class; saving predictions only.")

    preds_df = holdout_df.copy()
    preds_df["pred_proba_churn"] = yhold_proba
    preds_df["pred_label"] = yhold_pred
    preds_df.to_csv(PRED_OUT, index=False)
    print(f"\nSaved holdout predictions to: {PRED_OUT}")

    # Coefficients (linear model)
    # Get feature names after preprocessing to align with coefficients
    preprocess_fitted = model.named_steps["preprocess"]
    feature_names = extract_feature_names(preprocess_fitted, num_cols, cat_cols)
    coefs = model.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs}) \
                .sort_values("coefficient", ascending=False)
    coef_df.to_csv(COEF_OUT, index=False)
    print(f"Saved coefficients to: {COEF_OUT}")

    # Show top signals
    top_pos = coef_df.head(15)
    top_neg = coef_df.tail(15)
    print("\nTop positive churn signals:")
    print(top_pos.to_string(index=False))
    print("\nTop negative (retention) signals:")
    print(top_neg.to_string(index=False))

if __name__ == "__main__":
    main(use_exact_logreg=False)
