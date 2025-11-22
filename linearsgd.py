import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, RocCurveDisplay
)

def find_target_col(df):
    for c in df.columns:
        if c == "Churn":
            return c
    for c in df.columns:
        if "churn" in c.lower():
            return c
    raise ValueError("No churn-like target column found.")

def to_binary(series: pd.Series) -> pd.Series:
    if series.dtype.name == "category":
        series = series.astype(str)
    if series.dtype == bool:
        return series.astype(int)
    s = series.astype(str).str.strip().str.lower()
    true_vals = {"yes", "true", "1", "churned", "y", "t"}
    return s.isin(true_vals).astype(int)

train_path = "cell2celltrain.csv"
holdout_path = "cell2cellholdout.csv"

train_data = pd.read_csv(train_path)
holdout_data = pd.read_csv(holdout_path)

target = find_target_col(train_data)
y = to_binary(train_data[target])

id_like = [c for c in train_data.columns if "customerid" in c.lower()]
numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
features = [f for f in numeric_features if f not in id_like and f != target]

X = train_data[features]
X_holdout = holdout_data[[c for c in features if c in holdout_data.columns]]

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

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

val_probs = pipeline.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.5).astype(int)

print("\n=== Validation Results ===")
print(f"AUC:       {roc_auc_score(y_val, val_probs):.4f}")
print(f"Accuracy:  {accuracy_score(y_val, val_preds):.4f}")
print("Confusion matrix:\n", confusion_matrix(y_val, val_preds))
print("\nClassification report:\n", classification_report(y_val, val_preds))

RocCurveDisplay.from_predictions(y_val, val_probs)
plt.title("Validation ROC Curve")
plt.tight_layout()
plt.show()

holdout_probs = pipeline.predict_proba(X_holdout)[:, 1]
holdout_preds = (holdout_probs >= 0.5).astype(int)

cust_id_col = [c for c in holdout_data.columns if "customerid" in c.lower()][0]
out = holdout_data[[cust_id_col]].copy()
out["prob"] = holdout_probs
out["pred"] = holdout_preds
out.to_csv("holdout_predictions_numeric_only.csv", index=False)

print("\nSaved: holdout_predictions_numeric_only.csv  Columns:", list(out.columns))
