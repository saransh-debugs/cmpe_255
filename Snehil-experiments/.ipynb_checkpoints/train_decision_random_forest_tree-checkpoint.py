import warnings

# STEP 1 – Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    precision_recall_curve,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")


def run_tree_random_forest():
    print("--- Training Decision Tree & Random Forest ---")

    # STEP 2 – Load the training dataset
    df = pd.read_csv("cell2celltrain.csv")

    # Quick checks
    print(df.shape)
    print(df.columns)
    df.head()

    # STEP 3 – Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    print(df.isnull().sum().sum())

    # STEP 4 – Encode categorical columns
    encoder = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    print(df.dtypes.head())

    # STEP 5 – Define features (X) and target (y) and split

    # Target column
    target_col = "Churn"

    # Columns to drop
    drop_cols = [
        "CustomerID",
        "ServiceArea",
        "Occupation",
        "PrizmCode",
        "NewCellphoneUser"
    ]

    # Features and target
    X = df.drop(columns=[target_col] + [c for c in drop_cols if c in df.columns])
    y = df[target_col]

    # Train_test split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(X_train.shape, X_val.shape)

    # Initialize models
    dt = DecisionTreeClassifier(random_state=42)

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    # Train models
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # STEP 7 – Evaluation: numeric metrics
    def evaluate_model(name, model, X_val, y_val):
        y_pred = model.predict(X_val)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_proba = None

        print(f"\n=== {name} ===")
        print("Accuracy:", round(accuracy_score(y_val, y_pred), 4))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
        print("Classification Report:\n", classification_report(y_val, y_pred))

        if y_proba is not None:
            print("ROC AUC:", round(roc_auc_score(y_val, y_proba), 4))
            print("Log Loss:", round(log_loss(y_val, y_proba), 4))

        return y_pred, y_proba

    dt_pred, dt_proba = evaluate_model("Decision Tree", dt, X_val, y_val)
    rf_pred, rf_proba = evaluate_model("Random Forest", rf, X_val, y_val)

    # Improve churn detection using lower threshold
    threshold = 0.30
    rf_pred_low = (rf_proba > threshold).astype(int)

    print("\n=== Random Forest (Lower Threshold) ===")
    print("Threshold:", threshold)
    print(confusion_matrix(y_val, rf_pred_low))

    # STEP 8 – Visual 1: Confusion matrices
    def plot_confusion(cm, title):
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title(title)
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    dt_cm = confusion_matrix(y_val, dt_pred)
    rf_cm_low = confusion_matrix(y_val, rf_pred_low)

    plot_confusion(dt_cm, "Decision Tree - Confusion Matrix")
    plot_confusion(rf_cm_low, "Random Forest - Confusion Matrix")

    # STEP 9 – Visual 2: Feature importance
    def plot_feature_importance(model, feature_names, title, top_k=15):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_k]
        feats = np.array(feature_names)[idx]
        vals = importances[idx]

        fig = plt.figure(figsize=(7, 5))
        y_pos = np.arange(len(feats))
        plt.barh(y_pos, vals)
        plt.yticks(y_pos, feats)
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    plot_feature_importance(dt, X.columns, "Decision Tree - Top 15 Features")
    plot_feature_importance(rf, X.columns, "Random Forest - Top 15 Features")

    # STEP 10 – ROC & Precision–Recall curves (Random Forest)

    # ROC Curve for Random Forest
    fpr, tpr, _ = roc_curve(y_val, rf_proba)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Random Forest - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Precision–Recall Curve for Random Forest
    prec, rec, _ = precision_recall_curve(y_val, rf_proba)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(rec, prec)
    plt.title("Random Forest - Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.show()

    return {
        "dt_pred": dt_pred,
        "dt_proba": dt_proba,
        "rf_pred": rf_pred,
        "rf_proba": rf_proba,
        "rf_pred_low": rf_pred_low
    }


if __name__ == "__main__":
    run_tree_random_forest()
