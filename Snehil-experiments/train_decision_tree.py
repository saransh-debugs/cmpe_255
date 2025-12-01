import warnings

# STEP 1 – Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)

warnings.filterwarnings("ignore")

def run_decision_tree():
    print("--- Training Decision Tree ---")

    # STEP 2 – Load the training dataset
    df = pd.read_csv("cell2celltrain.csv")

    print(df.shape)
    print(df.columns)
    df.head()

    # STEP 3 – Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    print("Total missing values after imputation:", df.isnull().sum().sum())

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

    print("Train / Val shapes:", X_train.shape, X_val.shape)

    # Initialize Decision Tree
    dt = DecisionTreeClassifier(random_state=42)

    # Train model
    dt.fit(X_train, y_train)

    # Evaluation: numeric metrics
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

    # STEP 8 – Visual 1: Confusion matrix
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
    plot_confusion(dt_cm, "Decision Tree - Confusion Matrix")

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

    return {
        "dt_model": dt,
        "dt_pred": dt_pred,
        "dt_proba": dt_proba
    }


if __name__ == "__main__":
    run_decision_tree()
