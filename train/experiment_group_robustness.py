import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OrdinalEncoder
import warnings

warnings.filterwarnings('ignore')

def run_group_robustness():
    print("--- Running Group Robustness (ServiceArea Split) ---")
    
    import sys
    sys.path.insert(0, '.')
    from churn_pipeline_saransh import ChurnPipeline
    
    pipeline = ChurnPipeline(strict_mode=True)
    
    # Load Data
    try:
        df = pipeline.load_data('cell2celltrain.csv')
    except FileNotFoundError:
        print("Error: cell2celltrain.csv not found.")
        return

    print("  Cleaning and Engineering...")
    df_eng = pipeline.clean_and_engineer(df)
    
    # Ensure ServiceArea is usable as a group (must be non-null)
    # Fill NA ServiceArea with 'Unknown'
    if 'ServiceArea' in df_eng.columns:
        df_eng['ServiceArea'] = df_eng['ServiceArea'].fillna('Unknown')
    else:
        print("Error: ServiceArea column missing.")
        return

    # Prepare X and y
    target_col = 'Churn'
    y = df_eng[target_col]
    groups = df_eng['ServiceArea']
    X = df_eng.drop(columns=[target_col, 'CustomerID'], errors='ignore')
    
    # We need to process (encode) features. 
    # Since we are doing a custom split loop, we will do simple processing for speed
    # Just Ordinal Encode categoricals for XGBoost
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = X[col].astype(str)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[col] = oe.fit_transform(X[[col]])
    
    # Loop
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    pr_scores = []
    
    print("\n  Starting Group CV Loop...")
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train
        model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        pr = average_precision_score(y_val, y_prob)
        
        auc_scores.append(auc)
        pr_scores.append(pr)
        
        print(f"  Fold {fold+1}: AUC={auc:.4f}, PR={pr:.4f} (Val Size: {len(y_val)})")
        
    print("\n" + "="*60)
    print("GROUP ROBUSTNESS RESULTS (Split by ServiceArea)")
    print("="*60)
    print(f"Average AUC: {np.mean(auc_scores):.4f}")
    print(f"Average PR-AUC: {np.mean(pr_scores):.4f}")
    print("-" * 60)
    print("Comparison to Random Split (Baseline XGBoost ~0.6560):")
    delta = np.mean(auc_scores) - 0.6560
    print(f"Delta: {delta:+.4f}")
    if delta < -0.02:
        print(">> WARNING: Significant drop. Model relies on region-specific patterns that don't generalize.")
    else:
        print(">> SUCCESS: Model generalizes well across different regions.")

if __name__ == "__main__":
    run_group_robustness()

