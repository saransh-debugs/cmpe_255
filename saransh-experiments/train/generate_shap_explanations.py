import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

def generate_shap_plots():
    print("--- Generating SHAP Plots for Explainability ---")
    
    # Create output directory
    output_dir = "saransh-experiments/shap_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ONE fold of data (Fold 1) for visualization
    base_dir = 'saransh-experiments/processed_data_folds_strict'
    try:
        X_train = pd.read_csv(f"{base_dir}/fold_1_train_X.csv")
        y_train = pd.read_csv(f"{base_dir}/fold_1_train_y.csv").iloc[:, 0]
        X_val = pd.read_csv(f"{base_dir}/fold_1_val_X.csv")
        y_val = pd.read_csv(f"{base_dir}/fold_1_val_y.csv").iloc[:, 0]
    except FileNotFoundError:
        print("Error: Could not find processed data. Please run generate_variants.py first.")
        return

    print("  Training XGBoost model for explanation...")
    # Use XGBoost as it has native, fast TreeExplainer support
    model = xgb.XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("  Calculating SHAP values...")
    # TreeExplainer is optimized for trees
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_val)
    
    # 1. Summary Plot (Beeswarm) - The "Global" View
    print("  Generating Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, show=False)
    plt.title("SHAP Summary: Top Drivers of Churn", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_beeswarm.png", dpi=300)
    plt.close()
    
    # 2. Bar Plot - Feature Importance Ranking
    print("  Generating Bar Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance Ranking", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_importance_bar.png", dpi=300)
    plt.close()
    
    # 3. Dependence Plot - Interaction Effect (e.g., CurrentEquipmentDays)
    # Find top feature index automatically
    top_feature_idx = np.abs(shap_values.values).mean(0).argmax()
    top_feature_name = X_val.columns[top_feature_idx]
    
    print(f"  Generating Dependence Plot for {top_feature_name}...")
    # Create a dedicated figure for dependence plot
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.dependence_plot(
        top_feature_name, 
        shap_values.values, 
        X_val, 
        ax=ax, 
        show=False,
        interaction_index="auto" # Automatically find the strongest interaction
    )
    plt.title(f"SHAP Dependence: {top_feature_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_dependence_{top_feature_name}.png", dpi=300)
    plt.close()

    print(f"\nSUCCESS: Plots saved to {output_dir}/")

if __name__ == "__main__":
    generate_shap_plots()



