import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

def enhance_features(X):
    X = X.copy()
    epsilon = 1e-6
    
    # 1. "Shock" Flags (Binary indicators of extreme behavior)
    if 'PercChangeMinutes' in X.columns:
        X['Usage_Drop_Significant'] = (X['PercChangeMinutes'] < -0.2).astype(int)
        X['Usage_Spike_Significant'] = (X['PercChangeMinutes'] > 0.2).astype(int)
        
    if 'PercChangeRevenues' in X.columns:
        X['Revenue_Drop_Significant'] = (X['PercChangeRevenues'] < -0.2).astype(int)

    # 2. Pain Points
    if 'DroppedCalls' in X.columns and 'MonthlyMinutes' in X.columns:
        # Interaction: High usage AND high drops = Anger
        X['HighUsage_HighDrops'] = (X['MonthlyMinutes'] * X['DroppedCalls'])
        
    if 'OverageMinutes' in X.columns and 'MonthlyRevenue' in X.columns:
        # Interaction: Overage relative to bill
        X['Overage_Bill_Ratio'] = X['OverageMinutes'] / (X['MonthlyRevenue'] + epsilon)

    # 3. Tenure Cohorts (Non-linear effects)
    if 'MonthsInService' in X.columns:
        X['New_Customer'] = (X['MonthsInService'] < 6).astype(int)
        X['Loyal_Customer'] = (X['MonthsInService'] > 24).astype(int)
        
    # 4. Device Risk
    if 'CurrentEquipmentDays' in X.columns:
        X['Old_Device'] = (X['CurrentEquipmentDays'] > 365).astype(int)
        
    return X

def run_enhanced_training():
    print(f"--- Training CatBoost with ENHANCED Features ---")
    base_dir = 'processed_data_folds_strict'
    folds = range(1, 6)
    
    metrics = {
        'AUC': [], 
        'Recall_1_Top20': []
    }

    for fold in folds:
        try:
            X_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_X.csv")
            y_train = pd.read_csv(f"{base_dir}/fold_{fold}_train_y.csv").iloc[:, 0]
            X_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_X.csv")
            y_val = pd.read_csv(f"{base_dir}/fold_{fold}_val_y.csv").iloc[:, 0]
            
            # --- APPLY ENHANCEMENTS ---
            X_train_aug = enhance_features(X_train)
            X_val_aug = enhance_features(X_val)
            
            # CatBoost
            model = CatBoostClassifier(
                random_state=42, verbose=0, allow_writing_files=False, thread_count=-1
            )
            model.fit(X_train_aug, y_train)
            
            y_prob = model.predict_proba(X_val_aug)[:, 1]
            
            # Metrics
            auc = roc_auc_score(y_val, y_prob)
            
            # Top 20% Recall
            sorted_indices = np.argsort(y_prob)[::-1]
            k = int(len(y_val) * 0.2)
            top_k_indices = sorted_indices[:k]
            y_pred_top20 = np.zeros_like(y_val)
            y_pred_top20[top_k_indices] = 1
            
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_top20).ravel()
            rec_1_top20 = tp / (tp + fn)
            
            metrics['AUC'].append(auc)
            metrics['Recall_1_Top20'].append(rec_1_top20)
            
            print(f"  Fold {fold}: AUC={auc:.4f}, Recall@Top20%={rec_1_top20:.4f}")
            
        except FileNotFoundError:
            print(f"  Fold {fold}: Data not found.")

    # Summary
    print("\n" + "="*60)
    print("ENHANCED MODEL RESULTS")
    print("="*60)
    avg_auc = np.mean(metrics['AUC'])
    avg_rec = np.mean(metrics['Recall_1_Top20'])
    
    print(f"Average AUC:              {avg_auc:.4f} (Prev: 0.6740)")
    print(f"Average Recall @ Top 20%: {avg_rec:.4f} (Prev: 0.3327)")
    
    if avg_rec > 0.3327:
        print("\nSUCCESS: Feature Engineering improved Recall!")
        print(f"Lift gained: +{(avg_rec - 0.3327)*100:.2f}%")
    else:
        print("\nRESULT: No significant improvement. Signal limit reached.")

if __name__ == "__main__":
    run_enhanced_training()

