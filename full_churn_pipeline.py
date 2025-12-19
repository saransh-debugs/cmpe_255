import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

class ChurnAnalyzer:
    def __init__(self, data_dir='processed_data_folds_strict'):
        self.data_dir = data_dir
        self.feature_families = {
            'Usage_Revenue': [
                'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 
                'OverageMinutes', 'RoamingCalls', 'PercChangeMinutes', 'PercChangeRevenues',
                'ARPU', 'OverageShare'
            ],
            'Friction': [
                'DroppedCalls', 'BlockedCalls', 'UnansweredCalls', 'CustomerCareCalls',
                'DroppedCallRate', 'BlockedCallRate', 'UnansweredRate', 'CustomerCareRatio'
            ],
            'Call_Mix': [
                'ThreewayCalls', 'ReceivedCalls', 'OutboundCalls', 'InboundCalls', 
                'PeakCallsInOut', 'OffPeakCallsInOut', 'InboundShare', 'OffPeakShare',
                'CallForwardingCalls', 'CallWaitingCalls'
            ],
            'Lifecycle': [
                'MonthsInService', 'UniqueSubs', 'ActiveSubs', 'Handsets', 
                'HandsetModels', 'CurrentEquipmentDays', 'RetentionCalls', 
                'RetentionOffersAccepted' # If present
            ],
            'Demographics': [
                'AgeHH1', 'AgeHH2', 'ChildrenInHH', 'HandsetRefurbished', 
                'HandsetWebCapable', 'TruckOwner', 'RVOwner', 'Homeownership', 
                'BuysViaMailOrder', 'RespondsToMailOffers', 'OptOutMailings', 
                'NonUSTravel', 'OwnsComputer', 'HasCreditCard', 'NewCellphoneUser', 
                'IncomeGroup', 'PrizmCode', 'Occupation', 'MaritalStatus', 'ServiceArea',
                'CreditRating', 'ReferralsMadeBySubscriber', 'AdjustmentsToCreditRating'
            ]
        }

    def load_fold(self, fold_num):
        try:
            X_train = pd.read_csv(f"{self.data_dir}/fold_{fold_num}_train_X.csv")
            y_train = pd.read_csv(f"{self.data_dir}/fold_{fold_num}_train_y.csv").iloc[:, 0]
            X_val = pd.read_csv(f"{self.data_dir}/fold_{fold_num}_val_X.csv")
            y_val = pd.read_csv(f"{self.data_dir}/fold_{fold_num}_val_y.csv").iloc[:, 0]
            return X_train, y_train, X_val, y_val
        except FileNotFoundError:
            return None, None, None, None

    def get_family_features(self, all_columns, family_name):
        """Matches family keywords to actual column names (handling OHE)."""
        family_keywords = self.feature_families.get(family_name, [])
        selected_cols = []
        for col in all_columns:
            # Check  col start with any keyword (for one hot encoping) or exact match
            for keyword in family_keywords:
                if col == keyword or col.startswith(f"{keyword}_"):
                    selected_cols.append(col)
                    break
        return selected_cols

    def run_full_analysis(self):
        print("Starting Full Churn Analysis (Strict View)...")
        
        # Use Fold 1 for detailed analysis (SHAP, Permutation) to save time
        X_train, y_train, X_val, y_val = self.load_fold(1)
        if X_train is None:
            print("Error loading data.")
            return

        model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        
        print("\n1. Training Base Model (XGBoost)...")
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)
        
        print(f"   Base AUC: {auc:.4f}")
        print(f"   Brier Score: {brier:.4f}")
        
        # 2. Calibration Analysis
        print("\n2. Calibration Analysis...")
        prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10)
        print("   Calibration Curve (Bin Means):")
        for t, p in zip(prob_true, prob_pred):
            print(f"     True: {t:.4f}, Pred: {p:.4f}")
            
        # 3. Permutation Importance
        print("\n3. Permutation Importance (Top 10)...")
        perm_importance = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=-1)
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        
        top_10 = sorted_idx[:10]
        for i in top_10:
            print(f"   {X_val.columns[i]}: {perm_importance.importances_mean[i]:.4f}")
            
        # 4. Feature Family Ablation
        print("\n4. Feature Family Ablation Study...")
        print(f"   {'Family Removed':<20} | {'AUC':<8} | {'Drop':<8}")
        print("-" * 45)
        
        baseline_auc = auc
        
        for family in self.feature_families.keys():
            # Identify cols to remove
            cols_to_remove = self.get_family_features(X_train.columns, family)
            
            if not cols_to_remove:
                continue
                
            # Create ablated datasets
            X_train_abl = X_train.drop(columns=cols_to_remove)
            X_val_abl = X_val.drop(columns=cols_to_remove)
            
            # Train and Evaluate
            model_abl = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
            model_abl.fit(X_train_abl, y_train)
            y_prob_abl = model_abl.predict_proba(X_val_abl)[:, 1]
            auc_abl = roc_auc_score(y_val, y_prob_abl)
            
            drop = baseline_auc - auc_abl
            print(f"   {family:<20} | {auc_abl:.4f}   | {drop:+.4f}")

        print("\n5. SHAP Values (Summary)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': X_val.columns,
            'Mean_SHAP': mean_shap
        }).sort_values(by='Mean_SHAP', ascending=False).head(10)
        
        print("   Top 10 Features by SHAP:")
        print(shap_df)

if __name__ == "__main__":
    analyzer = ChurnAnalyzer()
    analyzer.run_full_analysis()

