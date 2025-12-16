import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SimpleTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing
        self.mapping = {}
        self.global_mean = 0

    def fit(self, X, y):
        X = X.copy()
        self.global_mean = y.mean()
        for col in self.cols:
            stats = y.groupby(X[col]).agg(['mean', 'count'])
            # Smoothed mean
            smooth = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
            self.mapping[col] = smooth
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in self.mapping:
                # Map known values, fill missing (unknown during fit) with global mean
                X[col] = X[col].map(self.mapping[col]).fillna(self.global_mean)
            else:
                pass # Should not happen if cols checked
        return X

class ChurnPipeline:
    def __init__(self, strict_mode=True, random_state=42):
        """
        Initialize the pipeline.
        
        Args:
            strict_mode (bool): If True, drops leakage-prone columns.
            random_state (int): Seed for reproducibility.
        """
        self.strict_mode = strict_mode
        self.random_state = random_state
        self.target_col = 'Churn'
        self.high_cardinality_cols = ['ServiceArea', 'PrizmCode', 'Occupation']
        self.leakage_cols = ['MadeCallToRetentionTeam', 'RetentionCalls', 'RetentionOffersAccepted']
        self.numeric_cols = None
        self.categorical_cols = None
        
    def load_data(self, filepath):
        """Loads the dataset."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        return df

    def clean_and_engineer(self, df):
        """
        Performs initial cleaning and feature engineering that is row-independent 
        (safe to do before split, or strictly defined per row).
        """
        print("Performing cleaning and feature engineering...")
        df = df.copy()
        
        # 1. Target Encoding
        if df[self.target_col].dtype == object:
            df[self.target_col] = df[self.target_col].map({'Yes': 1, 'No': 0})
            
        # 2. Handle Missing Values (Initial Pass for Feature Engineering)
        # epsilon for division
        epsilon = 1e-6
        
        # 3. Usage & Revenue Dynamics
        # ARPU
        df['ARPU'] = df['MonthlyRevenue'] / (df['MonthlyMinutes'] + epsilon)
        # Overage Share
        df['OverageShare'] = df['OverageMinutes'] / (df['MonthlyMinutes'] + epsilon)
        
        # 4. Care/Network Frictions
        total_calls = df['InboundCalls'] + df['OutboundCalls'] + epsilon
        df['DroppedCallRate'] = df['DroppedCalls'] / total_calls
        df['BlockedCallRate'] = df['BlockedCalls'] / total_calls
        df['UnansweredRate'] = df['UnansweredCalls'] / total_calls
        df['CustomerCareRatio'] = df['CustomerCareCalls'] / (df['MonthlyMinutes'] + epsilon)
        
        # 5. Call-Mix
        df['InboundShare'] = df['InboundCalls'] / total_calls
        peak_total = df['PeakCallsInOut'] + df['OffPeakCallsInOut'] + epsilon
        df['OffPeakShare'] = df['OffPeakCallsInOut'] / peak_total
        
        # Winsorization prep: we don't do it here to avoid leakage, but we define columns to watch
        
        return df

    def get_folds(self, df, n_splits=5, group_col=None):
        """
        Returns a generator of (train_idx, val_idx).
        If group_col is provided, uses StratifiedGroupKFold.
        Otherwise uses StratifiedKFold.
        """
        y = df[self.target_col]
        
        if group_col:
            from sklearn.model_selection import StratifiedGroupKFold
            sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=self.random_state, shuffle=True)
            groups = df[group_col]
            return sgkf.split(df, y, groups=groups)
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            return skf.split(df, y)

    def process_fold(self, train_df, val_df):
        """
        Fits imputers/encoders/scalers on train_df and transforms both.
        Returns processed X_train, y_train, X_val, y_val.
        """
        train_df = train_df.copy()
        val_df = val_df.copy()
        
        # Separate X and y
        y_train = train_df[self.target_col].values
        y_val = val_df[self.target_col].values
        
        X_train = train_df.drop(columns=[self.target_col, 'CustomerID']) # Drop ID if present
        X_val = val_df.drop(columns=[self.target_col, 'CustomerID'])
        
        # Drop leakage columns if Strict
        if self.strict_mode:
            X_train = X_train.drop(columns=self.leakage_cols, errors='ignore')
            X_val = X_val.drop(columns=self.leakage_cols, errors='ignore')
            
        # Identify Numeric and Categorical columns
        if self.numeric_cols is None:
            self.numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            
        # 1. Imputation
        # Numeric: Median
        medians = X_train[self.numeric_cols].median()
        X_train[self.numeric_cols] = X_train[self.numeric_cols].fillna(medians)
        X_val[self.numeric_cols] = X_val[self.numeric_cols].fillna(medians)
        
        # Categorical: Unknown
        # Ensure categorical columns are strings
        for col in self.categorical_cols:
            X_train[col] = X_train[col].fillna("Unknown").astype(str)
            X_val[col] = X_val[col].fillna("Unknown").astype(str)
        
        # 2. Winsorization (PercChangeMinutes, PercChangeRevenues)
        # 1st and 99th percentile
        for col in ['PercChangeMinutes', 'PercChangeRevenues']:
            if col in X_train.columns:
                lower = X_train[col].quantile(0.01)
                upper = X_train[col].quantile(0.99)
                X_train[col] = X_train[col].clip(lower, upper)
                X_val[col] = X_val[col].clip(lower, upper)

        # 3. Log1p (Heavy tails)
        skew_candidates = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 
                           'OverageMinutes', 'RoamingCalls', 'DroppedCalls', 'BlockedCalls', 
                           'UnansweredCalls', 'CustomerCareCalls', 'ThreewayCalls', 
                           'ReceivedCalls', 'OutboundCalls', 'InboundCalls', 
                           'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls', 
                           'CallForwardingCalls', 'CallWaitingCalls', 'MonthsInService', 
                           'UniqueSubs', 'ActiveSubs', 'Handsets', 'HandsetModels', 
                           'CurrentEquipmentDays']
        
        for col in skew_candidates:
            if col in X_train.columns:
                # Ensure non-negative for log1p
                if X_train[col].min() >= 0:
                    X_train[col] = np.log1p(X_train[col])
                    X_val[col] = np.log1p(X_val[col])

        # 4. Categorical Encoding
        high_card_cols = [c for c in self.high_cardinality_cols if c in X_train.columns]
        other_cats = [c for c in self.categorical_cols if c not in high_card_cols]
        
        # Target Encoder
        if high_card_cols:
            te = SimpleTargetEncoder(cols=high_card_cols, smoothing=10)
            # Need y_train as pandas Series
            te.fit(X_train, pd.Series(y_train, index=X_train.index))
            X_train = te.transform(X_train)
            X_val = te.transform(X_val)
            
        # One Hot Encoder
        if other_cats:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            # Fit on train
            ohe.fit(X_train[other_cats])
            
            # Transform
            train_ohe = pd.DataFrame(ohe.transform(X_train[other_cats]), 
                                     columns=ohe.get_feature_names_out(other_cats), 
                                     index=X_train.index)
            val_ohe = pd.DataFrame(ohe.transform(X_val[other_cats]), 
                                   columns=ohe.get_feature_names_out(other_cats), 
                                   index=X_val.index)
            
            # Drop original cats and concat ohe
            X_train = pd.concat([X_train.drop(columns=other_cats), train_ohe], axis=1)
            X_val = pd.concat([X_val.drop(columns=other_cats), val_ohe], axis=1)
            
        # 5. Scaling (StandardScaler)
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        
        return X_train, y_train, X_val, y_val

    def save_folds(self, df, output_dir='processed_folds', n_splits=5):
        """
        Generates folds, processes them, and saves to CSV in output_dir.
        Useful if you want to separate preprocessing from training completely.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        folds = self.get_folds(df, n_splits=n_splits)
        
        for i, (train_idx, val_idx) in enumerate(folds):
            fold_num = i + 1
            print(f"Processing and saving Fold {fold_num}...")
            
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            X_train, y_train, X_val, y_val = self.process_fold(train_df, val_df)
            
            # Save
            X_train.to_csv(f"{output_dir}/fold_{fold_num}_train_X.csv", index=False)
            pd.DataFrame(y_train, columns=[self.target_col]).to_csv(f"{output_dir}/fold_{fold_num}_train_y.csv", index=False)
            X_val.to_csv(f"{output_dir}/fold_{fold_num}_val_X.csv", index=False)
            pd.DataFrame(y_val, columns=[self.target_col]).to_csv(f"{output_dir}/fold_{fold_num}_val_y.csv", index=False)
            
        print(f"All folds saved to {output_dir}/")

class ResultHandler:
    @staticmethod
    def evaluate(y_true, y_pred_prob, threshold=None):
        """
        Calculates metrics: ROC-AUC, PR-AUC, Confusion Matrix.
        If threshold is None, estimates best threshold based on F1 or Distance to Top-Left.
        For simplicity here, uses 0.5 default or provided.
        """
        if threshold is None:
            threshold = 0.5
            
        roc = roc_auc_score(y_true, y_pred_prob)
        pr_auc = average_precision_score(y_true, y_pred_prob)
        brier = brier_score_loss(y_true, y_pred_prob)
        
        y_pred = (y_pred_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'ROC-AUC': roc,
            'PR-AUC': pr_auc,
            'Brier Score': brier,
            'Confusion Matrix': cm
        }

    @staticmethod
    def print_metrics(metrics):
        print("\n--- Evaluation Metrics ---")
        print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
        print(f"PR-AUC:  {metrics['PR-AUC']:.4f}")
        print(f"Brier Score: {metrics['Brier Score']:.4f}")
        print("Confusion Matrix:")
        print(metrics['Confusion Matrix'])

if __name__ == "__main__":
    # Default Usage: Process and Save Folds
    pipeline = ChurnPipeline(strict_mode=True)
    
    # Try finding the file in common locations
    possible_paths = [
        'saransh-experiments/cell2celltrain.csv',
        'cell2celltrain.csv',
        'linear_models/cell2celltrain.csv'
    ]
    
    file_path = None
    for path in possible_paths:
        try:
            f = open(path)
            f.close()
            file_path = path
            break
        except FileNotFoundError:
            continue
            
    if file_path:
        print(f"Using data file: {file_path}")
        df = pipeline.load_data(file_path)
        df_eng = pipeline.clean_and_engineer(df)
        
        print(f"\nDataset Shape: {df_eng.shape}")
        print(f"Strict Mode: {pipeline.strict_mode}")
        
        # Save folds for the user
        pipeline.save_folds(df_eng, output_dir='processed_data_folds')
        
    else:
        print("Error: cell2celltrain.csv not found in common locations.")
