from churn_pipeline_saransh import ChurnPipeline
import os
import pandas as pd

def generate_data():
    # Path to raw data
    data_path = 'cell2celltrain.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # 1. Generate Strict Folds (Leakage Removed)
    print("Generating STRICT folds (Leakage Removed)...")
    pipeline_strict = ChurnPipeline(strict_mode=True)
    df = pipeline_strict.load_data(data_path)
    df_eng = pipeline_strict.clean_and_engineer(df)
    pipeline_strict.save_folds(df_eng, output_dir='processed_data_folds_strict')

    # 2. Generate Permissive Folds (Leakage Included)
    print("\nGenerating PERMISSIVE folds (Leakage Included)...")
    pipeline_permissive = ChurnPipeline(strict_mode=False)
    pipeline_permissive.save_folds(df_eng, output_dir='processed_data_folds_permissive')

if __name__ == "__main__":
    generate_data()

