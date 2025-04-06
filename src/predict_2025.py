from data_utils import load_historical_data, create_aggregates
from pipeline import load_mlp_ensemble
import pandas as pd
import joblib

def main():
    features = joblib.load("models/ensemble_engineered/features.joblib")
    top5 = joblib.load("models/ensemble_engineered/top5_configs.joblib")

    df_hist = load_historical_data(up_to=2023, feature_cols=features)
    df_2024 = batting_stats(2024)

    agg = create_aggregates(df_hist, features)
    ...  # Merge, Pos, fillna, etc.

    X = df_final[features].fillna(0).values
    df_final['OPS_2025_pred'] = load_mlp_ensemble(X, top5)

    df_final[['Name', 'OPS_2025_pred']].to_csv("output/2025_predictions.csv", index=False)
    print("✅ Saved 2025 predictions to output/2025_predictions.csv")

if __name__ == "__main__":
    main()
