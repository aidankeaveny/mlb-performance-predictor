import pandas as pd
import numpy as np
import torch
import joblib
from catboost import CatBoostRegressor
from pybaseball import batting_stats
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


from data_utils import (
    build_y1_features,
    create_aggregates,
    encode_positions
)
from model_utils import PlayerMLP
from pipeline import load_mlp_ensemble
import os

MODEL_DIR = "models/ensemble_engineered"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")



def main():
    print(f"torch version: {torch.__version__}")  # Should print 2.x.x+cpu
    print(torch.backends.mps.is_available())  # Should print True if MPS is available

    # Load models and assets
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))

    # === Load Assets ===
    top5 = joblib.load("models/ensemble_engineered/top5_configs.joblib")
    catboost_model = joblib.load("models/ensemble_engineered/catboost_model.joblib")
    print("Assets loaded.")

    print("Generating 2025 Predictions...")

    # === Pull Latest Data ===
    df_2024 = batting_stats(2024)
    df_2024['Season'] = 2024
    print("2024 data loaded.")

    # === Load Historical Data in Parallel ===
    with ThreadPoolExecutor() as executor:
        all_years = list(tqdm(executor.map(load_year_data, range(1900, 2024)), total=2024 - 1900, desc="Loading Historical Data", unit="year"))
    
    # Filter out None values for failed years
    all_years = [year_data for year_data in all_years if year_data is not None]
    df_hist = pd.concat(all_years)
    print("Historical data loaded.")

    # === Create Aggregates from Historical Data ===
    df_agg = create_aggregates(df_hist)
    print("Historical data aggregated.")

    # === Build *_y1 features ===
    df_y1 = build_y1_features(df_2024)
    print("2024 features built.")

    # === Merge historical and y1 ===
    df = pd.merge(df_y1, df_agg, on='Name', how='inner')
    df = encode_positions(df, df_2024)
    print("Data merged and positions encoded.")

    # === Filter to saved features and scale ===
    X = df[features].fillna(0).values
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    print("Data filtered to features and scaled.")

    # === MLP Ensemble Prediction ===
    mlp_preds = load_mlp_ensemble(X_tensor, top5)
    df['OPS_2025_pred_MLP'] = mlp_preds

    # === CatBoost Prediction ===
    cat_preds = catboost_model.predict(X_scaled)
    df['OPS_2025_pred_CatBoost'] = cat_preds

    # === Save Output ===
    output_cols = ['Name', 'OPS_2025_pred_MLP', 'OPS_2025_pred_CatBoost']
    df[output_cols].to_csv("output/2025_predictions.csv", index=False)

    print("2025 predictions saved to: output/2025_predictions.csv")

def load_year_data(year):
    try:
        return batting_stats(year)
    except Exception as e:
        print(f"Failed to load data for year {year}: {e}")
        return None

def load_mlp_ensemble(X_tensor, top5):
    ensemble_preds = []
    for i in range(5):
        # Extract hidden dimensions and dropout
        hidden_dims = top5[i][0]
        dropout = top5[i][4]
        
        # Create the model using these configurations
        model = PlayerMLP(
            input_dim=X_tensor.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation='relu'
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"mlp_model_{i}.pt")))
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor).cpu().numpy().flatten()
            ensemble_preds.append(preds)
    return np.mean(ensemble_preds, axis=0)

if __name__ == "__main__":
    main()
