import os
import pandas as pd
import numpy as np
import torch
import joblib
from catboost import CatBoostRegressor
from pybaseball import batting_stats
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import TARGET_VAR, FEATURE_COLS
from data_utils import (
    build_y1_features,
    create_aggregates,
    encode_positions
)
from model_utils import PlayerMLP

MODEL_DIR = "models/ensemble_engineered"
OUTPUT_DIR = "output"
YEAR=2025
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_year_data(year):
    try:
        df = batting_stats(year)
        df['Season'] = year
        return df
    except Exception as e:
        print(f"Failed to load data for year {year}: {e}")
        return None

def main():
    print("Device:", device)
    print("Predicting:", TARGET_VAR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models and assets
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.joblib"))

    # === Load Assets ===
    top5 = joblib.load("models/ensemble_engineered/top5_configs.joblib")
    catboost_model = joblib.load("models/ensemble_engineered/catboost_model.joblib")
    print("Assets loaded.")


    # === Pull Latest Data ===
    print("Loading 2024 data...")
    df_2024 = batting_stats(2024)
    df_2024['Season'] = 2024

    # === Load Historical Data in Parallel ===
    print("Loading historical data...")
    with ThreadPoolExecutor() as executor:
        all_years = list(tqdm(executor.map(load_year_data, range(1900, 2024)), total=2024 - 1900))
    df_hist = pd.concat([d for d in all_years if d is not None])


    # === Create Aggregates from Historical Data ===
    print("Creating aggregates...")
    df_agg = create_aggregates(df_hist)

    # === Build *_y1 features ===
    print("Building features...")
    df_y1 = build_y1_features(df_2024)

    # === Merge historical and y1 ===
    df = pd.merge(df_y1, df_agg, on='Name', how='inner')
    df = encode_positions(df, df_2024)

    # === Filter to saved features and scale ===
    X = df[features].fillna(0).values
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # === MLP Ensemble Prediction ===
    print("Predicting with MLP ensemble...")
    mlp_preds = load_mlp_ensemble(X_tensor, top5)
    df[f"{TARGET_VAR}_{YEAR}_pred_MLP"] = mlp_preds

    # === CatBoost Prediction ===
    print("Predicting with CatBoost...")
    cat_preds = catboost_model.predict(X_scaled)
    df[f"{TARGET_VAR}_{YEAR}_pred_CatBoost"] = cat_preds

    # === Save Output ===
    output_cols = ['Name', f"{TARGET_VAR}_{YEAR}_pred_MLP", f"{TARGET_VAR}_{YEAR}_pred_CatBoost"]
    df[output_cols].to_csv(os.path.join(OUTPUT_DIR, f"{TARGET_VAR}_{YEAR}_predictions.csv"), index=False)

    print(f"Saved {TARGET_VAR} predictions to: {OUTPUT_DIR}/{TARGET_VAR}_{YEAR}_predictions.csv")


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
