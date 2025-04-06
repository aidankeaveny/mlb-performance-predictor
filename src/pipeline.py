import joblib
import numpy as np
import torch
from model_utils import PlayerMLP

def load_mlp_ensemble(X_np, top5_configs, model_dir="models/ensemble_engineered"):
    scaler = joblib.load(f"{model_dir}/scaler.joblib")
    features = joblib.load(f"{model_dir}/features.joblib")
    X_scaled = scaler.transform(X_np)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    preds = []
    for i in range(5):
        cfg = top5_configs[i][0]
        model = PlayerMLP(X_tensor.shape[1], cfg[0], dropout=cfg[4], activation='relu')
        model.load_state_dict(torch.load(f"{model_dir}/mlp_model_{i}.pt"))
        model.eval()
        with torch.no_grad():
            preds.append(model(X_tensor).numpy().flatten())

    return np.mean(preds, axis=0)
