# run_lgb.py
import sys
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load preprocessed train/test data
X_train = np.load("../X_train_tree.npy")
y_train = np.load("../y_train_tree.npy")
X_test = np.load("../X_test_tree.npy")
y_test = np.load("../y_test_tree.npy")

model = LGBMRegressor(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=64,
    min_data_in_leaf=10,
    n_jobs=1,
    verbose=-1,
    random_state=42
)

print("Fitting LightGBM...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print(f"✅ LightGBM → MAE: {mae:.4f}, RMSE: {rmse:.4f}")
