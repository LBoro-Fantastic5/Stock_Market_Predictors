import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score # Choose suitable metrics, these are ideas
import numpy as np

# Load the models, download them from Kaggle
# Link: https://www.kaggle.com/datasets/mdtarekislam/s-and-p-500-stock-data-01012021-30112024?select=rf_close_model.pkl 
rf_close = joblib.load("rf_close_model.pkl")
# Link: https://www.kaggle.com/datasets/mdtarekislam/s-and-p-500-stock-data-01012021-30112024?select=rf_volume_model.pkl 
rf_volume = joblib.load("rf_volume_model.pkl")

# Load the dataset, download from Kaggle
# Link: https://www.kaggle.com/datasets/mdtarekislam/s-and-p-500-stock-data-01012021-30112024?select=preprocessed_stock_data.csv 
dataset_version = "" # Can change to smaller datasets, but leave as is.
data = pd.read_csv(f"preprocessed_stock_data{dataset_version}.csv")

# Create 7-day lagged features for `close` and `volume`
for lag in range(0,7):
    data[f"close_lag_{lag}"] = data.groupby("ticker_idx")["close"].shift(lag)
    data[f"volume_lag_{lag}"] = data.groupby("ticker_idx")["volume"].shift(lag)

# Create `future_close` and `future_volume` features
data["future_close"] = data.groupby("ticker_idx")["close"].shift(-7)
data["future_volume"] = data.groupby("ticker_idx")["volume"].shift(-7)

# Drop rows with NaN values after shifting
data = data.dropna()

# Define features and targets
X = data[["days_since_start","ticker_idx"] +
         [f"close_lag_{lag}" for lag in range(0,7)] +
         [f"volume_lag_{lag}" for lag in range(0,7)]]
y_close = data["future_close"] # Target for close price prediction
y_volume = data["future_volume"] # Target for volume prediction

# Train-test split, 80:20, only concerned about the test set
_, X_test, _, y_close_test = train_test_split(X, y_close, test_size=0.2, random_state=42)
_, _, _, y_volume_test = train_test_split(X, y_volume, test_size=0.2, random_state=42)

# Evaluate the models
y_close_pred = rf_close.predict(X_test)
y_volume_pred = rf_volume.predict(X_test)

# Metrics for `close` price prediction
print("Close Price Prediction:")
### CODE GOES HERE

# Metrics for `volume` prediction
print("\nVolume Prediction:")
### CODE GOES HERE