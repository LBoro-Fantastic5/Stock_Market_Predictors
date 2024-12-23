import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error # Choose suitable metrics, these are ideas

# Load the saved model, donwload from Kaggle
# Link: https://www.kaggle.com/datasets/mdtarekislam/s-and-p-500-stock-data-01012021-30112024?select=stock_price_volume_nn_p15.h5 
model = load_model("stock_price_volume_nn_p15.h5", custom_objects={"mse": MeanSquaredError()})

# Load dataset, donwload from Kaggle
# Link: https://www.kaggle.com/datasets/mdtarekislam/s-and-p-500-stock-data-01012021-30112024?select=preprocessed_data_for_NN_RF_LR.csv 
data = pd.read_csv("preprocessed_data_for_NN_RF_LR.csv")

# Prepare features and labels
X_lag = data[[
    "close_lag_6", "volume_lag_6",
    "close_lag_5", "volume_lag_5",
    "close_lag_4", "volume_lag_4",
    "close_lag_3", "volume_lag_3",
    "close_lag_2", "volume_lag_2",
    "close_lag_1", "volume_lag_1",
    "close_lag_0", "volume_lag_0",
]].values

ticker_idx = data["ticker_idx"].values
y = data[["future_close", "future_volume"]].values

# Reshape lag features for LSTM
X_lag = X_lag.reshape((X_lag.shape[0],7,2)) # 7 days lag, 2 features (close & volume)

# Split data into train and test sets
_, X_lag_test, _, ticker_test, _, y_test = train_test_split(
    X_lag, ticker_idx, y, test_size=0.2, random_state=42
)

# Evaluate the model
y_pred = model.predict([X_lag_test, ticker_test])

print("Evaluation Metrics:")
# Metrics for `close` price prediction
print("Future Close Price:")
### CODE GOES HERE

# Metrics for `volume` prediction
print("\nFuture Volume:")
### CODE GOES HERE