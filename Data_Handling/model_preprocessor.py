import pandas as pd
import numpy as np

class ModelPreprocessor:
    def __init__(self, dataset_name):
        # Load dataset
        self.data = pd.read_csv(dataset_name)

    def preprocess_data(self, num_days_prior=7, num_days_future=7):
        # Create 7-day lagged features for `close` and `volume`
        for lag in range(0,num_days_prior):
            self.data[f"close_lag_{lag}"] = self.data.groupby("ticker_idx")["close"].shift(lag)
            self.data[f"volume_lag_{lag}"] = self.data.groupby("ticker_idx")["volume"].shift(lag)

        # Create `future_close` and `future_volume` features
        self.data["future_close"] = self.data.groupby("ticker_idx")["close"].shift(-num_days_future)
        self.data["future_volume"] = self.data.groupby("ticker_idx")["volume"].shift(-num_days_future)
        self.data["close_increase"] = (self.data["future_close"] > self.data["close_lag_0"]).astype(int)
        self.data["volume_increase"] = (self.data["future_volume"] > self.data["volume_lag_0"]).astype(int)

        # Drop rows with NaN values after shifting
        self.data = self.data.dropna()
        duplicates = self.data.duplicated()
        #print(f"Number of duplicated rows: {duplicates.sum()}")

        self.data = self.data.drop_duplicates()

        # Define a preprocessed dataset
        preprocessed_data = self.data[["days_since_start","ticker_idx"] +
         [f"close_lag_{lag}" for lag in range(0,7)] +
         [f"volume_lag_{lag}" for lag in range(0,7)] +
         ["future_close", "future_volume", "close_increase", "volume_increase"]]
        
        return preprocessed_data
    
    def save_data_to_csv(self, preprocessed_data, filename="preprocessed_data_for_NN_RF_LR.csv", idx=""):
        preprocessed_data.to_csv(filename, index=idx)

    def get_data(self):
        return self.data

# filename = "preprocessed_stock_data.csv"
# preprocessor = ModelPreprocessor(filename)
# data = preprocessor.preprocess_data()
# preprocessor.save_data_to_csv(data)
