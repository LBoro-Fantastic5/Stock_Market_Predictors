import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # Some ideas for the classification metrics
import joblib
from model_preprocessor import ModelPreprocessor # Download the model_preprocessor module from Github (in Data_Handling folder)

# Load the models, download from Kaggle.
# Link: https://www.kaggle.com/datasets/mdtarekislam/s-and-p-500-stock-data-01012021-30112024?select=logistic_model_close.h5 
model_close = joblib.load("logistic_model_close.h5")
# Link: https://www.kaggle.com/datasets/mdtarekislam/s-and-p-500-stock-data-01012021-30112024?select=logistic_model_volume.h5 
model_volume = joblib.load("logistic_model_volume.h5")

# Load the dataset, download from Kaggle.
# Link: https://www.kaggle.com/datasets/mdtarekislam/s-and-p-500-stock-data-01012021-30112024?select=preprocessed_data_for_NN_RF_LR.csv 
filename = "preprocessed_data_for_NN_RF_LR.csv"
data = ModelPreprocessor(filename).get_data()

# Define features and targets
X_close = data[["days_since_start","ticker_idx"] +
               [f"close_lag_{lag}" for lag in range(0,7)]]
y_close = data["close_increase"]

X_volume = data[["days_since_start","ticker_idx"] +
                [f"volume_lag_{lag}" for lag in range(0,7)]]
y_volume = data["volume_increase"]

# Select columns to standardise
cols_to_std = ["days_since_start", "ticker_idx"]

# Separate columns to standardise and those already scaled
X_close_std = X_close[cols_to_std]
X_close_mnmx = X_close.drop(columns=cols_to_std)

X_volume_std = X_volume[cols_to_std]
X_volume_mnmx = X_volume.drop(columns=cols_to_std)

# Initialise scaler
# Download from GitHub (in Data_Handling folder)
scaler = joblib.load("scaler.pkl") #StandardScaler() #has negative vals

# Fit-transform only the selected cols
X_close_std = scaler.fit_transform(X_close_std)
X_volume_std = scaler.fit_transform(X_volume_std)

X_close = pd.concat([
    pd.DataFrame(X_close_std, columns=cols_to_std),
    X_close_mnmx.reset_index(drop=True)
], axis=1)

X_volume = pd.concat([
    pd.DataFrame(X_volume_std, columns=cols_to_std),
    X_volume_mnmx.reset_index(drop=True)
], axis=1)

_, X_close_test, _, y_close_test = train_test_split(
    X_close, y_close, test_size=0.2, random_state=42
)

_, X_volume_test, _, y_volume_test = train_test_split(
    X_volume, y_volume, test_size=0.2, random_state=42
)

# Predictions and evaluation for close price prediction
y_close_pred = model_close.predict(X_close_test)
print("Evaluation for Close Price Increase Prediction:")
### CODE GOES HERE

# Predictions and evaluation for volume prediction
y_volume_pred = model_volume.predict(X_volume_test)
print("Evaluation for Volume Increase Prediction:")
### CODE GOES HERE