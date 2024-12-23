import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from model_preprocessor import ModelPreprocessor

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

X_close_train, X_close_test, y_close_train, y_close_test = train_test_split(
    X_close, y_close, test_size=0.2, random_state=42
)

X_volume_train, X_volume_test, y_volume_train, y_volume_test = train_test_split(
    X_volume, y_volume, test_size=0.2, random_state=42
)

# Train Logistic Regression with Cross-Validation
model_close = LogisticRegressionCV(
    Cs=20,  # Number of different C values to try
    cv=10,  # 5-fold cross-validation
    class_weight="balanced",
    max_iter=1000,
    scoring="accuracy",
    random_state=42
)

model_volume = LogisticRegressionCV(
    Cs=20,
    cv=10,
    max_iter=1000,
    scoring="accuracy",
    random_state=42
)

model_close.fit(X_close_train, y_close_train)
model_volume.fit(X_volume_train, y_volume_train)

joblib.dump(model_close, "logistic_model_close.h5")
joblib.dump(model_volume, "logistic_model_volume.h5")
