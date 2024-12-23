import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

# Load dataset
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
X_lag_train, X_lag_test, ticker_train, ticker_test, y_train, y_test = train_test_split(
    X_lag, ticker_idx, y, test_size=0.2, random_state=42
)

# Define the number of unique stocks for embedding
num_tickers = data["ticker_idx"].nunique()

# Build the model
# LSTM Input
lstm_input = Input(shape=(7,2), name="LSTM_Input") # Lagged features for LSTM

# Ticker embedding input
ticker_input = Input(shape=(1,), name="Ticker_Input") # Stock index
ticker_embedding = Embedding(input_dim=num_tickers, output_dim=5, name="Ticker_Embedding")(ticker_input)
ticker_embedding = Flatten()(ticker_embedding)

# LSTM branch
lstm_branch = LSTM(64, return_sequences=True)(lstm_input)
lstm_branch = LSTM(64, return_sequences=False)(lstm_branch)

# Concatenate LSTM and ticker embedding
concat = Concatenate(name="Concatenate_LSTM_Embedding")([lstm_branch, ticker_embedding])
dense = Dense(32, activation="relu", name="Dense_Layer_1")(concat)
dense = Dropout(0.2, name="Dropout_1")(dense)
output = Dense(2, activation="linear", name="Output_Layer")(dense) # 2 outputs: future_close and future_volume

# Compile the model
model = Model(inputs=[lstm_input, ticker_input], outputs=output)
model.compile(optimizer="adam",loss="mse",metrics=["mae"])
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
history = model.fit(
    [X_lag_train, ticker_train], y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
save_model(model, "stock_price_volume_nn_p15.h5")
