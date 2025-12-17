# ====================== Bitcoin Price Predictor with LSTM ======================
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------- Parameters ----------------
TICKER = "BTC-USD"
START_DATE = "2018-01-01"
END_DATE = "2025-10-28"
LOOKBACK = 60
TEST_RATIO = 0.2
EPOCHS = 60
BATCH_SIZE = 32

# ---------------- Download Bitcoin Data ----------------
print("📥 Downloading Bitcoin price data...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)
data = data[['Close']]
data.dropna(inplace=True)
print("✅ Data downloaded successfully!")

# ---------------- Scale Data ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ---------------- Prepare Training Data ----------------
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        Y.append(dataset[i, 0])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_data, LOOKBACK)

train_size = int(len(X) * (1 - TEST_RATIO))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# ---------------- Build Model ----------------
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# ---------------- Train Model ----------------
print("🚀 Training model...")
early_stop = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('bitcoin_model.h5', save_best_only=True, monitor='val_loss')
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    verbose=1
)
print("✅ Model trained successfully!")

# ---------------- Evaluate Model ----------------
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = math.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"\n📊 Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# ---------------- Plot Results ----------------
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label="Actual Price", color='blue')
plt.plot(y_pred_rescaled, label="Predicted Price", color='orange')
plt.title("📈 Bitcoin Price Prediction (LSTM)")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# ---------------- Predict Next Day ----------------
last_window = scaled_data[-LOOKBACK:]
last_window = np.reshape(last_window, (1, LOOKBACK, 1))
next_scaled = model.predict(last_window)
next_price = scaler.inverse_transform(next_scaled)
print(f"\n💰 Predicted Bitcoin price for next day: ${next_price[0][0]:.2f}")