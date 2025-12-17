import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ======================== إعداد البيانات ========================
TICKER = "BTC-USD"  # رمز البيتكوين
START_DATE = "2020-01-01"
END_DATE = "2025-10-28"

print("[INFO] Downloading Bitcoin data...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE, interval="1d")

# التحقق من البيانات
if data.empty:
    raise ValueError("❌ لم يتم تحميل بيانات البيتكوين، تحقق من الاتصال أو من الرمز!")

# الحصول على آخر إغلاق فقط
last_close = data['Close'].iloc[-1]
last_date = data.index[-1].strftime("%Y-%m-%d")

print(f"[INFO] Data updated successfully! Latest date: {last_date}, Latest close: ${last_close:.2f}")

# ======================== معالجة البيانات ========================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

LOOKBACK = 60  # عدد الأيام المستخدمة للتنبؤ
x_train, y_train = [], []

for i in range(LOOKBACK, len(scaled_data)):
    x_train.append(scaled_data[i - LOOKBACK:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# ======================== بناء نموذج LSTM ========================
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("[INFO] Training the model...")
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
print("[INFO] Training completed!")

# ======================== التنبؤ ========================
last_60_days = scaled_data[-LOOKBACK:]
x_test = np.array([last_60_days])
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"\n[RESULT] Predicted Bitcoin price for next step: ${predicted_price[0][0]:.2f}")

# ======================== الرسم ========================
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label="Actual Bitcoin Price")
plt.title("Bitcoin Price History")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()