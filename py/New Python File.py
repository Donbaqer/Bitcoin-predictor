# ✅ اختبار جميع المكتبات

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

print("كل المكتبات تعمل بشكل صحيح ✅")

# تجربة numpy
arr = np.array([1, 2, 3, 4, 5])
print("Numpy array:", arr)

# تجربة pandas
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [2, 4, 6, 8, 10]
})
print("\nDataFrame من pandas:")
print(data)

# تجربة scikit-learn
model = LinearRegression()
model.fit(data[['x']], data['y'])
pred = model.predict([[6]])
print("\nتنبؤ scikit-learn لقيمة x=6 هو:", pred[0])

# تجربة matplotlib
plt.plot(data['x'], data['y'], marker='o')
plt.title("رسم بياني تجريبي")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# تجربة yfinance
print("\nجلب بيانات سهم AAPL (آبل) من yfinance...")
apple = yf.download("AAPL", period="5d")
print(apple)