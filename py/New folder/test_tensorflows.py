import tensorflow as tf
import numpy as np

# إنشاء بيانات التدريب
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# إنشاء نموذج بسيط (شبكة عصبية بسيطة)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# تجميع النموذج (تحديد الخوارزمية وطريقة التعلم)
model.compile(optimizer='sgd', loss='mean_squared_error')

# تدريب النموذج على البيانات
print("📊 جاري تدريب النموذج ...")
model.fit(x, y, epochs=500, verbose=False)
print("✅ تم تدريب النموذج بنجاح!")

# تجربة النموذج على رقم جديد
new_number = 10.0
prediction = model.predict(np.array([new_number]))  # ← هنا التعديل
print(f"🔮 توقع النموذج عندما x = {new_number}: y = {prediction[0][0]}")