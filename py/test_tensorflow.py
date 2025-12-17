import tensorflow as tf
import numpy as np

# إنشاء بيانات تدريب (x = [1,2,3,4], y = [2,4,6,8])
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([2, 4, 6, 8], dtype=float)

# بناء نموذج بسيط يحتوي على طبقة واحدة (شبكة عصبية بسيطة)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# تجميع النموذج (تحديد الخوارزمية وطريقة التعلم)
model.compile(optimizer='sgd', loss='mean_squared_error')

# تدريب النموذج على البيانات
print("🔄 جاري تدريب النموذج...")
model.fit(x, y, epochs=500, verbose=False)
print("✅ تم تدريب النموذج بنجاح!")

# تجربة النموذج على رقم جديد
new_number = 10.0
prediction = model.predict([new_number])
print(f"🔮 توقع النموذج عندما x = {new_number}: y = {prediction[0][0]}")