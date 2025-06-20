import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# trainning Data
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,), name="layer1"))

# Compile
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train
model.fit(x, y, epochs=500, verbose=0)

# Predict
predictions = model.predict(x)
print("Predictions:", predictions)

# Plot
plt.scatter(x, y, label='data_points')
plt.plot(x, predictions, label='Model Prediction', color='red')
plt.legend()
plt.show()
