import tensorflow as tf
import numpy as np

# Sample data: y = 2x
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# 1. Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,), name="dense_layer")
])

# 2. Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# 3. Train the model
model.fit(x, y, epochs=100, verbose=0)

# 4. Get and print trained weights
trained_weights = model.get_weights()
print("Trained Weights and Bias:")
print("  Kernel (slope):", trained_weights[0])
print("  Bias (intercept):", trained_weights[1])

# 5. Make a prediction with trained weights
prediction = model.predict(np.array([6.0]))
print(f"\nPrediction with trained weights for x=6: {prediction[0][0]:.2f}")

# 6. Manually set new weights: y = 3x + 1
new_weights = [np.array([[3.0]]), np.array([1.0])]
model.set_weights(new_weights)

# 7. Make a prediction with manual weights
manual_prediction = model.predict(np.array([6.0]))
print(f"\nPrediction with manual weights (y = 3x + 1) for x=6: {manual_prediction[0][0]:.2f}")
