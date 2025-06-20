import tensorflow as tf
import numpy as np

# Sample data: y = 2x
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,), name="dense_layer")
])

# Compile the model
model.compile(
    optimizer='sgd',             # Stochastic Gradient Descent
    loss='mean_squared_error',   # Loss for regression
    metrics=['mae']              # Mean Absolute Error
)

# Print model summary
model.summary()

# Train the model
model.fit(x, y, epochs=100, verbose=0)

# Test prediction
test_input = np.array([6], dtype=np.float32)
prediction = model.predict(test_input)
print(f"Prediction for input 6: {prediction[0][0]:.2f}")
