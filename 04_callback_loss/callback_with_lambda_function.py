import tensorflow as tf
import numpy as np

# Dummy data
x = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)

# Lambda callback that prints loss after every epoch
log_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1}, Loss: {logs['loss']:.4f}")
)

# Simple model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=3, callbacks=[log_callback], verbose=0)
