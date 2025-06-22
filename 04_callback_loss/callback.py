import tensorflow as tf
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30], dtype=float)

# Define simple callback
class SimpleLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting epoch {epoch + 1}...")

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('accuracy')  # This will be None unless accuracy is a metric
        if acc is not None:
            print(f"Epoch {epoch + 1} ended. Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        else:
            print(f"Epoch {epoch + 1} ended. Loss: {loss:.4f}")

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(1,))
])

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])  # Accuracy won't be meaningful here

# Train the model with the callback
model.fit(x, y, epochs=5, callbacks=[SimpleLogger()], verbose=0)
