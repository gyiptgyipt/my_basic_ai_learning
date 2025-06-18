import tensorflow as tf
import numpy as np


x = tf.Variable(2.0, dtype=tf.float32)

with tf.GradientTape() as tape:
    y = x**2 + 3*x + 5  # Define a simple function

# Compute the gradient of y with respect to x
dy_dx = tape.gradient(y, x)
print("Value of y:", y.numpy())
print("Gradient dy/dx:", dy_dx.numpy())