import tensorflow as tf
import numpy as np


x = tf.Variable(2.0, dtype=tf.float32)

with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x**2 + 3*x + 5  # Define a simple function
# Compute the gradient of y with respect to x
        dy_dx = t1.gradient(y, x)
# Compute the second derivative of y with respect to x
        d2y_dx2 = t2.gradient(dy_dx, x)
print("Value of y:", y.numpy())
print("Gradient dy/dx:", dy_dx.numpy())
print("Second derivative d2y/dx2:", d2y_dx2.numpy())
