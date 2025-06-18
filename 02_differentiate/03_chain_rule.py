import tensorflow as tf
import numpy as np

x = tf.Variable(3.0, dtype=tf.float32)

# Make the tape persistent so we can call .gradient() multiple times
with tf.GradientTape(persistent=True) as tape:
    y = x**3
    z = 2 * y

dy_dx = tape.gradient(y, x)  # dy/dx = 3x^2
dz_dy = tape.gradient(z, y)  # dz/dy = 2
dz_dx = tape.gradient(z, x)  # dz/dx = 2 * dy/dx

# Clean up to avoid memory leaks
del tape

print("Value of y:", y.numpy())
print("Value of z:", z.numpy())
print("Gradient dy/dx:", dy_dx.numpy())
print("Gradient dz/dy:", dz_dy.numpy())
print("Gradient dz/dx:", dz_dx.numpy())
