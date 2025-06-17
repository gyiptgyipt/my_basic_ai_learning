import tensorflow as tf


x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

dataset = tf.data.Dataset.from_tensor_slices(x)

for item in dataset:
    print(item.numpy())