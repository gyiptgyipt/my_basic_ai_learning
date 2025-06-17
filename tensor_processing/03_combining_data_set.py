import tensorflow as tf

tf.random.set_seed(42)
x =  tf.random.uniform(shape=(6, 2), minval=0, maxval=10)
y = tf.range(6)

print(x)
print(y)