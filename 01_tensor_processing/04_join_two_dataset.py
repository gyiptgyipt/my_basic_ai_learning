import tensorflow as tf

tf.random.set_seed(123)
x = tf.random.uniform(shape=(5,2), minval=0, maxval=10)
y = tf.range(5)

data_x = tf.data.Dataset.from_tensor_slices(x)
data_y = tf.data.Dataset.from_tensor_slices(y)

data_join = tf.data.Dataset.zip((data_x, data_y)) # zip need same dimensions both x an y
for item in data_join:
    print('x :', item[0].numpy(), 'y:', item[1].numpy())