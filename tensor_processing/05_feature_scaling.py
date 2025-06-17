import tensorflow as tf

tf.random.set_seed(123)
x = tf.random.uniform(shape=(5,2), minval=0, maxval=10)
y = tf.range(5)

data_x = tf.data.Dataset.from_tensor_slices(x)
data_y = tf.data.Dataset.from_tensor_slices(y)

data_join = tf.data.Dataset.zip((data_x, data_y)) # zip need same dimensions both x an y


data_scaled = data_join.map(lambda x, y: (x / 10.0, y))  # Scale x by dividing by 10
for item in data_scaled:
    print('x :', item[0].numpy(), 'y:', item[1].numpy())
# This code scales the x values in the dataset by dividing them by 10.