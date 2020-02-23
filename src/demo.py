import tensorflow as tf
x = tf.placeholder("float")
y = 2 * x
data = tf.ones([4, 5], tf.int32)
with tf.Session() as sess:
    x_data = sess.run(data)
    print(sess.run(y, feed_dict = {x:x_data}))