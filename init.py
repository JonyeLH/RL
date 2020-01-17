#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

value = [0, 1, 2, 3, 4, 5, 6, 7]
init = tf.constant_initializer(value)

with tf.Session() as sess:
    x = tf.get_variable('x', shape=[8], initializer=init)
    x.initializer.run()
    print(x.eval())

# output:
# [ 0.  1.  2.  3.  4.  5.  6.  7.]