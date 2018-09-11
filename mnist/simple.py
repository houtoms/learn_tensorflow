import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
sess = tf.Session()
data_dir = 'MNIST_data/'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
# * is element-wise mul; tf.matmul is matrix multiplication
h = -tf.reduce_sum(y_*tf.log(y), axis=1)
cross_entropy = tf.reduce_mean(h)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

with sess.as_default():
    corrected_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(corrected_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
