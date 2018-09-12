import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.Session()
data_dir = 'MNIST_data/'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

in_units = 784
h1_units = 300
x = tf.placeholder(tf.float32, [None, in_units])
# Weights before ReLU layer
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
# Weights before Softmax layer
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
# Use labels to calculate entropy
y_ = tf.placeholder(tf.float32, [None, 10])
# * is element-wise multiplication; tf.matmul is matrix multiplication
h = -tf.reduce_sum(y_*tf.log(y), axis=1)
cross_entropy = tf.reduce_mean(h)
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# Print accuracy
with sess.as_default():
    corrected_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(corrected_prediction, tf.float32))
    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.75}))
