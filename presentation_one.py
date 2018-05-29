import tensorflow as tf

# batch size (100), 28x28 greyscale images, 1 = greyscale, 3 = rgb, etc.
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# Training = computing variables W and b
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

# model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)
# placeholder for correct answers
Y_ = tf.placeholder(ft.float32, [None, 10])

""" Five Layers:
K = 200
L = 100
M = 60
N = 30

W1 = tf.Variable(tf.truncated_normal([28*28, K], stddev=0.1))
B1 = tf.Variable(zeros([K]))
W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(zeros([L]))
W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(zeros([M]))
W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(zeros([N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(zeros([10]))

X = tf.reshape(X, [-1, 28*28])

Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
Y = tf.nn.sigmoid(tf.matmul(Y4, W5) + B5)

# better activation function option (don't know why)
# it is not bound/flat
# Y = tf.nn.relu(tf.matmul(X, W) + b)
"""

""" Dropout:
# pkeep probability that each neuron remains in network
# if 75%, shoot 25% of neurons
# in vector representing output, replace 25% of values by 0
# remaining values are slightly boosted to not change overall output/avg

pkeep = tf.placeholder(tf.float32)
Yf = tf.nn.relu(tf.matmul(X, W) + B)
Y = tf.nn.dropout(Yf, pkeep)
"""

# loss function (cross entropy)
# sum across vector (reduce sum)
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers fount in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Gradient descent optimizer => one of simplest optimizers in library
# tell optimizer, please minimize cross entropy
# computes gradient of this function (vector of all partial derivatives relative
# to all weights and biases in system)
# gradient points down (arrow to smallest loss) (really up, but minus sign)
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    # train
    sess.run(train_step, feed_dict = train_data)

    if (i % 100 == 0):
        # success?
        a,c = sess.run([accuracy, cross_entropy], feed_dict = train_data)

        # success on test data?
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a,c = sess.run([accuracy, cross_entropy], feed = test_data)
