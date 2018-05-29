import tensorflow as tf

with tf.Session() as sess:
    c = tf.constant([1, 2, 3])
    tensor = c * c
    print(tensor.eval())

    p = tf.placeholder(tf.float32)
    t = p + 1.0
    # print(t.eval()) - Fails with no dynamic placeholder
    print(t.eval(feed_dict={p:2.0}))
