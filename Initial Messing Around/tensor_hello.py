import tensorflow as tf
c = tf.constant('Testing Tensorflow')

# decode() is used because without returns a
# byte string (str is surrounded with b'<str>')

# first option
with tf.Session() as sess_one:
    print(sess_one.run(c).decode())

# second option
sess_two = tf.Session()
print(sess_two.run(c).decode())
