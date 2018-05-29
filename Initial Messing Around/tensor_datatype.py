import tensorflow as tf

sess = tf.Session()

matrix = tf.constant([1, 2, 3])

# cast to floating point
f_matrix = tf.cast(matrix, dtype=tf.float32)

print('matrix type: ' + str(matrix.dtype))
print('matrix:\n' + str(sess.run(matrix)))
print('\nf_matrix type: ' + str(f_matrix.dtype))
print('f_matrix:\n' + str(sess.run(f_matrix)))

"""
Can specify in tensor creation, otherwise, tensorflow chooses.
tf.int32 default for integers
tf.float32 default for floats
"""
