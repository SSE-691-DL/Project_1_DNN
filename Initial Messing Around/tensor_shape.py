import tensorflow as tf

sess = tf.Session()

matrix = tf.Variable([[3, 4, 5, 6], [7, 6, 5, 4],
                      [9, 3, 1, 9], [0, 8, 4, 2]], tf.int32)

# vector of zeros with same number of columns as matrix
c_zeros = tf.zeros(matrix.shape[1])
zeros = tf.zeros(matrix.shape)
print('c_zeros: ' + str(sess.run(c_zeros)))
print('zeros:\n' + str(sess.run(zeros)))


# reshaping a tensor object - all objects have same number of elements

rank_three_tensor = tf.ones([3, 4, 5])

# reshape into a 6x10 matrix
m_1 = tf.reshape(rank_three_tensor, [6, 10])

# reshape into a 3x20 matrix
# -1 tells reshape to calculate the size of this dimension
m_2 = tf.reshape(m_1, [3, -1])

# reshape into a 4x3x5 tensor
m_3 = tf.reshape(m_2, [4, 3, -1])

print('\nrank_three_tensor:\n' + str(sess.run(rank_three_tensor)))
print('\nm_1:\n' + str(sess.run(m_1)))
print('\nm_2:\n' + str(sess.run(m_2)))
print('\nm_3:\n' + str(sess.run(m_3)))

"""
Example of error generation:
Number of elements in reshape must match the original number of elements.
Errors are generated if this is not true -
    m_err = tf.reshape(m_3, [13, 2, -1])
"""
