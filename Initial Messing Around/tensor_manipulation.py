import tensorflow as tf

# vector addition
with tf.Graph().as_default():
    # create a 6-element vector (1-D tensor)
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

    # create 6-element vector. each element in the vector will
    # be initialized to 1. first argument is in shape of the
    # tensor.
    ones = tf.ones([6], dtype=tf.int32)

    # add the two vectors. result is 6 element vector
    plus_primes = tf.add(primes, ones)

    # run default graph in session
    with tf.Session() as sess:
        print(plus_primes.eval())

# shapes again
with tf.Graph().as_default():
    # scalar (0-D tensor)
    scalar = tf.zeros([])

    # vector with 3 elements
    vector = tf.zeros([3])

    # matrix with 2 rows and 3 columns
    matrix = tf.zeros([2, 3])

    with tf.Session() as sess:
        print('Scalar shape: ', scalar.get_shape())
        print('\nScalar value: ', scalar.eval())
        print('\nVector shape: ', vector.get_shape())
        print('\nVector value: ', vector.eval())
        print('\nMatrix shape: ', matrix.get_shape())
        print('\nMatrix value:\n', matrix.eval())

# broadcasting
with tf.Graph().as_default():
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
    ones = tf.constant(1, dtype=tf.int32)
    plus_primes = tf.add(primes, ones)
    with tf.Session() as sess:
        print("\n", plus_primes.eval())

# matrix multiplication
with tf.Graph().as_default():
    x = tf.constant([[5, 2, 4, 3],
                    [5, 1, 6, -2],
                    [-1, 3, -1, -2]],
                    dtype=tf.int32)
    y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)
    result = tf.matmul(x, y)
    with tf.Session() as sess:
        print("\nmultiplication:\n", result.eval())

# reshaping again
with tf.Graph().as_default():
    matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8],
                          [9, 10], [11, 12], [13, 14], [15, 16]],
                          dtype=tf.int32)
    rs_2x8_m = tf.reshape(matrix, [2, 8])
    rs_4x4_m = tf.reshape(matrix, [4, 4])
    rs_2x2x4_m = tf.reshape(matrix, [2, 2, 4])
    rs_1_m = tf.reshape(matrix, [16])
    with tf.Session() as sess:
        print('\nOriginal:\n', matrix.eval())
        print('\n2x8:\n', rs_2x8_m.eval())
        print('\n4x4:\n', rs_4x4_m.eval())
        print('\n2x2x4:\n', rs_2x2x4_m.eval())
        print('\n1 dimensional:\n', rs_1_m.eval())

# variable initializiation
with tf.Graph().as_default():
    v = tf.Variable([3])
    w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))
    with tf.Session() as sess:
        try:
            v.eval()
        except tf.errors.FailedPreconditionError as e:
                print('\nCaught: ', e)
        sess.run(tf.global_variables_initializer())
        print(w.eval())
        print(w.eval())

        print('\ninitial: ', v.eval())
        assignment = tf.assign(v, [7])
        print('new unchanged: ', v.eval())
        sess.run(assignment)
        print('new changed: ', v.eval())

# reshape to multiply
with tf.Graph().as_default():
    with tf.Session() as sess:
        a = tf.constant([5, 3, 2, 7, 1, 4])
        b = tf.constant([4, 6, 3])
        print('\na: ', a.eval())
        print('b: ', b.eval())

        # reshape so numCol == numRow for a,b
        r_a = tf.reshape(a, [2, 3])
        r_b = tf.reshape(b, [3, 1])
        print('\nr_a:\n', r_a.eval())
        print('\nr_b: \n', r_b.eval())
        
        c = tf.matmul(r_a, r_b)
        print('\nc:\n', c.eval())

        r_a_2 = tf.reshape(a, [6, 1])
        r_b_2 = tf.reshape(b, [1, 3])
        print('\nr_a_2:\n', r_a_2.eval())
        print('\nr_b_2: \n', r_b_2.eval())

        c_2 = tf.matmul(r_a_2, r_b_2)
        print('\nc_2:\n', c_2.eval())

# dice simulation
with tf.Graph().as_default(), tf.Session() as sess:
    d1 = tf.Variable(tf.random_uniform([10, 1],
                                       minval=1,
                                       maxval=7,
                                       dtype=tf.int32))
    d2 = tf.Variable(tf.random_uniform([10, 1],
                                       minval=1,
                                       maxval=7,
                                       dtype=tf.int32))
    d_sum = tf.add(d1, d2)
    result = tf.concat(values=[d1, d2, d_sum], axis=1)
    sess.run(tf.global_variables_initializer())
    print('\nd1:\n', d1.eval())
    print('\nd2:\n', d2.eval())
    print('\ndice result:\n', result.eval())
