# Exploration of tensor ranks

import tensorflow as tf

sess = tf.Session()

# Rank 0 (Scalar) Variables (strings are treated as single items in tensorflow)
mammal = tf.Variable("Elephant", tf.string)
integer = tf.Variable(45, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
complicated = tf.Variable(2.2 - 4.49j, tf.complex64)

# Rank 1 (Vector) Variables
str_ex = tf.Variable(["val1", "val2"], tf.string)
numbers = tf.Variable([3.14159, 2.71828], tf.float32)
primes = tf.Variable([1, 2, 5, 7, 11], tf.int32)
multi_comp = tf.Variable([2.2 - 4.49j, 4.3 - 5.33j], tf.complex64)

# Rank 2 (Matrix) Variables - Need a row and column
mat = tf.Variable([[7],[11]], tf.int16)
mat_bool = tf.Variable([[False, True],[True, False]], tf.bool)
mat_sqrs = tf.Variable([[1], [4], [9], [16]], tf.int32)
mat_sqrs2 = tf.Variable([[1, 4], [9, 16]], tf.int32)

# Higher ranks - follow 2's footsteps
# Rank 4 popular in image processing
# (example-in-batch, image width, image height, color channel)
image = tf.zeros([10, 299, 299, 3]) # batch x width x height x cc

# Getting object rank
r_image = tf.rank(image)
r_mat = tf.rank(mat)

# Getting slices (access cell in tensor by n-indices)
vector = tf.Variable([1, 2, 5, 7, 11], tf.int32)
scalar = vector[3]
matrix = tf.Variable([[1, 2, 3, 4], [11, 12, 13, 14]], tf.int32)
scalar2 = matrix[1, 2]
row_vector = matrix[1]
col_vector = matrix[:, 2] # : slices, leaving dimension alone

# initialize the variables (note: not required for image)
init = tf.global_variables_initializer()
sess.run(init)

print('mammal: ' + str(mammal))
print('mammal: ' + str(sess.run(mammal).decode()))
print('\ninteger: ' + str(integer))
print('integer: ' + str(sess.run(integer)))
print('\nfloating: ' + str(floating))
print('floating: ' + str(sess.run(floating)))
print('\ncomplicated: ' + str(complicated))
print('complicated: ' + str(sess.run(complicated)))
print('\nstr_ex: ' + str(str_ex))
print('str_ex: ' + str(sess.run(str_ex)))
print('\nnumbers: ' + str(numbers))
print('numbers: ' + str(sess.run(numbers)))
print('\nprimes: ' + str(primes))
print('primes: ' + str(sess.run(primes)))
print('\nmulti_comp: ' + str(multi_comp))
print('multi_comp: ' + str(sess.run(multi_comp)))
print('\nmat: ' + str(mat))
print('mat:\n' + str(sess.run(mat)))
print('\nmat_bool: ' + str(mat_bool))
print('mat_bool:\n' + str(sess.run(mat_bool)))
print('\nmat_sqrs: ' + str(mat_sqrs))
print('mat_sqrs:\n' + str(sess.run(mat_sqrs)))
print('\nmat_sqrs2: ' + str(mat_sqrs2))
print('mat_sqrs2:\n' + str(sess.run(mat_sqrs2)))

print('\nImage: ' + str(image))
print('Image: \n' + str(sess.run(image)))

print('\nRank of image: ' + str(r_image))
print('Rank of image: ' + str(sess.run(r_image)))
print('Rank of mat: ' + str(r_mat))
print('Rank of mat: ' + str(sess.run(r_mat)))

print('\nvector: ' + str(vector))
print('vector: ' + str(sess.run(vector)))
print('\nScalar from vector: ' + str(scalar))
print('Scalar from vector: ' + str(sess.run(scalar)))
print('\nmatrix: ' + str(matrix))
print('matrix:\n' + str(sess.run(matrix)))
print('\nscalar from matrix: ' + str(scalar2))
print('scalar from matrix: ' + str(sess.run(scalar2)))
print('\nrow from matrix: ' + str(row_vector))
print('row from matrix: ' + str(sess.run(row_vector)))
print('\ncol from matrix: ' + str(col_vector))
print('col from matrix: ' + str(sess.run(col_vector)))
