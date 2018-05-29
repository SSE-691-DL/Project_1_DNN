import tensorflow as tf
import matplotlib.pyplot as plt # dataset visualization
import numpy as np              # low-level numerical Python library
import pandas as pd             # higher-level numerical Python library

g = tf.Graph() # create graph

# establish the graph as the default graph
with g.as_default():
    # assemble a graph consisting of the following 3 operations:
    #   * two tf.constant operations to create the operands
    #   * one tf.add operation to add the 2 operands
    x = tf.constant(8, name="x_const")
    y = tf.constant(7, name="y_const")
    xy = tf.add(x, y, name="xy_sum")
    z = tf.constant(3, name="z_const")
    xyz = tf.add_n([x,y,z], name="xyz_sum")

    # create session to run the default graph
    with tf.Session() as sess:
        print('x: ' + str(sess.run(x)))
        print('y: ' + str(sess.run(y)))
        print('XY Sum:')
        print(xy.eval())
        print('z: ' + str(sess.run(z)))
        print('XYZ Sum:')
        print(xyz.eval())
