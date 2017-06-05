#! /usr/bin/env python


# Compatibility to python 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np


class DvnNet(object):
    def __init__(self, batch_size=1, classes=None):
        self._batch_size = batch_size
        self._classes = classes
        self._num_classes = len(classes)

    def _weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def _bias_variable(self, shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def _conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


    def build_network(self):
        x1 = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3]) # 3+k
        x2 = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, self._num_classes])


        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1,28,28,1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


        # The raw formulation of cross-entropy,
        #
        #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
        #                                 reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
        # outputs of 'y', and then average across the batch.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        sess.run(tf.global_variables_initializer())

        vgg = scipy.io.loadmat(path)
        vgg_layers = vgg['layers']

        graph = {}
        graph['conv1_1'] = _conv2d_relu(vgg_layers, input_image, 0, 'conv1_1')
        graph['conv1_2'] = _conv2d_relu(vgg_layers, graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = _avgpool(graph['conv1_2'])
        graph['conv2_1'] = _conv2d_relu(vgg_layers, graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2'] = _conv2d_relu(vgg_layers, graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = _avgpool(graph['conv2_2'])
        graph['conv3_1'] = _conv2d_relu(vgg_layers, graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2'] = _conv2d_relu(vgg_layers, graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3'] = _conv2d_relu(vgg_layers, graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4'] = _conv2d_relu(vgg_layers, graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = _avgpool(graph['conv3_4'])
        graph['conv4_1'] = _conv2d_relu(vgg_layers, graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2'] = _conv2d_relu(vgg_layers, graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3'] = _conv2d_relu(vgg_layers, graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4'] = _conv2d_relu(vgg_layers, graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = _avgpool(graph['conv4_4'])
        graph['conv5_1'] = _conv2d_relu(vgg_layers, graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2'] = _conv2d_relu(vgg_layers, graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3'] = _conv2d_relu(vgg_layers, graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4'] = _conv2d_relu(vgg_layers, graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = _avgpool(graph['conv5_4'])

        return graph