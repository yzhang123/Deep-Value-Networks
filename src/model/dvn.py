# -*- coding: utf-8 -*-
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
    def __init__(self, classes=None, batch_size=1, img_width=24, img_height=24):
        self._batch_size = batch_size
        self._img_width = img_width
        self._img_height = img_height
        self._classes = classes
        self._num_classes = len(classes)
        self._graph = {}
        self._learning_rate = 0.0001
        self._keep_prob = 0.75

    def conv_acti_layer(self, bottom, filter_shape, filter_depth, name, stride, padding='SAME'):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            pre_depth = bottom.get_shape()[3].value
            weights_shape = filter_shape + [pre_depth, filter_depth]

            weight = tf.get_variable(name + "_weight", weights_shape,
                                     initializer=tf.orthogonal_initializer(gain=1.0, seed=None),
                                     collections=['variables'])
            bias = tf.get_variable(name + "_bias", filter_depth, initializer=tf.constant_initializer(0.001),
                                   collections=['variables'])

            conv = tf.nn.conv2d(bottom, weight, strides=strides, padding=padding)
            return tf.nn.relu(conv + bias)

    def _weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def _bias_variable(self, shape):
      initial = tf.constant(0.001, shape=shape)
      return tf.Variable(initial)

    def _conv2d(self, x, W, stride=1):
      return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def _relu(self, conv, b):
        return tf.nn.relu(conv+b)

    def _max_pool(self, x, size=2, stride=2):
      return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                            strides=[1, stride, stride, 1], padding='SAME')

    def _oracle_score(self, y, y_gt):
        """
        also referred to as v*
        :param y: segmantation masks [0, 1] * image_size x num_classes
        :param y_gt: ground truth segmantation mask
        :return: relaxed IoU score between both
        """

        y_min = tf.reduce_sum(tf.minimum(y, y_gt), [1,2])
        y_max = tf.reduce_sum(tf.maximum(y, y_gt), [1,2])
        y_divide = tf.divide(y_min, y_max)
        return tf.reduce_mean(y_divide, 1)

    def _create_loss(self, score, y, y_gt):
        """
        loss over batch
        :param score: score value/model output, shape=[batchsize x 1]
        :param y: input segementation mask
        :param y_gt: ground truth segmantation mask
        :return:
        """
        sim_score = self._oracle_score(y, y_gt) # shape=[batchsize x 1]
        loss_CE = -sim_score * tf.log(score) - (1-sim_score) * tf.log(1-score)
        return tf.reduce_mean(loss_CE, 0), sim_score


    def build_network(self):

        x = tf.placeholder(tf.float32, shape=[None, None, None, 3]) # 3+k between 0 and 1
        self._graph['x'] = x
        #y = tf.placeholder(tf.float32, shape=[None, None, None, self._num_classes]) # between zero and 1
        y = tf.Variable(tf.zeros([self._batch_size, self._img_width, self._img_height, self._num_classes]),
                        dtype=tf.float32)
        self._graph['y'] = y
        y_gt = tf.placeholder(tf.float32, shape=[None, None, None, self._num_classes]) # ground truth segmentation

        self._graph['y_gt'] = y_gt

        self._graph['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        x_concat = tf.concat([x, y], 3)

        self._graph['conv1'] = self.conv_acti_layer(x_concat, [5,5], 64, "conv1", 1)
        self._graph['conv2'] = self.conv_acti_layer(self._graph['conv1'], [5,5], 128, "conv2", 2)
        self._graph['conv3'] = self.conv_acti_layer(self._graph['conv2'], [5,5], 128, "conv3", 2)


        conv3_flat = tf.reshape(self._graph['conv3'], [-1, 6*6*128])

        W_fc1 = self._weight_variable([6 * 6 * 128, 384])
        b_fc1 = self._bias_variable([384])
        h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)


        self._graph['fc1'] = h_fc1

        #keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self._keep_prob)

        W_fc2 = self._weight_variable([384, 192])
        b_fc2 = self._bias_variable([192])
        self._graph['b_fc2'] = b_fc2
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        self._graph['fc2'] = h_fc2

        W_fc3 = self._weight_variable([192, 1])
        self._graph['W_fc3'] = W_fc3
        b_fc3 = self._bias_variable([1])
        self._graph['b_fc3'] = b_fc3
        y_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3

        self._graph['y_fc3'] = y_fc3
        o_fc3 = tf.nn.sigmoid(y_fc3) #batch_size x 1


        self._graph['fc3'] = o_fc3

        loss, sim_score = self._create_loss(o_fc3, y, y_gt)
        self._graph['loss'] = loss
        self._graph['sim_score'] = sim_score

        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._graph['loss'], global_step=self._graph['global_step'])
        self._graph['train_optimizer'] = optimizer

        inference_optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._graph['loss'], global_step=self._graph['global_step'], var_list=[self._graph['y']])
        self._graph['inference_optimizer'] = inference_optimizer

        return self._graph