# -*- coding: utf-8 -*-
# Compatibility to python 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


import tensorflow as tf
import numpy as np
import sys
#print(sys.path)
from dvn.src.util.loss import _oracle_score


class DvnNet(object):
    def __init__(self, classes=None, batch_size=1, img_height=24, img_width=24):
        self._batch_size = batch_size
        self._img_width = img_width
        self._img_height = img_height
        self._classes = classes
        self._num_classes = len(classes)
        self._graph = {}
        #self._learning_rate = 0.01
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

    def _weight_variable(self, shape, name=None):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

    def _bias_variable(self, shape, name=None):
      initial = tf.constant(0.001, shape=shape)
      return tf.Variable(initial, name=name)

    def _conv2d(self, x, W, stride=1):
      return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def _relu(self, conv, b):
        return tf.nn.relu(conv+b)

    def _max_pool(self, x, size=2, stride=2):
      return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                            strides=[1, stride, stride, 1], padding='SAME')

    def _create_loss(self, score, y, y_gt):
        """
        loss over batch
        :param score: score value/model output, shape=[batchsize x 1]
        :param y: input segementation mask
        :param y_gt: ground truth segmantation mask
        :return:
        """
        sim_score = _oracle_score(y, y_gt) # shape=[batchsize x 1]
        loss_CE = -sim_score * tf.log(score) - (1-sim_score) * tf.log(1-score)
        return tf.reduce_mean(loss_CE, 0), sim_score


    def build_network(self, train=True):

        self._graph['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self._graph['x'] = tf.placeholder(tf.float32, shape=[None, None, None, 3]) # 3+k between 0 and 1

        self._graph['y_gt'] = tf.placeholder(tf.float32, shape=[None, None, None, self._num_classes]) # ground truth segmentation

        self._graph['y1'] = tf.Variable(tf.zeros([self._batch_size, self._img_height, self._img_width, self._num_classes]), trainable=True, name='y1',
                         dtype=tf.float32)

        self._graph['y'] = tf.placeholder_with_default(self._graph['y1'], shape=[None, None, None, self._num_classes], name="y")

        self._graph['identity'] = tf.identity(self._graph['y'])

        self.x_concat = tf.concat([self._graph['x'], self._graph['identity']], 3)

        self._graph['conv1'] = self.conv_acti_layer(self.x_concat, [5,5], 64, "conv1", 1)
        self._graph['conv2'] = self.conv_acti_layer(self._graph['conv1'], [5,5], 128, "conv2", 2)
        self._graph['conv3'] = self.conv_acti_layer(self._graph['conv2'], [5,5], 128, "conv3", 2)

        self.conv3_flat = tf.reshape(self._graph['conv3'], [-1, 6*6*128])
        W_fc1 = self._weight_variable([6 * 6 * 128, 384])
        b_fc1 = self._bias_variable([384])
        self._graph['fc1'] = tf.nn.relu(tf.matmul(self.conv3_flat, W_fc1) + b_fc1)
        #keep_prob = tf.placeholder(tf.float32)
        if train:
            self._graph['fc1'] = tf.nn.dropout(self._graph['fc1'], self._keep_prob)

        W_fc2 = self._weight_variable([384, 192])
        b_fc2 = self._bias_variable([192])
        self._graph['fc2'] = tf.nn.relu(tf.matmul(self._graph['fc1'], W_fc2) + b_fc2)

        W_fc3 = self._weight_variable([192, 1])
        b_fc3 = self._bias_variable([1])
        self._graph['y_fc3'] = tf.matmul(self._graph['fc2'], W_fc3) + b_fc3

        self._graph['fc3'] = tf.nn.sigmoid(self._graph['y_fc3']) #batch_size x 1
        self._graph['loss'], self._graph['sim_score'] = self._create_loss(self._graph['fc3'], self._graph['y'], self._graph['y_gt'])

        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        self._graph['train_gradients'] = optimizer.compute_gradients(self._graph['loss'])
        self._graph['train_optimizer'] = optimizer.apply_gradients(self._graph['train_gradients'][1:], global_step=self._graph['global_step']) # protect y1:0 from being updated

        self._graph['inference_grad'] = tf.gradients(self._graph['fc3'], self._graph['identity'])
        #self._graph['inference_grad'] = tf.Print(self._graph['inference_grad'],
        #                                         [self._graph['inference_grad']],
        #                                         message="This is inference_grad: ",
        #                                         first_n=50)
        self._graph['inference_update'] = self._graph['y1'].assign(tf.clip_by_value(tf.subtract(self._graph['identity'], self._graph['inference_grad'][0]), 0., 1.))
        #self._graph['inference_update'] = tf.Print(self._graph['inference_update'],
        #                                           [self._graph['inference_update']],
        #                                           message="This is inference_update: ",
        #                                           first_n=50)


        self._graph['adverse_grad'] = tf.gradients(self._graph['fc3'], self._graph['identity'])
        self._graph['adverse_update'] = self._graph['y1'].assign(
            tf.clip_by_value(tf.add(self._graph['identity'], self._graph['inference_grad'][0]), 0., 1.))


        return self._graph




