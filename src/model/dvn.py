# -*- coding: utf-8 -*-
# Compatibility to python 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


import tensorflow as tf
import numpy as np
import sys
import logging
from dvn.src.util.loss import _oracle_score
from tensorflow.contrib.layers import layer_norm
from nn_toolbox.src.tf.tf_extend.tf_helpers import count_variables
from nn_toolbox.src.tf.tf_extend.metrics import r_squared
from tensorflow.contrib.layers import flatten

class DvnNet(object):
    def __init__(self, classes=None, batch_size=1, img_height=24, img_width=24, lr = 0.01):
        self._batch_size = batch_size
        self._img_width = img_width
        self._img_height = img_height
        self._classes = classes
        self._num_classes = len(classes)
        self._graph = {}
        self._learning_rate = lr
        self._keep_prob = 0.75
        self.regularizer=tf.nn.l2_loss
        self.weight_decay = 0.001

    def conv_acti_layer(self, bottom, filter_shape, filter_depth, name, stride, padding='SAME', layernorm=False):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            pre_depth = bottom.get_shape()[3].value
            weights_shape = filter_shape + [pre_depth, filter_depth]
            weight = self._weight_variable(weights_shape, name=name)
            bias = self._bias_variable(filter_depth, name=name)
            conv = tf.nn.conv2d(bottom, weight, strides=strides, padding=padding)

            if layer_norm:
                norm = layer_norm(conv)
                relu = tf.nn.relu(norm + bias)
            else:
                relu = tf.nn.relu(conv + bias)

            return relu

    def fc_acti_layer(self, bottom, weight_shape, bias_shape, name, activation_fn=tf.nn.relu, dropout=False, layernorm=False):
        with tf.variable_scope(name):
            W = self._weight_variable(weight_shape, name=name)
            b = self._bias_variable(bias_shape, name=name)
            preactivation = tf.matmul(bottom, W)
            tf.summary.histogram('pre_norm_activations', preactivation)
            if layernorm:
                norm = layer_norm(preactivation)
                acti = activation_fn(norm + b, name=name + '_relu')
            else:
                acti = activation_fn(preactivation + b, name=name + '_relu')
            tf.summary.histogram('activations', acti)
            if dropout:
                acti = tf.nn.dropout(acti, self._keep_prob, name=name + '_dropout')
            return acti

    def _weight_variable(self, shape, initializer=None, name=None):
        if not initializer:
            initializer = tf.contrib.layers.xavier_initializer()
        var = tf.get_variable(name + '_weight', shape, initializer=initializer, collections=['variables'])
        self.variable_summaries(var, 'weight')
        self._graph['loss'] += self.regularizer(var) * self.weight_decay
        return var

    def _bias_variable(self, shape, initializer=None, name=None):
        if not initializer:
            initializer = tf.constant_initializer(0.01)
        var = tf.get_variable(name + '_bias', shape, initializer=initializer, collections=['variables'])
        self.variable_summaries(var, name='bias')
        return var

    def _create_loss(self, score, y, y_gt):
        """
        loss over batch
        :param score:   score value/model output, shape=[batchsize x 1]
        :param y    :   input segementation mask
        :param y_gt :   ground truth segmantation mask
        :return:
        """
        sim_score = _oracle_score(y, y_gt) # shape=[batchsize x 1]
        self._graph['sim_score_vector'] = sim_score
        mean_sim_score = tf.reduce_mean(sim_score)
        tf.summary.scalar('mean_sim_score', mean_sim_score)
        loss_CE = tf.square(tf.subtract(sim_score, score))
        #loss_CE = -sim_score * tf.log(score) - (1-sim_score) * tf.log(1-score)
        mean_loss_CE = tf.reduce_mean(loss_CE)
        tf.summary.scalar('mean_loss', mean_loss_CE)
        logging.info("shape %s" % score.shape)
        self._graph['score_diff'] = tf.subtract(sim_score, tf.reshape(score, [-1]))
        tf.summary.scalar('mean_simscore-netoutput', tf.reduce_mean(self._graph['score_diff']))
        tf.summary.scalar('r_squared', r_squared(targets=sim_score, logits=tf.reshape(score, [-1])))
        return mean_loss_CE, mean_sim_score

    def variable_summaries(self, var, name=''):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean_'+ name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev_' + name, stddev)
            tf.summary.scalar('max_'+ name, tf.reduce_max(var))
            tf.summary.scalar('min_'+ name, tf.reduce_min(var))
            tf.summary.histogram('histogram_'+ name, var)



    def build_network(self, train=True):

        self._graph['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.variable_scope('input'):

            self._graph['x'] = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x-input') # 3+k between 0 and 1
            tf.summary.image("x", self._graph['x'])
            self._graph['y_gt'] = tf.placeholder(tf.float32, shape=[None, None, None, self._num_classes], name='y-gt') # ground truth segmentation
            self._graph['y'] = tf.placeholder(tf.float32, shape=[None, None, None, self._num_classes], name='y')
            self._graph['y_mean'] = tf.reduce_mean(self._graph['y'], [1,2])[:, 0]
        with tf.variable_scope('input-concat'):
            self.x_concat = tf.concat([self._graph['x'], self._graph['y']], 3, name='concat')

        self._graph['loss'] = 0
        with tf.variable_scope('conv1'):
            self._graph['conv1'] = self.conv_acti_layer(self.x_concat, [5,5], 64, "conv1", 1, layernorm=False)
        with tf.variable_scope('conv2'):
            self._graph['conv2'] = self.conv_acti_layer(self._graph['conv1'], [5,5], 128, "conv2", 2, layernorm=False)
        with tf.variable_scope('conv3'):
            self._graph['conv3'] = self.conv_acti_layer(self._graph['conv2'], [5,5], 128, "conv3", 2, layernorm=False)

        conv3_flat = tf.reshape(self._graph['conv3'], [-1, 6*6*128], name='pre_fc')
        self._graph['fc1'] = self.fc_acti_layer(conv3_flat, weight_shape=[6 * 6 * 128, 384], bias_shape=[384], name='fc1', dropout=False, layernorm=False)
        self._graph['fc2'] = self.fc_acti_layer(self._graph['fc1'], weight_shape=[384, 192], bias_shape=[192], name='fc2', layernorm=False)
        self._graph['fc3'] = self.fc_acti_layer(self._graph['fc2'], weight_shape=[192, 1], bias_shape=[1], activation_fn=tf.nn.sigmoid, name='fc3')

        with tf.variable_scope('loss'):
            self._graph['loss'], self._graph['sim_score'] = self._create_loss(self._graph['fc3'], self._graph['y'], self._graph['y_gt'])
        with tf.variable_scope('train'):
            #optimizer = tf.train.AdamOptimizer(self._learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
            self._graph['train_gradients'] = optimizer.compute_gradients(self._graph['loss'])
            self._graph['train_optimizer'] = optimizer.apply_gradients(self._graph['train_gradients'], global_step=self._graph['global_step']) # protect y1:0 from being updated

        with tf.variable_scope('inference'):
            self._graph['inference_grad'] = tf.gradients(self._graph['fc3'], self._graph['y'])
            tf.summary.histogram("histogram_gradient", self._graph['inference_grad'])

        with tf.variable_scope('adverse'):
            self._graph['adverse_grad'] = tf.gradients(self._graph['loss'], self._graph['y'])
            tf.summary.histogram("histogram_gradient", self._graph['adverse_grad'])

        self._graph['merged_summary'] = tf.summary.merge_all()

        n_vars = count_variables()
        logging.info("number of variables %s" % n_vars)
        return self._graph