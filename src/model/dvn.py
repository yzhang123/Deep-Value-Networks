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
        self.update_rate = 10.0

    def conv_acti_layer(self, bottom, filter_shape, filter_depth, name, stride, padding='SAME'):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            pre_depth = bottom.get_shape()[3].value
            weights_shape = filter_shape + [pre_depth, filter_depth]
            weight = self._weight_variable(weights_shape, name=name)
            bias = self._bias_variable(filter_depth, name=name)
            conv = tf.nn.conv2d(bottom, weight, strides=strides, padding=padding)
            return tf.nn.relu(conv + bias)

    def fc_acti_layer(self, bottom, weight_shape, bias_shape, name, activation_fn=tf.nn.relu, dropout=False):
        with tf.variable_scope(name):
            W = self._weight_variable(weight_shape, name=name)
            b = self._bias_variable(bias_shape, name=name)
            preactivation = tf.matmul(bottom, W) + b
            tf.summary.histogram('pre_activations', preactivation)
            acti = activation_fn(preactivation, name=name + '_relu')
            tf.summary.histogram('activations', acti)
            if dropout:
                acti = tf.nn.dropout(acti, self._keep_prob, name=name + '_dropout')
            return acti

    def _weight_variable(self, shape, initializer=None, name=None):
        if not initializer:
            initializer = tf.contrib.layers.xavier_initializer()
        var = tf.get_variable(name + '_weight', shape, initializer=initializer, collections=['variables'])
        self.variable_summaries(var)
        return var

    def _bias_variable(self, shape, initializer=None, name=None):
        if not initializer:
            initializer = tf.constant_initializer(0.0001)
        var = tf.get_variable(name + '_bias', shape, initializer=initializer, collections=['variables'])
        self.variable_summaries(var)
        return var
    #
    # def _conv2d(self, x, W, stride=1):
    #   return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    # def _relu(self, conv, b):
    #     return tf.nn.relu(conv+b)

    # def _max_pool(self, x, size=2, stride=2):
    #   return tf.nn.max_pool(x, ksize=[1, size, size, 1],
    #                         strides=[1, stride, stride, 1], padding='SAME')

    def _create_loss(self, score, y, y_gt):
        """
        loss over batch
        :param score:   score value/model output, shape=[batchsize x 1]
        :param y    :   input segementation mask
        :param y_gt :   ground truth segmantation mask
        :return:
        """
        sim_score = _oracle_score(y, y_gt) # shape=[batchsize x 1]
        mean_sim_score = tf.reduce_mean(sim_score)
        tf.summary.scalar('sim_score', mean_sim_score)
        loss_CE = -sim_score * tf.log(score) - (1-sim_score) * tf.log(1-score)
        mean_loss_CE = tf.reduce_mean(loss_CE)
        tf.summary.scalar('cross_entropy', mean_loss_CE)
        return mean_loss_CE, mean_sim_score

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def build_network2(self, train=True):

        self._graph['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.variable_scope('input'):

            self._graph['x'] = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x-input') # 3+k between 0 and 1

            self._graph['y_gt'] = tf.placeholder(tf.float32, shape=[None, None, None, self._num_classes], name='y-gt') # ground truth segmentation

            self._graph['y1'] = tf.Variable(tf.zeros([self._batch_size, self._img_height, self._img_width, self._num_classes]), trainable=True, name='y-default',
                             dtype=tf.float32)

            self._graph['y'] = tf.placeholder_with_default(self._graph['y1'], shape=[None, None, None, self._num_classes], name="y-input")

        with tf.variable_scope('input-concat'):
            self._graph['identity'] = tf.identity(self._graph['y'], name='identity')
            self.x_concat = tf.concat([self._graph['x'], self._graph['identity']], 3, name='concat')

        with tf.variable_scope('conv1'):
            self._graph['conv1'] = self.conv_acti_layer(self.x_concat, [5,5], 64, "conv1", 1)
        with tf.variable_scope('conv2'):
            self._graph['conv2'] = self.conv_acti_layer(self._graph['conv1'], [5,5], 128, "conv2", 2)
        with tf.variable_scope('conv3'):
            self._graph['conv3'] = self.conv_acti_layer(self._graph['conv2'], [5,5], 128, "conv3", 2)


        self.conv3_flat = tf.reshape(self._graph['conv3'], [-1, 6*6*128], name='pre_fc')
        self._graph['fc1'] = self.fc_acti_layer(self.conv3_flat, weight_shape=[6 * 6 * 128, 384], bias_shape=[384], name='fc1', dropout=train)
        self._graph['fc2'] = self.fc_acti_layer(self._graph['fc1'], weight_shape=[384, 192], bias_shape=[192], name='fc2')
        self._graph['fc3'] = self.fc_acti_layer(self._graph['fc2'], weight_shape=[192, 1], bias_shape=[1], activation_fn=tf.nn.sigmoid, name='fc3')

        # name = 'fc2'
        # with tf.variable_scope('fc2'):
        #     W_fc2 = self._weight_variable()
        #     b_fc2 = self._bias_variable()
        #     self._graph['fc2'] = tf.nn.relu(tf.matmul(self._graph['fc1'], W_fc2) + b_fc2)
        #
        # with tf.variable_scope('fc3'):
        #     W_fc3 = self._weight_variable([192, 1])
        #     b_fc3 = self._bias_variable([1])
        #     self._graph['y_fc3'] = tf.matmul(self._graph['fc2'], W_fc3) + b_fc3
        #     self._graph['fc3'] = tf.nn.sigmoid(self._graph['y_fc3']) #batch_size x 1

        with tf.variable_scope('loss'):
            self._graph['loss'], self._graph['sim_score'] = self._create_loss(self._graph['fc3'], self._graph['y'], self._graph['y_gt'])
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self._learning_rate)
            self._graph['train_gradients'] = optimizer.compute_gradients(self._graph['loss'])
            self._graph['train_optimizer'] = optimizer.apply_gradients(self._graph['train_gradients'][1:], global_step=self._graph['global_step']) # protect y1:0 from being updated

        with tf.variable_scope('inference'):
            self._graph['inference_grad'] = tf.gradients(self._graph['fc3'], self._graph['identity'])
            #self._graph['inference_grad'] = tf.Print(self._graph['inference_grad'],
            #                                         [self._graph['inference_grad']],
            #                                         message="This is inference_grad: ",
            #                                         first_n=50)
            self._graph['inference_update'] = self._graph['y1'].assign(
                tf.clip_by_value(tf.add(self._graph['identity'], self.update_rate * self._graph['inference_grad'][0]), 0., 1.))
            #self._graph['inference_update'] = tf.Print(self._graph['inference_update'],
            #                                           [self._graph['inference_update']],
            #                                           message="This is inference_update: ",
            #                                           first_n=50)

        with tf.variable_scope('adverse'):
            self._graph['adverse_grad'] = tf.gradients(self._graph['fc3'], self._graph['identity'])
            self._graph['adverse_update'] = self._graph['y1'].assign(
                tf.clip_by_value(tf.add(self._graph['identity'], self._graph['inference_grad'][0]), 0., 1.))

        self._graph['merged_summary'] = tf.summary.merge_all()

        return self._graph




    def build_network(self, train=True):

        self._graph['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.variable_scope('input'):

            self._graph['x'] = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x-input') # 3+k between 0 and 1

            self._graph['y_gt'] = tf.placeholder(tf.float32, shape=[None, None, None, self._num_classes], name='y-gt') # ground truth segmentation

            self._graph['y'] = tf.placeholder(tf.float32, shape=[None, None, None, self._num_classes], name='y')

        with tf.variable_scope('input-concat'):
            self.x_concat = tf.concat([self._graph['x'], self._graph['y']], 3, name='concat')

        with tf.variable_scope('conv1'):
            self._graph['conv1'] = self.conv_acti_layer(self.x_concat, [5,5], 64, "conv1", 1)
        with tf.variable_scope('conv2'):
            self._graph['conv2'] = self.conv_acti_layer(self._graph['conv1'], [5,5], 128, "conv2", 2)
        with tf.variable_scope('conv3'):
            self._graph['conv3'] = self.conv_acti_layer(self._graph['conv2'], [5,5], 128, "conv3", 2)

        conv3_flat = tf.reshape(self._graph['conv3'], [-1, 6*6*128], name='pre_fc')
        self._graph['fc1'] = self.fc_acti_layer(conv3_flat, weight_shape=[6 * 6 * 128, 384], bias_shape=[384], name='fc1', dropout=train)
        self._graph['fc2'] = self.fc_acti_layer(self._graph['fc1'], weight_shape=[384, 192], bias_shape=[192], name='fc2')
        self._graph['fc3'] = self.fc_acti_layer(self._graph['fc2'], weight_shape=[192, 1], bias_shape=[1], activation_fn=tf.nn.sigmoid, name='fc3')

        with tf.variable_scope('loss'):
            self._graph['loss'], self._graph['sim_score'] = self._create_loss(self._graph['fc3'], self._graph['y'], self._graph['y_gt'])
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self._learning_rate)
            self._graph['train_gradients'] = optimizer.compute_gradients(self._graph['loss'])
            self._graph['train_optimizer'] = optimizer.apply_gradients(self._graph['train_gradients'], global_step=self._graph['global_step']) # protect y1:0 from being updated

        with tf.variable_scope('inference'):
            self._graph['inference_grad'] = tf.gradients(self._graph['fc3'], self._graph['y'])

        with tf.variable_scope('adverse'):
            self._graph['adverse_grad'] = tf.gradients(self._graph['loss'], self._graph['y'])

        self._graph['merged_summary'] = tf.summary.merge_all()

        return self._graph