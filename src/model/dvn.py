# -*- coding: utf-8 -*-
# Compatibility to python 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


import tensorflow as tf
import numpy as np
import sys
import logging
from tensorflow.contrib.layers import layer_norm
from nn_toolbox.src.tf.tf_extend.tf_helpers import count_variables
from nn_toolbox.src.tf.tf_extend.metrics import r_squared
from nn_toolbox.src.tf.blocks.basic_blocks import batch_renorm, get_bias_value, apply_dropout, layer_norm
from tensorflow.contrib.layers import flatten


regularizer = tf.nn.l2_loss

class DvnNet(object):
    def __init__(self, input_height=24, input_width=24, learning_rate = 0.0001, num_classes=None, weight_decay=None, keep_prob=None):
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.keep_prob = keep_prob
        self.regularizer = regularizer
        self.summary_train=list()
        self.summary_test=list()

    def conv(self, input, num_outputs, name, filter=(3, 3), stride=1, activation_fn=None, use_batch_norm=False,
             initializer=None, dropout=None, collection=None, use_layer_norm=False):
        """ 2D convolution with additional processing steps. In detail:
         (optional) batchnorm -> (optional) activation) -> 2D convolution -> (optional) dropout
         For padding, zero padding is used.

        :param input: the data is stored in the order of: [batch, in_height, in_width, in_channels]
        :param num_outputs: number of output channels
        :param name: name of the scope in which all operations are placed
        :param filter: 2-tuple defining the kernel size
        :param stride: stride of convolution (will be assumed to be symmetric)
        :param activation_fn: tf activation function to use. If none, no activation will be applied
        :param use_batch_norm: If True, a batch norm layer will be added. This can also be a tf tensor.
        :param initializer: initializer to use for variable initialization
        :param dropout: Specifying the keep probability of dropout layer. This can be a tensor or a float. If None, no
            dropout will be applied
        :param collection: If provided with a list, all variables are added to that list
        :param use_layer_norm: Alternative to batch normalization. Layer normalization behaves identical in train and inference
        :return: convoluted output tensor
        """
        assert isinstance(use_layer_norm, bool)

        strides = [1, stride, stride, 1]
        in_dim = input.get_shape().as_list()[3]

        with tf.variable_scope(name):
            # optionally normalize the input

            W = self.init_weight_var(name=name, shape=[filter[0], filter[1], in_dim, num_outputs],
                                initializer=initializer)
            bias = self.init_bias_var(name=name, shape=[num_outputs],
                                 value=get_bias_value(activation_fn=activation_fn))

            input = tf.nn.conv2d(input=input, filter=W, strides=strides, padding='SAME', use_cudnn_on_gpu=True,
                                 data_format='NHWC', name='conv2d')


            if use_batch_norm and not use_layer_norm:
                input = batch_renorm(input=input)
            if use_layer_norm:
                input = layer_norm(layer=input)

            input = tf.nn.bias_add(input, bias, name=name+'_Wb')

            if activation_fn is not None:
                input = activation_fn(input)

            # apply dropout with dropout probability
            if dropout is not None:
                input = apply_dropout(x=input, keep_prob=self.keep_prob, name='drop', activation_fn=activation_fn)

            if collection is not None:
                assert isinstance(collection, list)
                collection.extend([W, bias])

        return input

    def fully_connected(self, input, num_outputs, name, sparse_connections=None, activation_fn=None, use_batch_norm=False,
                        use_layer_norm=False, initializer=None, dropout=None, collection=None):
        """Fully connected multiplication with additional processing steps. In detail:
         (optional) batchnorm -> (optional) activation) -> fully connected -> (optional) dropout

        :param input: tf tensor of any shape
        :param num_outputs: number of outputs
        :param name: name of the scope in which all operations are placed
        :param sparse_connections: Number of connections per output unit (int).
            If None, no sparse connections will be used and the layer is a simple fully connected layer.
        :param activation_fn: tf activation function to use. If none, no activation will be applied
        :param use_batch_norm: If True, a batch norm layer will be added. This can also be a tf tensor.
        :param initializer: initializer to use for variable initialization
        :param dropout: Specifying the keep probability of dropout layer. This can be a tensor or a float. If None, no
            dropout will be applied
        :param collection: If provided with a list, all variables are added to that list
        :param use_layer_norm: Alternative to batch normalization. Layer normalization behaves identical in train and inference
        :return: tensor with shape n_batch x n_output
        """
        # TODO: Support collection handling (also with batch_norm)
        with tf.variable_scope(name):
            input_shape = input.get_shape().as_list()
            shape_fcn = [np.prod(input_shape[1:]), num_outputs]

            W = self.init_weight_var(name=name, shape=shape_fcn, initializer=initializer)
            bias = self.init_bias_var(name=name, shape=[num_outputs],
                                 value=get_bias_value(activation_fn=activation_fn))

            # define fixed dropout mask for sparse connections?
            if sparse_connections is not None:
                assert isinstance(sparse_connections, int)
                sparse_mask = sparse_drop_mask(shape=shape_fcn, num_connections=sparse_connections)
                W = tf.multiply(W, sparse_mask, name='sparse_drop')


            input = tf.matmul(tf.reshape(tensor=input, shape=[-1, shape_fcn[0]]), W, name='fully')

            self.summary_train.append(tf.summary.histogram('pre_norm_activations', input))

            # optionally normalize the input
            if use_batch_norm and not use_layer_norm:
                input = batch_renorm(input=input)
            if use_layer_norm:
                input = layer_norm(layer=input)

            input = tf.nn.bias_add(input, bias, name='_Wb')

            if activation_fn is not None:
                input = activation_fn(input, name=name + '_post_acti')


            self.summary_train.append(tf.summary.histogram('post_activation', input))
            # apply dropout with dropout probability
            if dropout is not None:
                input = apply_dropout(x=input, keep_prob=dropout, name='drop', activation_fn=activation_fn)

            if collection is not None:
                assert isinstance(collection, list)
                collection.extend([W, bias])

        return input

    # def conv_acti_layer(self, bottom, filter_shape, filter_depth, name, stride, padding='SAME', layernorm=False):
    #     strides = [1, stride, stride, 1]
    #     with tf.variable_scope(name):
    #         pre_depth = bottom.get_shape()[3].value
    #         weights_shape = filter_shape + [pre_depth, filter_depth]
    #         weight = self._weight_variable(weights_shape, name=name)
    #         bias = self._bias_variable(filter_depth, name=name)
    #         conv = tf.nn.conv2d(bottom, weight, strides=strides, padding=padding)
    #
    #         if layernorm:
    #             norm = layernorm(conv)
    #             relu = tf.nn.relu(norm + bias)
    #         else:
    #             relu = tf.nn.relu(conv + bias)
    #
    #         return relu

    # def fc_acti_layer(self, bottom, weight_shape, bias_shape, name, activation_fn=tf.nn.relu, dropout=False, layernorm=False):
    #     with tf.variable_scope(name):
    #         W = self._weight_variable(weight_shape, name=name)
    #         b = self._bias_variable(bias_shape, name=name)
    #         preactivation = tf.matmul(bottom, W)
    #         tf.summary.histogram('pre_norm_activations', preactivation)
    #         if layernorm:
    #             norm = layer_norm(preactivation)
    #             acti = activation_fn(norm + b, name=name + '_relu')
    #         else:
    #             acti = activation_fn(preactivation + b, name=name + '_relu')
    #         tf.summary.histogram('activations', acti)
    #         if dropout:
    #             acti = tf.nn.dropout(acti, self._keep_prob, name=name + '_dropout')
    #         return acti

    def init_weight_var(self, shape, initializer=None, datatype=tf.float32, name=None):
        """  Returns a weight tensor with specified shape and datatype. An optional variable scope is wrapped around the
        operation.

        :param name: name of the variable
        :param shape: shape of the variable tensor
        :param initializer: optional initializer to initialize the variables
        :param datatype: datatype of the variable (or more precise tensor)
        :return: tf.variable with the specified shape and datatype
        """
        if initializer is None:
            initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=datatype)

        if name is None:
            name = '_weight'
        else:
            name += '_weight'
        var = tf.get_variable(name=name, shape=shape, dtype=datatype, initializer=initializer, regularizer=None,
                              trainable=True, collections=['variables'], caching_device=None, partitioner=None,
                              validate_shape=True, custom_getter=None)

        self.variable_summaries(var=var, name='weight')
        return var

    def init_bias_var(self, shape, value=0.0, datatype=tf.float32, name=None):
        """ Returns a bias tensor with specified shape and datatype. An optional variable scope is wrapped around the
        operation.

        :param name: name of the variable
        :param shape: shape of the variable tensor
        :param value: Scalar value with which the tensor will be initialized
        :param datatype: datatype of the variable (or more precise tensor)
        :return: tf.variable with the specified shape and datatype and constant fill value """
        initializer = tf.constant(value, shape=shape)
        if name is None:
            name = '_bias'
        else:
            name += '_bias'
        var = tf.get_variable(name=name, dtype=datatype, initializer=initializer, regularizer=None, trainable=True,
                              collections=['variables'], caching_device=None, partitioner=None, validate_shape=True,
                              custom_getter=None)
        self.variable_summaries(var=var, name='bias')

        return var


    def create_loss(self, output, target):
        """
        parameters can be batches
        :param output: output of network
        :param target:
        :return: loss as scalar
        """

        output_shape = output.get_shape().as_list()
        target_shape = target.get_shape().as_list()
        assert output_shape == target_shape, "output_shape.shape : %s, target_shape.shape: %s" %(output_shape, target_shape)

        self.score_diff = tf.abs(tf.subtract(target, output))
        self.variable_summaries(name='abs_target-output', var=self.score_diff)
        #tf.summary.scalar('abs_target-output', tf.reduce_mean(self.score_diff))
        self.rsquared_train = r_squared(targets=target, logits=output)
        #self.summary_train.append(tf.summary.scalar('r_squared_train', self.rsquared_train))


        #loss = -output * tf.log(target) - (1-output) * tf.log(1-target)
        loss = tf.square(tf.subtract(output, target))
        #loss = tf.abs(tf.subtract(output, target)))

        if len(output_shape) != 1: #1-d is batch dimension
            raise Exception("input parameters length os %s is unexpected" % output_shape)

        loss = tf.reduce_mean(loss)

        return loss

    def variable_summaries(self, var, name=''):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            self.summary_train.append(tf.summary.scalar('mean_'+ name, mean))
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            self.summary_train.append(tf.summary.scalar('stddev_' + name, stddev))
            self.summary_train.append(tf.summary.scalar('max_'+ name, tf.reduce_max(var)))
            self.summary_train.append(tf.summary.scalar('min_'+ name, tf.reduce_min(var)))
            self.summary_train.append(tf.summary.histogram('histogram_'+ name, var))

    def get_train_summary(self):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('train'):
            self.summary_train.append(tf.summary.scalar('output_mean', tf.reduce_mean(self.output)))
            self.summary_train.append(tf.summary.scalar('r_squared', self.rsquared_train))
            self.summary_train.append(tf.summary.scalar('loss', self.loss))
            self.summary_train.append(tf.summary.scalar('y_mean', self.y_mean[0]))
            self.summary_train.append(tf.summary.image("x", self.x))
            self.summary_train.append(tf.summary.histogram("histogram_gradient", self.inference_grad))
            self.summary_train.append(tf.summary.histogram("histogram_gradient", self.adverse_grad))
            return tf.summary.merge(self.summary_train)

    def get_test_summary(self):
        with tf.variable_scope('test'):
            self.map_test = tf.placeholder(dtype=tf.float32, name='map')
            self.rsquared_test = tf.placeholder(dtype=tf.float32, name='rquared')
            self.acc_test = tf.placeholder(dtype=tf.float32, name='acc')
            self.recall_test = tf.placeholder(dtype=tf.float32, name='recall')
            self.loss_test = tf.placeholder(dtype=tf.float32, name='loss')
            self.summary_test.append(tf.summary.scalar('map', self.map_test))
            self.summary_test.append(tf.summary.scalar('rsquared', self.rsquared_test))
            self.summary_test.append(tf.summary.scalar('acc', self.acc_test))
            self.summary_test.append(tf.summary.scalar('recall', self.recall_test))
            self.summary_test.append(tf.summary.scalar('loss', self.loss_test))
            return tf.summary.merge(self.summary_test)

    def build_network(self):

        self.loss = 0
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        with tf.variable_scope('input'):

            self.x = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, 3], name='x-input') # 3+k between 0 and 1
            self.y = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.num_classes], name='mask')
            self.target_score = tf.placeholder(tf.float32, shape=[None], name='target_score') # ground truth segmentation

            self.y_mean = tf.reduce_mean(self.y, [1, 2])[..., 1]

        with tf.variable_scope('input-concat'):
            self.concat = tf.concat([self.x, self.y], 3, name='xy-concat')

        with tf.variable_scope('conv1'):
            self.conv1 = self.conv(input=self.concat, num_outputs=64, name='conv1', filter=(5,5), stride=1, activation_fn=tf.nn.elu, use_layer_norm=True)
        with tf.variable_scope('conv2'):
            self.conv2 = self.conv(input=self.conv1, num_outputs=128, name='conv2', filter=(5,5), stride=2, activation_fn=tf.nn.elu, use_layer_norm=True)
        with tf.variable_scope('conv3'):
            self.conv3 = self.conv(input=self.conv2, num_outputs=128, name='conv3', filter=(5,5), stride=2, activation_fn=tf.nn.elu, use_layer_norm=True)

        with tf.variable_scope('fc1'):
            self.fc1 = self.fully_connected(input=self.conv3, num_outputs=384, name='fc1', activation_fn=tf.nn.elu, use_layer_norm=True)

        with tf.variable_scope('fc2'):
            self.fc2 = self.fully_connected(input=self.fc1, num_outputs=80, name='fc2', activation_fn=tf.nn.elu, use_layer_norm=True)

        with tf.variable_scope('fc3'):
            self.fc3 = self.fully_connected(input=self.fc2, num_outputs=1, name='fc3', activation_fn=tf.nn.sigmoid, use_layer_norm=False)
        self.output = tf.reshape(self.fc3, [-1])

        with tf.variable_scope('loss'):
            self.loss += self.create_loss(self.output, self.target_score)

        with tf.variable_scope('train'):
            self.adam = tf.train.AdamOptimizer(self.lr)
            self.optimizer = self.adam.minimize(self.loss, global_step=self.global_step)

        with tf.variable_scope('inference'):
            self.inference_grad = tf.gradients(self.output, self.y)

        with tf.variable_scope('adverse'):
            self.adverse_grad = tf.gradients(self.loss, self.y)

        logging.info("#variables %s" % count_variables())

            # end_loss, self._graph['sim_score'] = self._create_loss(self._graph['fc3'], self._graph['y'], self._graph['y_gt'])
            # self._graph['loss'] += end_loss
            # optimizer = tf.train.AdamOptimizer(self._learning_rate)
            # #optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
            # self._graph['train_gradients'] = optimizer.compute_gradients(self._graph['loss'])
            # self._graph['train_optimizer'] = optimizer.apply_gradients(self._graph['train_gradients'], global_step=self._graph['global_step']) # protect y1:0 from being updated

        self.summary_test = self.get_test_summary()
        self.summary_train = self.get_train_summary()
