

# from tensorflow.contrib.layers import layer_norm
# from dvn.src.util.measures import _oracle_score_cpu

import  tensorflow as tf
from nn_toolbox.src.tf.blocks.basic_blocks import fully_connected, conv
from dvn.src.data.data_set import DataSet
from dvn.src.data.generate_data import DataGenerator
from dvn.src.util.data import randomMask, blackMask, sampleExponential, zeroMask, oneMask
from dvn.src.util.data import left_upper1_4_mask, left_upper2_4_mask, left_upper3_4_mask, left_upper2_2_mask, meanMask
import scipy.misc

from nn_toolbox.src.tf.tf_extend.tf_helpers import count_variables


import numpy as np
from os.path import join, dirname, abspath

class SimpleDVN:

    def __init__(self, learning_rate, input_height, input_width, num_classes):
        self.lr = learning_rate
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        self.build_CONV()

    def build_FC(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_height, 3], name='x-input')
        self.y = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.num_classes], name='y-input')
        self.target_score = tf.placeholder(tf.float32, shape=[None], name='target_score')

        self.concat = tf.concat([self.x, self.y], axis=3, name='x-y-concat')
        self.fc1 = fully_connected(input=self.concat, num_outputs=200, name='fc1')
        self.fc2 = fully_connected(input=self.fc1, num_outputs=1, name='fc2', activation_fn=tf.nn.relu)
        self.output = tf.reshape(tf.nn.sigmoid(self.fc2), [-1])
        self.loss = self.create_loss(self.output, self.target_score)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        print("#variables %s" % count_variables())

    def build_CONV(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_height, 3], name='x-input')
        self.y = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.num_classes], name='y-input')
        self.target_score = tf.placeholder(tf.float32, shape=[None], name='target_score')

        self.concat = tf.concat([self.x, self.y], axis=3, name='x-y-concat')
        self.conv1 = conv(input=self.concat, channel=24, name='conv1', filter=(7, 7), stride=1, activation_fn=None)
        self.conv2 = conv(input=self.conv1, channel=128, name='conv2', filter=(5, 5), stride=2, activation_fn=tf.nn.relu, use_layer_norm=True)
        self.conv3 = conv(input=self.conv2, channel=128, name='conv3', filter=(5, 5), stride=2, activation_fn=tf.nn.relu, use_layer_norm=True)
        self.fc1 = fully_connected(input=self.conv3, num_outputs=384, name='fc1', activation_fn=tf.nn.relu, use_layer_norm=True)
        self.fc2 = fully_connected(input=self.fc1, num_outputs=80, name='fc2', activation_fn=tf.nn.relu, use_layer_norm=True)
        self.fc3 = fully_connected(input=self.fc2, num_outputs=1, name='fc3', activation_fn=tf.nn.relu, use_layer_norm=True)
        self.output = tf.reshape(tf.nn.sigmoid(self.fc3), [-1])
        self.loss = self.create_loss(self.output, self.target_score)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        print("#variables %s" % count_variables())

    # def fc_acti_layer(self, bottom, weight_shape, bias_shape, name, activation_fn=tf.nn.relu, dropout=False,
    #                   layernorm=None):
    #     with tf.variable_scope(name):
    #         W = self._weight_variable(weight_shape, name=name)
    #         b = self._bias_variable(bias_shape, name=name)
    #         preactivation = tf.matmul(bottom, W)
    #         tf.summary.histogram('pre_norm_activations', preactivation)
    #         if layernorm:
    #             norm = layernorm(preactivation)
    #             acti = activation_fn(norm + b, name=name + '_relu')
    #         else:
    #             acti = activation_fn(preactivation + b, name=name + '_relu')
    #         tf.summary.histogram('activations', acti)
    #         return acti
    #
    # def _weight_variable(self, shape, initializer=None, name=None):
    #     if not initializer:
    #         initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    #     var = tf.get_variable(name + '_weight', shape, initializer=initializer, collections=['variables'])
    #     self.variable_summaries(var, 'weight')
    #     # self._graph['loss'] += self.regularizer(var) * self.weight_decay
    #     return var
    #
    # def _bias_variable(self, shape, initializer=None, name=None):
    #     if not initializer:
    #         initializer = tf.constant_initializer(0.01)
    #     var = tf.get_variable(name + '_bias', shape, initializer=initializer, collections=['variables'])
    #     self.variable_summaries(var, name='bias')
    #     return var
    #
    # def variable_summaries(self, var, name=''):
    #     """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    #     """
    #     with tf.name_scope('summaries'):
    #         mean = tf.reduce_mean(var)
    #         tf.summary.scalar('mean_'+ name, mean)
    #         with tf.name_scope('stddev'):
    #             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #         tf.summary.scalar('stddev_' + name, stddev)
    #         tf.summary.scalar('max_'+ name, tf.reduce_max(var))
    #         tf.summary.scalar('min_'+ name, tf.reduce_min(var))
    #         tf.summary.histogram('histogram_'+ name, var)



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

        loss = tf.square(tf.subtract(output, target))

        if len(output_shape) == 1:
            return loss
        if len(output_shape) == 2: # assumed first dimension is batch dimension
            loss = tf.reduce_mean(loss)
            return loss
        raise Exception("input parameters do not have expected shape")

class DataGenerator:

    def __init__(self, data, batch_size, img_height, img_width, num_classes):
        self.data = data
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes

    def gt(self):
        for img, img_mask in self.data:
            yield img, img_mask

    def generate_batch(self):
        """

        :return: images, input masks, and according ground truth masks as batches
        """
        shape = (self.img_height, self.img_width, self.num_classes)
        masks = list()
        masks.append(left_upper2_2_mask(shape))
        masks.append(left_upper1_4_mask(shape))
        masks.append(blackMask(shape))
        masks.append(left_upper2_4_mask(shape))
        masks.append(left_upper3_4_mask(shape))
        masks.append(zeroMask(shape))

        def _get_mask(img_mask):
            rand_idx = np.random.randint(0, len(masks) + 1)
            print("mask %s" %rand_idx)
            if rand_idx < len(masks):
                return masks[rand_idx]
            else:
                return img_mask

        while(True):
            imgs, img_masks = next(self.gt())
            input_masks = list()
            for i in range(img_masks.shape[0]):
                mask = _get_mask(img_masks[i])
                input_masks.append(mask)

            input_masks = np.stack(input_masks, axis=0)

            assert imgs.shape > input_masks.shape, "imgs.shape : %s, input_masks.shape : %s" % (
            imgs.shape, input_masks.shape)
            assert img_masks.shape == input_masks.shape

            yield imgs, input_masks, img_masks

def oracle_score(masks_a, masks_b):
    """

    :param masks_a: batch
    :param masks_b: batch
    :return: batches of relaxed iou scores, 1-d tensor
    """
    assert masks_a.shape == masks_b.shape
    assert len(masks_a.shape) == 4
    y_min = np.sum(np.sum(np.minimum(masks_a, masks_b), 2), 1)
    y_max = np.sum(np.sum(np.maximum(masks_a, masks_b), 2), 1)
    y_divide = np.divide(y_min, y_max)
    y_divide[y_max == 0.] = 1.

    scores = np.mean(y_divide, 1)
    assert scores.shape[0] == masks_a.shape[0], "target_score.shape: %s, masks.shape: %s" %(scores.shape, masks_a.shape)
    return scores



if __name__=='__main__':
    img_path = "/home/yang/data/weizmann_horse_db/rgb"
    # test_img_path = "/home/yang/data/weizmann_horse_db/rgb_1"
    img_gt_path = "/home/yang/data/weizmann_horse_db/figure_ground"

    output_img_folder = "/home/yang/projects/dvn/src/simple_dvn/output/img"

    print("img_dir %s" % img_path)
    print("img_gt_dir %s" % img_gt_path)
    print("output_img_folder %s" % output_img_folder)

    classes = ['__background__', 'horse']

    BATCH_SIZE = 10
    SIZE = (48, 48)
    LR = 0.00001
    NUM_CLASSES = len(classes)
    DATA_REPEAT = True
    DATA_SHUFFLE = True

    data = DataSet(classes, img_path, img_gt_path, size=SIZE, batch_size=BATCH_SIZE, repeat=DATA_REPEAT, shuffle=DATA_SHUFFLE)
    data_generator = DataGenerator(data=data, batch_size=BATCH_SIZE, img_height=SIZE[0], img_width=SIZE[1], num_classes=NUM_CLASSES)

    net = SimpleDVN(learning_rate=LR, input_height=SIZE[0], input_width=SIZE[1], num_classes=NUM_CLASSES)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        iter = 0
        for imgs, input_masks, img_masks in data_generator.generate_batch():
            target_scores = oracle_score(input_masks, img_masks)
            feed_dict = {net.x : imgs, net.y: input_masks, net.target_score: target_scores}

            _, loss, output = sess.run([net.optimizer, net.loss, net.output], feed_dict=feed_dict)
            print("iter %s: \noutput = %s, \ntarget=%s, \nloss = %s," %(iter, output, target_scores, loss))

            iter += 1


