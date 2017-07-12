

# from tensorflow.contrib.layers import layer_norm
# from dvn.src.util.measures import _oracle_score_cpu

import  tensorflow as tf
from nn_toolbox.src.tf.blocks.basic_blocks import fully_connected
from dvn.src.data.data_set import DataSet
from dvn.src.data.generate_data import DataGenerator
from dvn.src.util.data import randomMask, blackMask, sampleExponential, zeroMask, oneMask
from dvn.src.util.data import left_upper1_4_mask, left_upper2_4_mask, left_upper3_4_mask, left_upper2_2_mask, meanMask
import scipy.misc


import numpy as np
from os.path import join, dirname, abspath

class SimpleDVN:

    def __init__(self, learning_rate, input_height, input_width, num_classes):
        self.lr = learning_rate
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_height, 3], name='x-input')
        self.y = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.num_classes], name='y-input')
        self.target_score = tf.placeholder(tf.float32, shape=[None], name='target_score')

        self.concat = tf.concat([self.x, self.y], axis=3, name='x-y-concat')
        self.fc1 = fully_connected(input=self.concat, num_outputs=20, name='fc1')
        self.fc2 = fully_connected(input=self.fc1, num_outputs=1, name='fc2', activation_fn=tf.nn.relu)
        self.output = tf.reshape(tf.nn.sigmoid(self.fc2), [-1])
        self.loss = self.create_loss(self.output, self.target_score)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


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
    img_path = "/home/yang/data/weizmann_horse_db/rgb_1"
    # test_img_path = "/home/yang/data/weizmann_horse_db/rgb_1"
    img_gt_path = "/home/yang/data/weizmann_horse_db/figure_ground_1"

    output_img_folder = "/home/yang/projects/dvn/src/simple_dvn/output/img"

    print("img_dir %s" % img_path)
    print("img_gt_dir %s" % img_gt_path)
    print("output_img_folder %s" % output_img_folder)

    classes = ['__background__', 'horse']

    BATCH_SIZE = 1
    SIZE = (48, 48)
    LR = 0.001
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
            print("iter %s: loss = %s, output = %s" %(iter, loss, output))

            iter += 1

