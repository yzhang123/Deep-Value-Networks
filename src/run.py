# -*- coding: utf-8 -*-
# Compatibility to python 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import tensorflow as tf
import numpy as np
import os
import sys
from os.path import join
import argparse
from scipy.misc import imsave
import logging
import pprint

from understand_tensorflow.src.deeplearning import save_model
#from dkfz.train import result_sample_mapping
from dvn.src.model.dvn import DvnNet
from dvn.src.data.data_set import DataSet
from dvn.src.data.generate_data import DataGenerator
from dvn.src.util.data import pred_to_label, label_to_colorimg, blackMask, result_sample_mapping
from dvn.src.util.input_output import write_image
from dvn.src.util.model import inference as infer
from dvn.src.util.measures import calc_accuracy, calc_recall, oracle_score




module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)  # store dir_path for later use
root_path = join(dir_path, "../")
log_dir = join(root_path, "logs")
model_dir = join(root_path, 'checkpoints/')
tensorboard_dir = join(root_path, "tensorboard/")

# Number of training iterations
ITERS_TRAIN = 3000
# Number of iterations after a snapshot of the model is saved
ITERS_PER_SAVE = 100
# absolute path where model snapshots are saved
# number of batch size of incoming data



def train(net, data, data_update_rate):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        iter = initial_step = net.global_step.eval()

        train_writer = tf.summary.FileWriter(tensorboard_dir + '/train', sess.graph)
        generator = DataGenerator(sess, net, data, train=True, data_update_rate=data_update_rate)
        # x, _, gt = next(generator.gt())
        # logging.info("image: %s" % x)
        # logging.info("gt: %s" % gt)
        # np.save('arrays/x.npy', x)
        # np.save('arrays/y.npy', gt)

        #for img, mask, img_gt in generator.generate():
        for imgs, input_masks, img_masks in generator.generate_batch():
            target_scores = oracle_score(input_masks, img_masks)
            feed_dict = {net.x : imgs, net.y: input_masks, net.target_score: target_scores}

            summary, _, lr, loss, outputs, score_diffs, y_mean = sess.run([net.merged_summary, net.optimizer, net.adam._lr_t, net.loss, net.output, net.score_diff, net.y_mean], feed_dict=feed_dict)
            logging.info("iter %s: lr=%s, \noutputs = %s, \ntargets=%s, \nscore_diff=%s, \nloss = %s," %(iter, lr, outputs, target_scores, score_diffs, loss))

            train_writer.add_summary(summary, iter)
            np.set_printoptions(formatter={'all': lambda x:  str(x) + '\n'})

            iter += 1
            if iter % ITERS_PER_SAVE == 0:
                save_model(sess, model_dir, 'model', global_step=iter)
            if iter >= initial_step + ITERS_TRAIN:
                break

        train_writer.close()


def test(net, data, data_update_rate=10, model_dir=model_dir):
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        #saver = tf.train.import_meta_graph(modelpath + '.meta')
        #saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(modelpath)))
        saver = tf.train.Saver()
        saver.restore(sess, model_dir)

        writer = tf.summary.FileWriter(tensorboard_dir + '/test', sess.graph)
        generator = DataGenerator(sess, net, data, train=False, data_update_rate=data_update_rate)
        iter = 0
        for imgs, input_masks, img_masks in generator.generate_batch():
            target_scores = oracle_score(input_masks, img_masks)
            feed_dict = {net.x : imgs, net.y: img_masks, net.target_score: target_scores}
            summary, loss, outputs, score_diffs, y_mean = sess.run(
                [net.merged_summary, net.loss, net.output, net.score_diff, net.y_mean], feed_dict=feed_dict)

            # acc = calc_accuracy(img_gt, pred)
            # logging.info("it i = %s, acc = %s" %(iter, acc))
            # recall = calc_recall(img_gt, pred)
            # logging.info("it i = %s, recall = %s" %(iter, recall))
            # logging.debug("img %s" % img)
            # labels = pred_to_label(input_masks)
            # mapp_pred = result_sample_mapping(img_gt, pred)
            # write_image(mapp_pred, -1, data.index_list[iter])

            logging.info("iter %s: \noutputs = %s, \ntargets=%s, \nscore_diff=%s, \nloss = %s," %(iter, outputs, target_scores, score_diffs, loss))
            iter += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--loglevel', default='info')
    args = parser.parse_args()
    return args



if __name__== "__main__":

    args = parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    logging.basicConfig(filename=dir_path + '/log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=numeric_level)
    #logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=numeric_level)

    img_path = join(root_path, "data/weizmann_horse_db/rgb")
    test_img_path = join(root_path, "data/weizmann_horse_db/rgb")
    img_gt_path = join(root_path, "data/weizmann_horse_db/figure_ground")
    logging.info("imgs_dir %s" % img_path)
    logging.info("masks_gt_dir %s" % img_gt_path)

    classes = ['__background__', 'horse']
    BATCH_SIZE = 10
    SIZE = (48, 48)

    net_params = {
        'input_height': SIZE[0],
        'input_width': SIZE[1],
        'num_classes': 2,
        'learning_rate': 0.00001
    }

    net = DvnNet(**net_params)

    data_update_rate = 10

    if args.train:
        net.build_network(train=True)
        train_data = DataSet(classes=classes, img_dir=img_path, gt_dir=img_gt_path, batch_size=BATCH_SIZE, size=SIZE, train=True, repeat=True, shuffle=True)
        train(net=net, data=train_data, data_update_rate = data_update_rate)
    else:
        net.build_network(train=False)
        test_data = DataSet(classes=classes, img_dir=test_img_path, gt_dir=img_gt_path, batch_size=1, size=SIZE, train=False, repeat=False, shuffle=False)
        modelpath = '/home/yang/projects/dvn/checkpoints/model-500'
        test(net=net, model_dir=modelpath, data=test_data, data_update_rate=data_update_rate)


