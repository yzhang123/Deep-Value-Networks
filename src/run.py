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
import os
import sys
from os.path import join
import argparse
from scipy.misc import imsave
import logging

from understand_tensorflow.src.deeplearning import save_model
#from dkfz.train import result_sample_mapping
from model.dvn import DvnNet
from dvn.src.data.data_set import DataSet
from dvn.src.data.generate_data import DataGenerator
from dvn.src.util.data import pred_to_label, label_to_colorimg, blackMask, result_sample_mapping
from dvn.src.util.input_output import write_image
from dvn.src.util.model import inference as infer




module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)  # store dir_path for later use
root_path = join(dir_path, "../")


# Number of training iterations
ITERS_TRAIN = 60
# Number of inference iterations
ITERS_TEST = 30
# Number of iterations after a snapshot of the model is saved
ITERS_PER_SAVE = 50
# absolute path where model snapshots are saved
SAVE_PATH = join(root_path, 'checkpoints/')
# number of batch size of incoming data
BATCH_SIZE = 1



def train(graph, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        iter = initial_step = graph['global_step'].eval()

        generator = DataGenerator(sess, graph, data)
        for img, mask, img_gt in generator.generate():
            # print(iter)

            feed_dict = {graph['x']: img, graph['y_gt']: img_gt, graph['y']: mask}
            _, loss = sess.run([graph['train_optimizer'], graph['loss']], feed_dict=feed_dict)

            # feed_dict = {graph['x']: img, graph['y']: mask}
            # identity, inference_update, inference_grad = sess.run([graph['identity'], graph['inference_update'],
            #                                                        graph['inference_grad']], feed_dict=feed_dict)
            #
            # print('inference_grad')
            # print(inference_grad[0])
            # print('inference_update')
            # print(inference_update)
            logging.info("iteration %s: loss = %s" % (iter, loss[0]))

            iter += 1
            #save model
            if iter % ITERS_PER_SAVE == 0:
                save_model(sess, SAVE_PATH, 'model', global_step=iter)
            if iter >= initial_step + ITERS_TRAIN:
                break


def test(graph, modelpath, data):
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        saver = tf.train.import_meta_graph(modelpath + '.meta')
        #saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(modelpath)))
        saver.restore(sess, modelpath)

        generator = DataGenerator(sess, graph, data)
        iter = 0
        for img, mask, img_gt in generator.black():
            inference_update = infer(sess, graph, img, mask)
            #labels = pred_to_label(inference_update)
            mapp_pred = result_sample_mapping(img_gt, inference_update)
            write_image(mapp_pred, -1, data.index_list[iter])
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
    #logging.basicConfig(filename='log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=numeric_level)
    logging.basicConfig(filename='log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=numeric_level)

    img_path = join(dir_path, "../", "data/weizmann_horse_db/rgb")
    test_img_path = join(dir_path, "../", "data/weizmann_horse_db/gray")
    img_gt_path = join(dir_path, "../", "data/weizmann_horse_db/figure_ground")
    logging.info("img_dir %s" % img_path)
    logging.info("img_gt_dir %s" % img_gt_path)

    classes = ['__background__', 'horse']

    net = DvnNet(classes=classes, batch_size = BATCH_SIZE)


    if args.train:
        graph = net.build_network(train=True)
        train_data = DataSet(classes, img_path, img_gt_path, batch_size=BATCH_SIZE, train=True)
        train(graph, train_data)
    else:
        graph = net.build_network(train=False)
        test_data = DataSet(classes, test_img_path, img_gt_path, batch_size=BATCH_SIZE, train=False)
        modelpath = '/home/yang/projects/dvn/checkpoints/model-50'
        test(graph, modelpath, test_data)


