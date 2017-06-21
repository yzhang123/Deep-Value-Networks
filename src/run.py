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

from understand_tensorflow.src.deeplearning import save_model
#from dkfz.train import result_sample_mapping
from model.dvn import DvnNet
from dvn.src.data.data_set import DataSet
from dvn.src.data.generate_data import DataGenerator
from dvn.src.util.data import pred_to_label, label_to_colorimg, blackMask
from dvn.src.util.io import write_image
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
            print(iter)

            feed_dict = {graph['x']: img, graph['y_gt']: img_gt, graph['y']: mask}
            _, loss = sess.run([graph['train_optimizer'], graph['loss']], feed_dict=feed_dict)

            feed_dict = {graph['x']: img, graph['y']: mask}
            identity, inference_update, inference_grad = sess.run([graph['identity'], graph['inference_update'],
                                                                   graph['inference_grad']], feed_dict=feed_dict)

            print('inference_grad')
            print(inference_grad[0])
            print('inference_update')
            print(inference_update)
            print("iteration %s: loss = %s" % (iter, loss[0]))

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
            labels = pred_to_label(inference_update)
            mapp_pred = result_sample_mapping(img_gt, inference_update)
            write_image(mapp_pred, iter, '')
            iter += 1
            if iter >= ITERS_TRAIN:
                break

def result_sample_mapping(gt_labels, pred_labels):
    """

    :param gt_labels:
    :param pred_labels:
    :return:
    """

    mapped_pred = np.zeros((gt_labels.shape[0], gt_labels.shape[1], gt_labels.shape[2], 3))

    # print("Pred Label _ Shape: ", pred_labels.shape, "gt: ", gt_labels.shape)

    gt_labels = np.argmax(gt_labels, axis=-1)

    pred_labels = np.argmax(pred_labels, axis=-1)

    # print("Pred Label _ Shape: ", pred_labels.shape, "gt: ", gt_labels.shape)

    true_positives = np.logical_and((pred_labels == gt_labels), (gt_labels == 1))

    false_positives = np.logical_and((gt_labels == 0), (pred_labels == 1))

    false_negatives = np.logical_and((gt_labels == 1), (pred_labels == 0))

    mapped_pred[true_positives] = np.array([0, 1, 0])  # green
    mapped_pred[false_positives] = np.array([1, 0, 0])  # red
    mapped_pred[false_negatives] = np.array([0, 0, 1])  # blue

    return mapped_pred



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    return args

if __name__== "__main__":

    args = parse_args()

    img_path = join(dir_path, "../", "data/weizmann_horse_db/rgb")
    test_img_path = join(dir_path, "../", "data/weizmann_horse_db/gray")
    img_gt_path = join(dir_path, "../", "data/weizmann_horse_db/figure_ground")
    print("img_dir %s" % img_path)
    print("img_gt_dir %s" % img_gt_path)

    classes = ['__background__', 'horse']

    net = DvnNet(classes=classes, batch_size = BATCH_SIZE)


    if args.train:
        graph = net.build_network(train=True)
        train_data = DataSet(classes, img_path, img_gt_path, batch_size=BATCH_SIZE)
        train(graph, train_data)
    else:
        graph = net.build_network(train=False)
        test_data = DataSet(classes, test_img_path, img_gt_path, batch_size=BATCH_SIZE)
        modelpath = '/home/yang/projects/dvn/checkpoints/model-50'
        test(graph, modelpath, test_data)


