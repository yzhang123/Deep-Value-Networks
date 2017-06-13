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
import _init_paths
print(sys.path)
from understand_tensorflow.src.deeplearning import save_model

from scipy.misc import imsave

module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)  # store dir_path for later use
root_path = join(dir_path, "../")


from model.dvn import DvnNet
from data.data_set import DataSet

ITERS = 60
ITERS_PER_SAVE = 50
SAVE_PATH = join(root_path, 'checkpoints/')
BATCH_SIZE = 1

def train(graph, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        iter = initial_step = graph['global_step'].eval()

        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        for img, img_gt in data:
            iter += 1
            assert img.shape[0:1] == img_gt.shape[0:1]

            feed_dict = {graph['x']: img, graph['y_gt']: img_gt, graph['y']: img_gt}

            #sess.run(graph['y'].assign(img_gt))
            out, x, y = sess.run([graph['y_clipped'], graph['fc3'], graph['loss']], feed_dict=feed_dict)
            _, gradients, loss, gt_diff = sess.run([graph['train_optimizer'], graph['train_gradients'], graph['loss'], graph['sim_score']],
                                        feed_dict=feed_dict)

            print(out)

            print("iteration %s: loss = %s, sim_score = %s" % (iter, loss[0], gt_diff[0]))

            #save model?
            if iter % ITERS_PER_SAVE == 0:
                save_model(sess, SAVE_PATH, 'model', global_step=iter)
            if iter >= initial_step + ITERS:
                break


def test(graph, modelpath, data):
    ITERS = 50
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        saver = tf.train.import_meta_graph(modelpath)
        #saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(modelpath)))
        saver.restore(sess, '/home/yang/projects/dvn/checkpoints/model-50')

        black_batch = np.zeros([1, data.size[0], data.size[1], data.num_classes], dtype=np.float32)
        black_batch[:, :, :, 0] = 1.

        iter = 0
        for img, img_gt in data:
            feed_dict = {graph['x']: img, graph['y']: black_batch}
            score = sess.run(graph['inference_optimizer'], feed_dict=feed_dict)
            for i in range(ITERS):
                feed_dict = {graph['x']: img}
                score = sess.run(graph['inference_optimizer'], feed_dict=feed_dict)
            seg_pred =  graph['y_clipped'].eval()
            print(seg_pred)
            #labels = pred_to_label(seg_pred)
            #for x in labels:
            #    print(x)
            #mapp_pred = result_sample_mapping(img_gt, graph['y_clipped'].eval())
            #write_image(mapp_pred, iter, '')
            iter += 1


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


def pred_to_label(seg_masks):
    pred_labels = np.argmax(seg_masks, -1)
    return pred_labels

def label_to_colorimg(pred_labels, classes, color_map):
    imgs = np.zeros((pred_labels.shape[0], pred_labels.shape[1], pred_labels.shape[2], 3))
    for idx, c in enumerate(classes):
        class_mask = pred_labels == idx
        imgs[class_mask] = color_map[c]


def save_images(X, save_path, transform=False):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99 * X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples // rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        if transform:
            X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x

    imsave(save_path, img)

def write_image(images, iteration, name, mapping=True):
    """

    :param images:
    :param iteration:
    :param name:
    :param mapping:
    :return:
    """
    if mapping:
        images = ((images + 1.) * (255.99 / 2)).astype('int32')
    else:
        images = images.astype('int32')
    save_images(images, root_path + '/output/' + name + '{}.png'.format(iteration), False)


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
    graph = net.build_network()

    if args.train:
        train_data = DataSet(classes, img_path, img_gt_path, batch_size=BATCH_SIZE)
        train(graph, train_data)
    else:
        test_data = DataSet(classes, test_img_path, img_gt_path, batch_size=BATCH_SIZE)

        modelpath = '/home/yang/projects/dvn/checkpoints/model-50.meta'
        test(graph, modelpath, test_data)


