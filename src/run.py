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
from dvn.src.util.loss import calc_accuracy, calc_recall




module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)  # store dir_path for later use
root_path = join(dir_path, "../")
log_dir = join(root_path, "logs")
model_dir = join(root_path, "model")

# Number of training iterations
ITERS_TRAIN = 3000
# Number of iterations after a snapshot of the model is saved
ITERS_PER_SAVE = 100
# absolute path where model snapshots are saved
SAVE_PATH = join(root_path, 'checkpoints/')
# number of batch size of incoming data
BATCH_SIZE = 10



def train(graph, data, data_update_rate = 0.5):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        iter = initial_step = graph['global_step'].eval()

        train_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
        generator = DataGenerator(sess, graph, data, train=True, data_update_rate=data_update_rate)
        # x, _, gt = next(generator.gt())
        # logging.info("image: %s" % x)
        # logging.info("gt: %s" % gt)
        # np.save('arrays/x.npy', x)
        # np.save('arrays/y.npy', gt)

        #for img, mask, img_gt in generator.generate():
        for img, mask, img_gt in generator.helper():
            # print(iter)

            feed_dict = {graph['x']: img, graph['y_gt']: img_gt, graph['y']: mask}
            _, y_mean, score_diff, loss, sim_score, fc3, gradient, sim_score_vector, summary = sess.run([graph['train_optimizer'], graph['y_mean'], graph['score_diff'], graph['loss'], graph['sim_score_vector'], graph['fc3'], graph['inference_grad'], graph['sim_score_vector'], graph['merged_summary']], feed_dict=feed_dict)
            #loss, sim_score, fc3, gradient, summary = sess.run([graph['loss'], graph['sim_score'], graph['fc3'], graph['inference_grad'], graph['merged_summary']], feed_dict=feed_dict)

            train_writer.add_summary(summary, iter)
            # feed_dict = {graph['x']: img, graph['y']: mask}
            # identity, inference_update, inference_grad = sess.run([graph['identity'], graph['inference_update'],
            #                                                        graph['inference_grad']], feed_dict=feed_dict)
            #
            # print('inference_grad')
            # print(inference_grad[0])
            # print('inference_update')
            # print(inference_update)
            # logging.debug("mask : %s" % mask)
            #
            # np.save('arrays/y-%s.npy' % iter, mask)
            logging.info("iteration %s: loss = %s, sim_score = %s, fc3 = %s, sim_score - net_output=%s, " % (iter, loss, sim_score, fc3, sim_score-fc3.flatten()))
            # logging.info("sim_score vector ")
            # logging.info(pprint.pformat(sim_score_vector))
            logging.info("y_mean %s" % y_mean)
            iter += 1
            #save model
            if iter % ITERS_PER_SAVE == 0:
                save_model(sess, SAVE_PATH, 'model', global_step=iter)
            if iter >= initial_step + ITERS_TRAIN:
                break
        train_writer.close()

def test(graph, modelpath, data, data_update_rate=0.5):
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        #saver = tf.train.import_meta_graph(modelpath + '.meta')
        #saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(modelpath)))
        saver = tf.train.Saver()
        saver.restore(sess, modelpath)

        generator = DataGenerator(sess, graph, data, train=False, data_update_rate=data_update_rate)
        iter = 0
        for img, pred, img_gt in generator.generate():
        #for img, pred, img_gt in generator.helper():
            # logging.debug("infererred image %s" % pred)
            # acc = calc_accuracy(img_gt, pred)
            # logging.info("it i = %s, acc = %s" %(iter, acc))
            # recall = calc_recall(img_gt, pred)
            # logging.info("it i = %s, recall = %s" %(iter, recall))
            labels = pred_to_label(pred)
            mapp_pred = result_sample_mapping(img_gt, pred)
            write_image(mapp_pred, -1, data.index_list[iter])
            feed_dict = {graph['x']: img, graph['y_gt']: img_gt, graph['y']: pred}
            sim_score, fc3 = sess.run([graph['sim_score'], graph['fc3']], feed_dict=feed_dict)

            print("iter %s: simscore = %s, fc3 = %s, diff sim_score vs fc3 = %s" %(iter, sim_score, fc3, sim_score - fc3))
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

    img_path = join(dir_path, "../", "data/weizmann_horse_db/rgb_1")
    test_img_path = join(dir_path, "../", "data/weizmann_horse_db/rgb_1")
    img_gt_path = join(dir_path, "../", "data/weizmann_horse_db/figure_ground_1")
    logging.info("img_dir %s" % img_path)
    logging.info("img_gt_dir %s" % img_gt_path)

    classes = ['__background__', 'horse']

    net_params = {
        'classes': classes,
        'batch_size': BATCH_SIZE,
        'lr': 0.01
    }
    net = DvnNet(**net_params)
    #net = DvnNet(classes=classes, batch_size=BATCH_SIZE, lr=0.0001)

    data_update_rate = 10

    if args.train:
        graph = net.build_network(train=True)
        train_data = DataSet(classes, img_path, img_gt_path, batch_size=BATCH_SIZE, train=True)
        train(graph, train_data, data_update_rate = data_update_rate)
    else:
        graph = net.build_network(train=False)
        test_data = DataSet(classes, test_img_path, img_gt_path, batch_size=1, train=False) #train=False
        modelpath = '/home/yang/projects/dvn/checkpoints/model-500'
        test(graph, modelpath, test_data, data_update_rate=data_update_rate)


