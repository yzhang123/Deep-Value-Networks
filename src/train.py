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

from understand_tensorflow.src.deeplearning import save_model, load_model
#from dkfz.train import result_sample_mapping
from dvn.src.model.dvn import DvnNet
from dvn.src.data.data_set import DataSet
from dvn.src.data.generate_data import DataGenerator
from dvn.src.util.data import pred_to_label, label_to_colorimg, blackMask, result_sample_mapping, binarize_image
from dvn.src.util.input_output import write_image
from dvn.src.util.model import inference as infer
from dvn.src.util.measures import calc_accuracy, calc_recall, oracle_score
from nn_toolbox.src.tf.tf_extend.metrics import R_squared




module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)  # store dir_path for later use
root_path = join(dir_path, "../")
# log_dir = join(root_path, "logs")
# model_dir = join(root_path, 'checkpoints/')
# tensorboard_dir = join(root_path, "tensorboard/")

# Number of training iterations
ITERS_TRAIN = 3000
# Number of iterations after a snapshot of the model is saved
ITERS_PER_SAVE = 100
# absolute path where model snapshots are saved
# number of batch size of incoming data



def main(net, train_data, test_data, data_update_rate, model_dir, tensorboard_dir):
    """

    :param net:
    :param train_data: iterator
    :param test_data: iterator
    :param data_update_rate:
    :param model_dir:
    :param tensorboard_dir:
    :return:
    """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('loading model %s' % ckpt.model_checkpoint_path)
            load_model(session=sess, path_model=ckpt.model_checkpoint_path)

        train_writer = tf.summary.FileWriter(tensorboard_dir + '/train', sess.graph)
        generator_train = DataGenerator(session=sess, net=net, data=train_data, mode='train', data_update_rate=data_update_rate)
        generator_test = DataGenerator(session=sess, net=net, data=test_data, mode='test', data_update_rate=data_update_rate)
        # x, _, gt = next(generator_train.gt())
        # logging.info("image: %s" % x)
        # logging.info("gt: %s" % gt)
        # np.save('arrays/x.npy', x)
        # np.save('arrays/y.npy', gt)

        #np.set_printoptions(formatter={'all': lambda x: str(x) + '\n'})

        iter = initial_step = net.global_step.eval()
        for imgs, input_masks, img_masks in generator_train.generate():
            #binarize input mask
            bin_mask = binarize_image(input_masks)
            #target_scores = oracle_score(input_masks, img_masks)
            target_scores = oracle_score(bin_mask, img_masks)
            feed_dict = {net.x : imgs, net.y: input_masks, net.target_score: target_scores}

            summary, _, lr, loss, outputs, score_diffs, y_mean = sess.run([net.summary_train, net.optimizer, net.adam._lr_t, net.loss, net.output, net.score_diff, net.y_mean], feed_dict=feed_dict)
            logging.info("iter %s: lr=%s, \noutputs = %s, \ntargets=%s, \nscore_diff=%s, \nloss = %s," %(iter, lr, outputs, target_scores, score_diffs, loss))

            train_writer.add_summary(summary, iter)

            iter += 1
            if iter % ITERS_PER_SAVE == 0:
                logging.info('saving %s/model-%s' % (model_dir, iter))
                save_model(sess, model_dir, 'model', global_step=iter)

            if iter >= initial_step + ITERS_TRAIN:
                break

        train_writer.close()


data_dir = join(root_path, "data/weizmann_horse_db")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='data folder', default=data_dir, required=False)
    parser.add_argument('--loglevel', help='logging model, currently supports [info, debug]', default='info', required=False)
    parser.add_argument('--model_dir', help='model weight folder for to restore or save weights during training', default=None, required=False)
    parser.add_argument('--tensorboard_dir', help='folder to save logs for tensorboard visualization', required=True)
    parser.add_argument('--log_path', help='path to log file', required=True)
    args = parser.parse_args()
    return args



if __name__== "__main__":

    args = parse_args()
    print(args)
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    logging.basicConfig(filename=args.log_path, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=numeric_level)
    #logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=numeric_level)

    logging.info('script parameters')
    logging.info(args)


    data_dir=args.data_dir
    classes = ['__background__', 'horse']
    BATCH_SIZE = 10
    HEIGHT=48
    WIDTH=48

    net_params = {
        'input_height': HEIGHT,
        'input_width': WIDTH,
        'num_classes': len(classes),
        'learning_rate': 0.0001
    }
    logging.info('net parameters')
    logging.info(net_params)
    net = DvnNet(**net_params)

    data_update_rate = 100

    net.build_network()
    train_data = DataSet(data_dir=data_dir, classes=classes, batch_size=BATCH_SIZE, height=HEIGHT, width=WIDTH, mode='trainval')
    test_data = DataSet(data_dir=data_dir, classes=classes, batch_size=BATCH_SIZE, height=HEIGHT, width=WIDTH, mode='test')
    train_params = {
        'net': net,
        'train_data': train_data,
        'test_data': test_data,
        'data_update_rate': data_update_rate,
        'model_dir' : args.model_dir,
        'tensorboard_dir': args.tensorboard_dir
    }
    logging.info('training parameters')
    logging.info(train_params)
    main(**train_params)



