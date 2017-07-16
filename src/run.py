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
from dvn.src.util.data import pred_to_label, label_to_colorimg, blackMask, result_sample_mapping
from dvn.src.util.input_output import write_image
from dvn.src.util.model import inference as infer
from dvn.src.util.measures import calc_accuracy, calc_recall, oracle_score




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



def train(net, data, data_update_rate, model_dir, tensorboard_dir):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('loading model %s' % ckpt.model_checkpoint_path)
            load_model(session=sess, path_model=ckpt.model_checkpoint_path)

        train_writer = tf.summary.FileWriter(tensorboard_dir + '/train', sess.graph)
        generator = DataGenerator(sess, net, data, train=True, data_update_rate=data_update_rate)
        # x, _, gt = next(generator.gt())
        # logging.info("image: %s" % x)
        # logging.info("gt: %s" % gt)
        # np.save('arrays/x.npy', x)
        # np.save('arrays/y.npy', gt)

        #np.set_printoptions(formatter={'all': lambda x: str(x) + '\n'})

        iter = initial_step = net.global_step.eval()
        for imgs, input_masks, img_masks in generator.generate_batch(train=True):
            target_scores = oracle_score(input_masks, img_masks)
            feed_dict = {net.x : imgs, net.y: input_masks, net.target_score: target_scores}

            summary, _, lr, loss, outputs, score_diffs, y_mean = sess.run([net.merged_summary, net.optimizer, net.adam._lr_t, net.loss, net.output, net.score_diff, net.y_mean], feed_dict=feed_dict)
            logging.info("iter %s: lr=%s, \noutputs = %s, \ntargets=%s, \nscore_diff=%s, \nloss = %s," %(iter, lr, outputs, target_scores, score_diffs, loss))

            train_writer.add_summary(summary, iter)

            iter += 1
            if iter % ITERS_PER_SAVE == 0:
                print('saving %s/model-%s' % (model_dir, iter))
                save_model(sess, model_dir, 'model', global_step=iter)

            if iter >= initial_step + ITERS_TRAIN:
                break

        train_writer.close()


def test(net, data, modelpath, data_update_rate, tensorboard_dir):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        print('loading model %s' % modelpath)
        load_model(session=sess, path_model=modelpath)

        generator = DataGenerator(sess, net, data, train=False, data_update_rate=100)

        writer = tf.summary.FileWriter(tensorboard_dir + '/test', sess.graph)

        np.set_printoptions(formatter={'all': lambda x: str(x) + '\n'})

        iter = 0
        for imgs, input_masks, img_masks in generator.generate_batch(train=False):

            # target_scores = oracle_score(input_masks, img_masks)
            # feed_dict = {net.x : imgs, net.y: input_masks, net.target_score: target_scores}
            # summary, loss, outputs, score_diffs, y_mean = sess.run(
            #     [net.merged_summary, net.loss, net.output, net.score_diff, net.y_mean], feed_dict=feed_dict)

            # writer.add_summary(summary, iter)
            acc = calc_accuracy(img_masks, input_masks)
            recall = calc_recall(img_masks, input_masks)
            logging.info("iter i = %s, acc = %s, recall = %s" %(iter, acc, recall))
            logging.debug("img %s" % imgs)
            labels = pred_to_label(input_masks)
            mapp_pred = result_sample_mapping(img_masks, input_masks)
            write_image(mapp_pred, -1, data.index_list[iter])

            #logging.info("iter %s: \noutputs = %s, \ntargets=%s, \nscore_diff=%s, \nloss = %s," %(iter, outputs, target_scores, score_diffs, loss))
            iter += 1
        writer.close()


img_path = join(root_path, "data/weizmann_horse_db/rgb")
img_gt_path = join(root_path, "data/weizmann_horse_db/figure_ground")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--data', default=img_path)
    parser.add_argument('--data_gt', default=img_gt_path)
    parser.add_argument('--loglevel', default='info')
    parser.add_argument('--model_dir')
    parser.add_argument('--tensorboard_dir')
    parser.add_argument('--log_path')
    parser.add_argument('--model', type=str, help='model path, e.g. ../model-500')
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

    img_path=args.data
    img_gt_path=args.data_gt
    classes = ['__background__', 'horse']
    BATCH_SIZE = 10
    SIZE = (48, 48)

    net_params = {
        'input_height': SIZE[0],
        'input_width': SIZE[1],
        'num_classes': 2,
        'learning_rate': 0.0001
    }
    logging.info('net parameters')
    logging.info(net_params)
    net = DvnNet(**net_params)

    data_update_rate = 10

    net.build_network()
    if args.train:
        data = DataSet(classes=classes, img_dir=img_path, gt_dir=img_gt_path, batch_size=BATCH_SIZE, size=SIZE, train=True, repeat=True, shuffle=True)
        train_params = {
            'net': net,
            'data': data,
            'data_update_rate': data_update_rate,
            'model_dir' : args.model_dir,
            'tensorboard_dir': args.tensorboard_dir
        }
        logging.info('training parameters')
        logging.info(train_params)
        train(**train_params)
    else:
        data = DataSet(classes=classes, img_dir=img_path, gt_dir=img_gt_path, batch_size=1, size=SIZE, train=False, repeat=False, shuffle=False)

        if args.model:
            modelpath = args.model
        elif args.model_dir:
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                modelpath = ckpt.model_checkpoint_path
            else:
                raise Exception('no models found in specified directory %s' % args.model_dir)
        else:
            raise Exception('no model specified')

        # logging.debug("data tuples")
        # logging.debug(data.data_tuples)

        test_params = {
            'net': net,
            'data': data,
            'data_update_rate': data_update_rate,
            'modelpath' : modelpath,
            'tensorboard_dir': args.tensorboard_dir
        }
        logging.info('test parameters')
        logging.info(test_params)
        test(**test_params)


