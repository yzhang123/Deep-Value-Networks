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
import _init_paths
print(sys.path)
from understand_tensorflow.src.deeplearning import save_model

module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)  # store dir_path for later use
root_path = join(dir_path, "../")


from model.dvn import DvnNet
from data.data_set import DataSet

ITERS = 60
ITERS_PER_SAVE = 50
SAVE_PATH = join(root_path, 'checkpoints/')

def train(graph, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        iter = initial_step = graph['global_step'].eval()

        for img, img_gt in data:
            iter += 1
            assert img.shape[0:1] == img_gt.shape[0:1]

            feed_dict = {graph['x']: img, graph['y_gt']: img_gt}

            sess.run(graph['y'].assign(img_gt))
            x, y = sess.run([graph['fc3'], graph['loss']], feed_dict=feed_dict)
            _, loss, gt_diff = sess.run([graph['train_optimizer'], graph['loss'], graph['sim_score']],
                                        feed_dict=feed_dict)
            print(graph['y'].eval())

            print("iteration %s: loss = %s, sim_score = %s" % (iter, loss[0], gt_diff[0]))

            #save model?
            if iter % ITERS_PER_SAVE == 0:
                save_model(sess, SAVE_PATH, 'model', global_step=iter)
            if iter >= initial_step + ITERS:
                break


def test(graph, modelpath, data):
    ITERS = 50
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(modelpath)
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        for img, black_mask in data:
            assert img.shape[0:1] == black_mask.shape[0:1]

            feed_dict = {graph['x']: img}
            sess.run(graph['y'].assign(black_mask))

            for i in range(ITERS):
                score = sess.run(graph['fc3'], feed_dict=feed_dict)


if __name__== "__main__":
    img_path = join(dir_path, "../", "data/weizmann_horse_db/rgb")
    test_img_path = join(dir_path, "../", "data/weizmann_horse_db/gray")
    img_gt_path = join(dir_path, "../", "data/weizmann_horse_db/figure_ground")
    print("img_dir %s" % img_path)
    print("img_gt_dir %s" % img_gt_path)


    net = DvnNet(classes = ['__background__', 'horse'], batch_size=1)
    graph = net.build_network()
    train_data = DataSet(img_path, img_gt_path, batch_size=1)

    train(graph, train_data)

    test_data = DataSet(test_img_path, None, batch_size=1)

    modelpath = '/home/yang/projects/dvn/checkpoints/model-50.meta'
    test(graph, modelpath, test_data)


