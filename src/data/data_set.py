# -*- coding: utf-8 -*-
# Compatibility to python 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


import numpy as np
import random
from PIL import Image
import os
from os.path import isfile, join
from dvn.src.util.data import get_data_tuples, color_decode, get_image_index_list
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import logging

class BatchIterator(object):


    def __init__(self, iterator, batch_size):
        self.iterator = iterator
        self.batch_size = batch_size
        self.batch_iterator = self.stack()

    def __iter__(self):
        return self.batch_iterator

    def stack(self):
        """
        Aggregates list of datapoints to list of batched datapoints
        Creates new dimension for batches in the first dimension, so images are expected as 3D array
        """
        results = []
        data_count = 0

        while True:
            try:
                dp = next(self.iterator)
            except StopIteration:
                if data_count > 0:
                    yield [np.stack(batch, axis = 0) for batch in results]
                return

            if len(results) == 0:
                for _ in range(len(dp)):
                    results.append(list())

            for i in range(len(dp)):
                results[i].append(dp[i])

            data_count += 1

            if data_count >= self.batch_size:
                yield [np.stack(batch, axis=0) for batch in results]
                data_count = 0
                results = []



class DataSet(object):

    def __init__(self, classes, img_dir, gt_dir=None, batch_size=1, size=(0, 0), train=False, repeat=None, shuffle=None):

        self.trainingSet = []
        self.validationSet = []
        self.repeat = train
        self.shuffle = train

        if repeat is not None:
            self.repeat = repeat
        if shuffle is not None:
            self.shuffle = shuffle

        logging.info('repeat: %s' %self.repeat)
        logging.info('shuffle: %s' %self.shuffle)
        self.testSet = []
        self.classes = classes
        self.num_classes = len(classes)
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.color_map = self.get_color_map()
        self.height = size[0]
        self.width = size[1]
        self.batch_size = batch_size
        # index_list is only file name without file extension
        self.index_list, self.img_ext = get_image_index_list(img_dir)
        logging.info('img extension %s' %self.img_ext)
        # data_tuples = [(absolute img path, absolute img mask path)]
        self.data_tuples = get_data_tuples(img_dir, gt_dir, self.index_list, self.img_ext)
        self.batch_iterator = BatchIterator(self.image_iterator(repeat=self.repeat, shuffle=self.shuffle), batch_size=batch_size)
        #self.avg_img = gaussian_filter(self.get_avg_img(img_dir), 3)
        #self.avg_img, self.avg_mask = map(lambda x: gaussian_filter(x, 3), self.get_avg_img(img_dir))

    def __iter__(self):
        return iter(self.batch_iterator)


    def image_iterator(self, repeat=False, shuffle=False):
        if shuffle:
            random.shuffle(self.data_tuples)
        data_iterator = iter(self.data_tuples)
        while True:
            try:
                img_file, seg_file = next(data_iterator)
                img = Image.open(img_file)
                if img.size != (self.height, self.width):
                    img = img.resize((self.height, self.width), resample=Image.BILINEAR)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = np.asarray(img, dtype=np.uint8)# - self.mean_pixel
                img = np.asarray(img, dtype=np.float32) / 255.

                if self.gt_dir:
                    seg = Image.open(seg_file)
                    if seg.mode != "RGB":
                        seg = seg.convert("RGB")
                    if seg.size != (self.height, self.width):
                        seg = seg.resize((self.height, self.width), resample=Image.NEAREST)

                    #if img.size != seg.size:
                    #    raise OverflowError('Image and label size do not match %s != %s ' %(img.size, seg.size))
                    seg = np.array(seg, dtype=np.uint8)
                    seg = color_decode(seg, self.classes, self.color_map)
                else:
                    seg = np.zeros([self.height, self.width, self.num_classes], dtype=np.float32)
                    seg[:, :, 0] = 1.

                assert ((0 <= img) & (img <= 1.)).all()
                assert ((0. == seg) | (seg == 1.)).all()
                yield [img, seg]

            except StopIteration:
                if repeat:

                    if shuffle:
                        random.shuffle(self.data_tuples)
                    data_iterator = iter(self.data_tuples)
                else:
                    return
            except IOError:
                print('------- Broken image: %s or %s -------' % (img_file, seg_file))


    def get_color_map(self):
        color_dict = dict()
        color_dict['__background__'] = [0, 0, 0]
        color_dict['horse'] = [255, 255, 255]
        return color_dict


    def get_avg_img(self, img_dir):
        avg_img = np.zeros([self.height, self.width, 3], np.uint8)
        avg_mask = np.zeros([self.height, self.width, 3], np.uint8)
        for f1, f2 in self.data_tuples:
            img = Image.open(f1)
            mask = Image.open(f2)
            img = img.convert("RGB")
            mask = mask.convert("RGB")
            img = img.resize((self.height, self.width), resample=Image.BILINEAR)
            mask = mask.resize((self.height, self.width), resample=Image.NEAREST)
            img = np.asarray(img, dtype=np.uint8)
            mask = np.asarray(mask, dtype=np.uint8)
            avg_img += img
            avg_mask += mask
        avg_img = avg_img / len(self.data_tuples)
        avg_mask = avg_mask / len(self.data_tuples)
        return avg_img, avg_mask




if __name__=='__main__':
    module_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(module_path)  # store dir_path for later use
    root_path = join(dir_path, "../../")
    log_dir = join(root_path, "logs")
    model_dir = join(root_path, "model")

    img_path = join(root_path, "data/weizmann_horse_db/rgb")
    test_img_path = join(root_path, "data/weizmann_horse_db/rgb")
    img_gt_path = join(root_path, "data/weizmann_horse_db/figure_ground")

    classes = ['__background__', 'horse']
    data = DataSet(classes, img_path, img_gt_path, batch_size=1, train=True, size=(24, 24))

    import scipy.misc


    test_img = np.zeros([100, 100], dtype=np.float32)
    test_img[:, :50] = 1

    scipy.misc.imsave('test_img.png', data.avg_mask)

    idx = 0
    for img, img_gt in data:
        scipy.misc.imsave('img_%s.png' %idx, img[0, ...])
        scipy.misc.imsave('gt0_%s.png' % idx, img_gt[0, ..., 0])
        scipy.misc.imsave('gt1_%s.png'% idx, img_gt[0, ..., 1])
        idx += 1
        if (idx > 10):
            break
