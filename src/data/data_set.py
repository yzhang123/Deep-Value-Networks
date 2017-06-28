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

    def __init__(self, classes, img_dir, gt_dir=None, batch_size=1, size=(24, 24), train=True):

        self.trainingSet = []
        self.validationSet = []
        self.testSet = []
        self.classes = classes
        self.num_classes = len(classes)
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.color_map = self.get_color_map()
        self.size = size
        self.batch_size = batch_size
        self.index_list, self.img_ext = get_image_index_list(img_dir)
        self.data_tuples = get_data_tuples(img_dir, gt_dir, self.index_list, self.img_ext)
        self.mean_pixel = self.get_mean_pixel(img_dir)
        self.batch_iterator = BatchIterator(self.image_iterator(self.data_tuples, repeat=train, shuffle=train), batch_size=batch_size)

    def __iter__(self):
        return iter(self.batch_iterator)

    def image_iterator(self, data_tuples, index_list=None, repeat=True, shuffle=True):
        #assert isinstance(data_tuples, list), "data tuples it not list but %s " % type(data_tuples)
        if shuffle:
            random.shuffle(data_tuples)
        data_iterator = iter(data_tuples)

        while True:
            try:
                img_file, seg_file = next(data_iterator)
                img = Image.open(img_file)
                if img.size != self.size:
                    img = img.resize(self.size, resample=Image.BILINEAR)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = np.asarray(img, dtype=np.uint8)# - self.mean_pixel
                img = np.asarray(img, dtype=np.float32) / 255.

                if self.gt_dir:
                    seg = Image.open(seg_file)
                    if seg.size != self.size:
                        seg = seg.resize(self.size, resample=Image.NEAREST)

                    #if img.size != seg.size:
                    #    raise OverflowError('Image and label size do not match %s != %s ' %(img.size, seg.size))
                    if seg.mode != "RGB":
                        seg = seg.convert("RGB")
                    seg = np.asarray(seg, dtype=np.uint8)
                    seg = color_decode(seg, self.classes, self.color_map)
                else:
                    seg = np.zeros([self.size[0], self.size[1], self.num_classes], dtype=np.float32)
                    seg[:, :, 0] = 1.

                yield [img, seg]

            except StopIteration:
                if repeat:
                    if shuffle:
                        random.shuffle(data_tuples)
                    data_iterator = iter(data_tuples)
                else:
                    return
            except IOError:
                print('------- Broken image: %s or %s -------' % (img_file, seg_file))


    def get_color_map(self):
        color_dict = dict()
        color_dict['__background__'] = [0, 0, 0]
        color_dict['horse'] = [255, 255, 255]
        return color_dict


    def get_mean_pixel(self, img_dir):
        files = [a for a, b in self.data_tuples]
        mean_pixel = np.zeros([self.size[0], self.size[1], 3], np.uint8)
        for f in files:
            img = Image.open(f)
            img = img.resize(self.size, resample=Image.BILINEAR)
            img = img.convert("RGB")
            img = np.asarray(img, dtype=np.uint8)
            mean_pixel += img
        mean_pixel = mean_pixel / len(files)
        return mean_pixel





