# -*- coding: utf-8 -*-
# Compatibility to python 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import operator
import numpy as np
import random
from PIL import Image
import os
from os.path import join
from dvn.src.util.data import get_data_tuples, color_decode, read_file_to_list, get_file_extention
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

    def __init__(self, data_dir, classes, batch_size=1, height=48, width=48, mode='train', repeat=None, shuffle=None):
        """

        :param data_dir: folder that containes subfolder 'images', 'annotations' and index files 'train.txt', 'val.txt', 'test.txt', 'trainval.txt'
        :param classes: list of classnames, including __background__
        :param batch_size: batchsize
        :param height: image height
        :param width: image width
        :param mode: ['train', 'trainval', 'val', 'test'], determines which index file to read, modes have default repeat and shuffle values
        :param repeat: repeat data set, if set overrides mode default values
        :param shuffle: shuffle data set, if set overrides mode default values
        """

        self.mode = mode

        if mode == 'train' or mode == 'trainval':
            self.repeat = True
            self.shuffle = True
        elif mode == 'val' or mode == 'test':
            self.repeat = False
            self.shuffle = False
        else:
            raise Exception('mode %s unknown' % mode)
        if repeat is not None:
            self.repeat = repeat
        if shuffle is not None:
            self.shuffle = shuffle
        logging.info('repeat: %s' %self.repeat)
        logging.info('shuffle: %s' %self.shuffle)

        self.data_dir = data_dir
        self.classes = classes
        self.num_classes = len(classes)

        self.file_index_list = read_file_to_list(join(data_dir, mode+'.txt')) #list of names of files without directory name and extension, in sorted manner
        self.images_dir = join(data_dir, 'images')
        self.annotations_dir = join(data_dir, 'annotations')
        self.img_file_ext = get_file_extention(self.images_dir)
        self.annotation_file_ext = get_file_extention(self.annotations_dir)


        self.color_map = self.get_color_map()
        self.width = width
        self.height = height
        self.batch_size = batch_size
        #data_tuples = [(absolute img path, absolute img mask path)]
        self.data_tuples = get_data_tuples(data_dir=self.data_dir, index_list=self.file_index_list)

        self.indices = np.arange(len(self.data_tuples))
        self.data_iterator = self.image_iterator(repeat=self.repeat, shuffle=self.shuffle)
        self.batch_iterator = BatchIterator(iterator=self.data_iterator, batch_size=self.batch_size)
        #self.avg_img = gaussian_filter(self.get_avg_img(img_dir), 3)
        #self.avg_img, self.avg_mask = map(lambda x: gaussian_filter(x, 3), self.get_avg_img(img_dir))

    def __iter__(self):
        return iter(self.batch_iterator)

    def reset(self):
        self.data_iterator = self.image_iterator(repeat=self.repeat, shuffle=self.shuffle)
        self.batch_iterator = BatchIterator(iterator=self.data_iterator, batch_size=self.batch_size)


    def image_iterator(self, repeat=False, shuffle=False):
        """

        :param repeat:
        :param shuffle:
        :return: tuple of numpy arrays [img, seg]. img has 3 color channels, and np.float32 values are between 0 and 1.
                 seg has as many channels as there are classes (including background), np.uint8 binary values
        """
        data_tuples = self.data_tuples
        if shuffle:
            random.shuffle(self.indices)
            data_tuples = operator.itemgetter(*self.indices)(self.data_tuples)
        data_iterator = iter(data_tuples)

        while True:

            try:
                img_file, seg_file = next(data_iterator)
                img = Image.open(img_file)
                if img.size != (self.width, self.height):
                    img = img.resize((self.width, self.height), resample=Image.BILINEAR)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = np.asarray(img, dtype=np.uint8)# - self.mean_pixel
                img = np.asarray(img, dtype=np.float32) / 255.

                seg = Image.open(seg_file)
                if seg.mode != "RGB":
                    seg = seg.convert("RGB")
                if seg.size != (self.width, self.height):
                    seg = seg.resize((self.width, self.height), resample=Image.NEAREST)

                #if img.size != seg.size:
                #    raise OverflowError('Image and label size do not match %s != %s ' %(img.size, seg.size))
                seg = np.array(seg, dtype=np.uint8)
                seg = color_decode(seg, self.classes, self.color_map)


                assert ((0 <= img) & (img <= 1.)).all()
                assert ((0. == seg) | (seg == 1.)).all()
                yield [img, seg] # shape (self.height, self.width, 3)

            except StopIteration:
                if repeat:

                    if shuffle:
                        random.shuffle(self.indices)
                        data_tuples = operator.itemgetter(*self.indices)(self.data_tuples)
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


    # def get_avg_img(self, img_dir):
    #     avg_img = np.zeros([self.height, self.width, 3], np.uint8)
    #     avg_mask = np.zeros([self.height, self.width, 3], np.uint8)
    #     for f1, f2 in self.data_tuples:
    #         img = Image.open(f1)
    #         mask = Image.open(f2)
    #         img = img.convert("RGB")
    #         mask = mask.convert("RGB")
    #         img = img.resize((self.height, self.width), resample=Image.BILINEAR)
    #         mask = mask.resize((self.height, self.width), resample=Image.NEAREST)
    #         img = np.asarray(img, dtype=np.uint8)
    #         mask = np.asarray(mask, dtype=np.uint8)
    #         avg_img += img
    #         avg_mask += mask
    #     avg_img = avg_img / len(self.data_tuples)
    #     avg_mask = avg_mask / len(self.data_tuples)
    #     return avg_img, avg_mask




if __name__=='__main__':
    module_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(module_path)  # store dir_path for later us
    root_path = join(dir_path, "../../")
    log_dir = join(root_path, "logs")
    model_dir = join(root_path, "model")

    data_dir = join(root_path, "data/weizmann_horse_db")

    classes = ['__background__', 'horse']
    data = DataSet(data_dir=data_dir, classes=classes, batch_size=1, height=48, width=48, mode='test')

    import scipy.misc


    # scipy.misc.imsave('test_img.png', data.avg_mask)

    # idx = 0
    # while(True):
    #     for img, img_gt in data:
    #         print(idx)
    #         idx +=1
    #     data.reset()
    ref =data.image_iterator(data)
    print(id(ref))
    ref =data.image_iterator(data)
    print(id(ref))