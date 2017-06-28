import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import logging

from dvn.src.util.loss import _oracle_score_cpu


def randomMask(shape):
    rand_mask =  np.random.rand(*shape)
    dims = len(shape)
    rand_mask[:, :, :, 1] = 1.0 - rand_mask[:, :, :, 0]
    return rand_mask

def blackMask(shape):
    black_batch = np.zeros(shape, dtype=np.float32)
    black_batch[:, :, :, 0] = 1.
    return black_batch
def greyMask(shape):
    batch = np.zeros(shape, dtype=np.float32)
    batch[:, :, :, 0] = 0.5
    batch[:, :, :, 1] = 0.5
    return batch

def pred_to_label(seg_masks):
    pred_labels = np.argmax(seg_masks, -1)
    return pred_labels

def label_to_colorimg(pred_labels, classes, color_map):
    imgs = np.zeros((pred_labels.shape[0], pred_labels.shape[1], pred_labels.shape[2], 3))
    for idx, c in enumerate(classes):
        class_mask = pred_labels == idx
        imgs[class_mask] = color_map[c]

def get_image_index_list(img_dir):
    assert os.path.exists(img_dir), "path %s doesnt exist" % img_dir
    index = [os.path.basename(os.path.splitext(f)[0]) for f in sorted(listdir(img_dir)) if isfile(join(img_dir, f))]
    extension = os.path.splitext(listdir(img_dir)[0])[1]
    return index, extension

def get_data_tuples(img_dir, labels_dir, index_list=None, extension=None):
    """
    returns absolute file paths in the dataPath directory
    :param dataPath: directory
    :param dataPath: directory
    :return: list of tuples in form of [(img_name, img_gt_name)]
    """
    assert os.path.exists(img_dir), "img dir %s doesnt exist" % img_dir
    if index_list:
        files = [f + extension for f in index_list]
    else:
        files = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]
    img_files = [join(img_dir, f) for f in files]

    if labels_dir:
        assert os.path.exists(labels_dir), "labels dir %s doesnt exist" % labels_dir
        label_files = [join(labels_dir, f) for f in files]
        assert len(img_files) == len(label_files), "image and label file numbers do not match"
        return [(img_files[i], label_files[i]) for i in range(len(img_files))]
    else:
        return [(img_files[i], None) for i in range(len(img_files))]

def color_decode(orig_img, classes, color_map):
    seg = np.zeros([orig_img.shape[0], orig_img.shape[1], len(classes)], dtype=np.float32)
    for id, key in enumerate(classes):
        seg[:, :, id] = np.all(orig_img == color_map[key], axis = 2)
    return seg

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

    #print("Pred Label _ Shape: ", pred_labels.shape, "gt: ", gt_labels.shape)

    true_positives = np.logical_and((pred_labels == gt_labels), (gt_labels == 1))

    false_positives = np.logical_and((gt_labels == 0), (pred_labels == 1))

    false_negatives = np.logical_and((gt_labels == 1), (pred_labels == 0))

    mapped_pred[true_positives] = np.array([0, 1, 0])  # green
    mapped_pred[false_positives] = np.array([1, 0, 0])  # red
    mapped_pred[false_negatives] = np.array([0, 0, 1])  # blue

    return mapped_pred

def generate_similar_image(img, iou):
    """
    returns generated image with an IoU difference of diff_iou to img
    :param img: image with only zero and 1s. last dimension sums up up to 1.
    :param diff_iou:
    :return:
    """

    assert len(img.shape) == 4
    mask = np.copy(img)
    batch_size, height, width, channels = img.shape
    mmin = np.ones([batch_size, channels])
    mmax = np.ones([batch_size, channels])

    logging.debug("target iou %s" % iou)
    # only change background layer
    for i in range(batch_size):
        mask_iou = 1.0
        while (mask_iou > iou):
            cancel = False
            logging.debug("batch i=%s, current_iou %s" %(i, mask_iou))
            repeat = 50
            while True:
                rand = np.random.randint(height * width)
                x = rand // height
                y = rand - x * width
                if mask[i][x][y][0] == 0. or mask[i][x][y][0] == 1.:
                    logging.debug("found pixel")
                    break
                else:
                    logging.debug("mask[%s][%s][%s][0] = %s" %(i, x, y, mask[i][x][y][0]))
                    repeat -= 1
                    logging.debug("repeat=%s" %repeat)
                    if repeat == 0:
                        cancel = True
                        break
            if cancel:
                break
            diff_pixel_value = sampleExponential(0.05, 1.0)
            if mask[i][x][y][0] == 0.:
                mask[i][x][y][0] += diff_pixel_value
            else:
                mask[i][x][y][0] -= diff_pixel_value

                logging.debug("new mask[%s][%s][%s][0] = %s" %(i, x, y, mask[i][x][y][0]))
            mask[i][x][y][1] = 1-mask[i][x][y][0]
            mask_iou = _oracle_score_cpu(mask[i], img[i])
        logging.debug("current done")
        logging.info("generated similar mask %s " %mask)
    return mask

def sampleExponential(beta, maxVal):
    """
    returns sample drawn from exponential distribution 1/beta * np.exp(-x/beta)
    in the range [0, maxVal]
    :param beta:
    :param maxVal:
    :return:
    """
    while True:
        rand = np.random.exponential(beta)
        if rand <= maxVal:
            return rand


def generate_random_sample(shape, teta, img_gt):
    diff_iou = sampleExponential(teta, 1.0)
    logging.debug("iou diff %s " % diff_iou)
    new_mask = generate_similar_image(img_gt, 1 - diff_iou)
    return new_mask
#
# for i in range(10):
#     print(sampleExponential(1., 1.0))
