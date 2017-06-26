import numpy as np
import tensorflow as tf
import logging

def _oracle_score(y, y_gt):
    """
    also referred to as v*, batch version
    :param y: segmantation masks [0, 1] * image_size x num_classes
    :param y_gt: ground truth segmantation mask
    :return: relaxed IoU score between both
    """

    y_min = tf.reduce_sum(tf.minimum(y, y_gt), [1, 2])
    y_max = tf.reduce_sum(tf.maximum(y, y_gt), [1, 2])
    y_divide = tf.divide(y_min, y_max)
    return tf.reduce_mean(y_divide, 1)


def _oracle_score_cpu(y, y_gt):
    """
    also referred to as v*
    :param y: segmantation masks [0, 1] * image_size x num_classes
    :param y_gt: ground truth segmantation mask
    :return: relaxed IoU score between both
    """
    if len(y.shape) == 4:
        y_min = np.sum(np.sum(np.minimum(y, y_gt), 2), 1)
        y_max = np.sum(np.sum(np.maximum(y, y_gt), 2), 1)
        y_divide = np.divide(y_min, y_max)
        return np.mean(y_divide, 1)
    elif len(y.shape) == 3:
        y_min = np.sum(np.sum(np.minimum(y, y_gt), 1), 0)
        y_max = np.sum(np.sum(np.maximum(y, y_gt), 1), 0)
        y_divide = np.divide(y_min, y_max)
        return np.mean(y_divide)
    else:
        raise Exception("wrong input dimension %s" % y.shape)

def calc_accuracy(img_reference, img_pred):
    ref_labels = np.argmax(img_reference, axis=-1)
    pred_labels = np.argmax(img_pred, axis=-1)
    logging.debug("gt: %s" % ref_labels)
    logging.debug("pred: %s" % pred_labels)
    correct = (pred_labels == ref_labels)
    acc = np.mean(correct)
    return acc

def calc_recall(img_reference, img_pred):
    ref_labels = np.argmax(img_reference, axis=-1)
    pred_labels = np.argmax(img_pred, axis=-1)
    logging.debug("gt: %s" % ref_labels)
    logging.debug("pred: %s" % pred_labels)
    true_positive = np.logical_and((pred_labels == ref_labels), ref_labels == 1)
    acc = np.sum(true_positive) *1.0 /np.sum(ref_labels == 1)
    return acc