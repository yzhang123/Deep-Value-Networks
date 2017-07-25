import os
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import logging

from dvn.src.util.measures import oracle_score


def oneMask(shape):
    batch = np.ones(shape, dtype=np.float32)
    return batch

def left_upper1_4_mask(shape):
    batch = np.ones(shape, dtype=np.float32)
    if len(shape) == 3:
        height = shape[0]/4
        width = shape[1]/4
        batch[:height, :width, 0] = 0.
        batch[:, :, 1] = 1. - batch[ :, :, 0]
        return batch
    elif len(shape)  == 4:
        height = shape[1]/4
        width = shape[2]/4
        batch[:, height, :width, 0] = 0.
        batch[..., 1] = 1. - batch[..., 0]
        return batch
    else:
        raise Exception('shape need to have length 3 or 4 but has length %s' % len(shape))


def left_upper2_4_mask(shape):
    batch = np.ones(shape, dtype=np.float32)
    if len(shape) == 3:
        height = shape[0]/2
        width = shape[1]/2
        batch[:height, :width, 0] = 0.
        batch[:, :, 1] = 1. - batch[ :, :, 0]
        return batch
    elif len(shape)  == 4:
        height = shape[1]/2
        width = shape[2]/2
        batch[:, height, :width, 0] = 0.
        batch[..., 1] = 1. - batch[..., 0]
        return batch
    else:
        raise Exception('shape need to have length 3 or 4 but has length %s' % len(shape))

def left_upper3_4_mask(shape):
    batch = np.ones(shape, dtype=np.float32)
    if len(shape) == 3:
        height = shape[0]*3/4
        width = shape[1]*3/4
        batch[:height, :width, 0] = 0.
        batch[:, :, 1] = 1. - batch[ :, :, 0]
        return batch
    elif len(shape)  == 4:
        height = shape[1]*3/4
        width = shape[2]*3/4
        batch[:, height, :width, 0] = 0.
        batch[..., 1] = 1. - batch[..., 0]
        return batch
    else:
        raise Exception('shape need to have length 3 or 4 but has length %s' % len(shape))



def left_upper2_2_mask(shape):
    batch = np.ones(shape, dtype=np.float32)
    if len(shape) == 3:
        height = shape[0]/4
        width = shape[1]/4
        batch[height:-height, width:-width, 0] = 0.
        batch[..., 1] = 1. - batch[..., 0]
        return batch
    elif len(shape) == 4:
        height = shape[1]/4
        width = shape[2]/4
        batch[:, height:-height, width:-width, 0] = 0.
        batch[..., 1] = 1. - batch[..., 0]
        return batch
    else:
        raise Exception('shape need to have length 3 or 4 but has length %s' % len(shape))


def zeroMask(shape):
    black_batch = np.zeros(shape, dtype=np.float32)
    return black_batch

def blackMask(shape):
    black_batch = np.zeros(shape, dtype=np.float32)
    black_batch[..., 0] = 1.
    return black_batch

def meanMask(shape):
    batch = 0.5 * np.ones(shape, dtype=np.float32)
    return batch

def randomMask(shape):
    batch = np.zeros(shape, dtype=np.float32)
    if len(shape) == 3 or len(shape) == 4:
        batch[..., 0] = np.random.rand()
        batch[..., 1] = 1 - batch[..., 0]
        return batch
    else:
        raise Exception('shape need to have length 3 or 4 but has length %s' % len(shape))

def pred_to_label(seg_masks):
    pred_labels = np.argmax(seg_masks, -1)
    return pred_labels

def label_to_colorimg(pred_labels, classes, color_map):
    imgs = np.zeros((pred_labels.shape[0], pred_labels.shape[1], pred_labels.shape[2], 3))
    for idx, c in enumerate(classes):
        class_mask = pred_labels == idx
        imgs[class_mask] = color_map[c]

def get_image_index_list(data_dir):
    assert os.path.exists(data_dir), "path %s doesnt exist" % data_dir
    img_dir = join(data_dir, 'images')
    assert os.path.exists(img_dir), "images directory %s doesnt exist %" % img_dir
    index = [os.path.basename(os.path.splitext(f)[0]) for f in sorted(listdir(img_dir))]
    extension = os.path.splitext(listdir(data_dir)[0])[1]
    return index, extension

def get_file_extention(file_dir):
    first_file_in_dir = os.listdir(file_dir)[0]
    return os.path.splitext(first_file_in_dir)[1]

def get_data_tuples(data_dir, index_list):
    """
    returns absolute file paths in the dataPath directory
    :param dataPath: directory
    :param dataPath: directory
    :return: list of tuples in form of [(img_name, img_gt_name)]
    """
    assert os.path.exists(data_dir), "img dir %s doesnt exist" % data_dir
    images_dir = join(data_dir, 'images')
    annotations_dir = join(data_dir, 'annotations')
    img_file_ext = get_file_extention(images_dir)
    annotation_file_ext = get_file_extention(annotations_dir)


    list_abs_img_file_path = [join(images_dir, f + img_file_ext) for f in index_list]
    list_abs_annotation_file_path = [join(annotations_dir, f + annotation_file_ext) for f in index_list]

    return zip(list_abs_img_file_path, list_abs_annotation_file_path)

def read_file_to_list(file_path):
    with open(file_path) as f:
        return f.read().splitlines()


def color_decode(orig_img, classes, color_map):
    seg = np.zeros([orig_img.shape[0], orig_img.shape[1], len(classes)], dtype=np.float32)
    for id, key in enumerate(classes[1:]):
        #seg[:, :, id+1] = np.all(orig_img == color_map[key], axis = 2)
        seg[:, :, id+1] = np.all(orig_img > (127, 127, 127), axis = 2)
    seg[:, :, 0] = 1 - seg[:, :, 1]
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
    # only change background layer
    for i in range(batch_size):
        mask_iou = 1.0
        while (mask_iou > iou):
            cancel = False
            logging.debug("batch i=%s, current_iou %s" %(i, mask_iou))
            repeat = 50
            while True:
                rand = np.random.randint(height * width)
                x = rand // width
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
            mask_iou = oracle_score(mask[i], img[i])
        logging.debug("current done")
        logging.info("similar mask %s has mean %s " %(i, np.mean(mask[i, ..., 1])))
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
    new_mask = generate_similar_image(img_gt, 1 - diff_iou)
    return new_mask
#
# for i in range(10):
#     print(sampleExponential(1., 1.0))


def binarize_image(img):
    """
    :param img: numpy image, can have batch dimension
    :return:
    """
    max_indices = img == img.max(axis=-1, keepdims=True)
    min_indices = True ^ max_indices
    bin_image = img.copy()
    bin_image[max_indices] = 1.
    bin_image[min_indices] = 0.
    return bin_image


