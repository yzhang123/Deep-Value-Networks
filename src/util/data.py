import numpy as np

def randomMask(shape):
    rand_mask =  np.random.rand(*shape)
    dims = len(shape)
    rand_mask[:, :, :, 1] = 1.0 - rand_mask[:, :, :, 0]
    return rand_mask

def blackMask(shape):
    black_batch = np.zeros(*shape, dtype=np.float32)
    black_batch[:, :, :, 0] = 1.
    return black_batch


def pred_to_label(seg_masks):
    pred_labels = np.argmax(seg_masks, -1)
    return pred_labels

def label_to_colorimg(pred_labels, classes, color_map):
    imgs = np.zeros((pred_labels.shape[0], pred_labels.shape[1], pred_labels.shape[2], 3))
    for idx, c in enumerate(classes):
        class_mask = pred_labels == idx
        imgs[class_mask] = color_map[c]

