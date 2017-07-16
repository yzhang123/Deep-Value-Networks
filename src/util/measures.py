import numpy as np
import tensorflow as tf
import logging



def oracle_score(masks_a, masks_b):
    """

    :param masks_a: batch
    :param masks_b: batch
    :return: batches of relaxed iou scores, 1-d tensor
    """
    assert masks_a.shape == masks_b.shape
    assert len(masks_a.shape) == 3 or len(masks_b.shape) == 4, 'input has wrong shape'
    if len(masks_a.shape) == 4:
        y_min = np.sum(np.sum(np.minimum(masks_a, masks_b), 2), 1)
        y_max = np.sum(np.sum(np.maximum(masks_a, masks_b), 2), 1)
        y_divide = np.divide(y_min, np.maximum(y_max, 10 ** -8))
        y_divide[y_max == 0.] = 1.
        scores = np.mean(y_divide, 1)
        assert scores.shape[0] == masks_a.shape[0], "target_score.shape: %s, masks.shape: %s" % (
            scores.shape, masks_a.shape)

    if len(masks_a.shape) == 3:
        y_min = np.sum(np.sum(np.minimum(masks_a, masks_b), 1), 0)
        y_max = np.sum(np.sum(np.maximum(masks_a, masks_b), 1), 0)
        y_divide = np.divide(y_min, np.maximum(y_max, 10 ** -8))
        y_divide[y_max == 0.] = 1.
        scores = np.mean(y_divide)

    assert (0.0 <= scores).all and (scores <= 1.0).all, 'oracle scores are in [0, 1]'
    return scores


def calc_accuracy(img_reference, img_pred):
    ref_labels = np.argmax(img_reference, axis=-1)
    pred_labels = np.argmax(img_pred, axis=-1)
    # logging.debug("gt: %s" % ref_labels)
    # logging.debug("pred: %s" % pred_labels)
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



if __name__=='__main__':
    from dvn.src.model.dvn import DvnNet
    from dvn.src.data.data_set import DataSet
    import tensorflow as tf
    from os.path import join, abspath, dirname
    from dvn.src.data.generate_data import DataGenerator

    module_path = abspath(__file__)
    dir_path = dirname(module_path)
    root_path = join(dir_path, "../../")
    SAVE_PATH = join(root_path, 'checkpoints/')

    img_path = join(dir_path, "../../", "data/weizmann_horse_db/rgb_1")
    test_img_path = join(dir_path, "../../", "data/weizmann_horse_db/gray_1")
    img_gt_path = join(dir_path, "../../", "data/weizmann_horse_db/figure_ground_1")
    print("img_dir %s" % img_path)
    print("img_gt_dir %s" % img_gt_path)
    classes = ['__background__', 'horse']




    train_data = DataSet(classes, img_path, img_gt_path, batch_size=1, train=False)
    # img, gt = next(train_data.__iter__())

    img = np.load('arrays/x.npy')
    gt = np.load('arrays/y.npy')
    y = np.load('arrays/y-1.npy')
    print("iimg %s "% img)
    print("gt %s "% gt)
    print("y %s "% y)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        score = tf.reduce_mean(_oracle_score(gt, y))
        score_cpu = _oracle_score_cpu(gt, y)
        s1  = sess.run(score)
        print("s1: %s" % s1)
        print("s2: %s" %score_cpu)