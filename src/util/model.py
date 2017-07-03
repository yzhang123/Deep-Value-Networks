import logging
import tensorflow as tf

# Number of iterations for inference
ITERS_TRAIN = 50
ITERS_TEST = 500
update_rate = 0.1
def inference(sess, graph, img, pred_mask, data_update_rate=update_rate, train=False):

    iterations = ITERS_TRAIN if train else ITERS_TEST
    for idx in range(iterations):
        feed_dict = {graph['x']: img, graph['y']: pred_mask}
        gradient = sess.run(graph['inference_grad'], feed_dict=feed_dict)
        pred_mask += update_rate * gradient[0]
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 1] = 1
        logging.debug("infer iter %s : %s" % (idx, gradient))

    logging.debug("infer update %s" % pred_mask)
    return pred_mask

def adversarial(sess, graph, img, pred_mask, img_gt, data_update_rate=update_rate, train=False):

    iterations = ITERS_TRAIN if train else ITERS_TEST

    for idx in range(iterations):
        feed_dict = {graph['x']: img, graph['y']: pred_mask, graph['y_gt']: img_gt}
        gradient = sess.run(graph['adverse_grad'], feed_dict=feed_dict)
        pred_mask += update_rate * gradient[0]
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 1] = 1
        logging.debug("adverse iter %s : %s" % (idx, gradient))

    logging.debug("adverse update %s" % pred_mask)
    return pred_mask
