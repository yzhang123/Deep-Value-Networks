import logging
import tensorflow as tf
from dvn.src.util.measures import oracle_score

# Number of iterations for inference
ITERS_TRAIN = 30
ITERS_TEST = 30
update_rate = 0.1


def inference(session, net, img, init_mask, data_update_rate, train, iterations):
    logging.debug('update rate : %s' % data_update_rate)
    pred_mask = init_mask
    for idx in range(iterations):
        feed_dict = {net.x: img, net.y: pred_mask}
        gradient = session.run(net.inference_grad, feed_dict=feed_dict)
        pred_mask += data_update_rate * gradient[0]
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 1] = 1
        logging.debug("infer iter %s : %s" % (idx, gradient))
        logging.debug("infer update %s" % pred_mask)


    return pred_mask

def adversarial(session, net, img, init_mask, mask_gt, data_update_rate=update_rate, train=False, iterations=ITERS_TEST):

    pred_mask = init_mask
    for idx in range(iterations):
        target_scores = oracle_score(pred_mask, mask_gt)
        feed_dict = {net.x: img, net.y: pred_mask, net.target_score: target_scores}
        gradient = session.run(net.adverse_grad, feed_dict=feed_dict)
        pred_mask += data_update_rate * gradient[0]
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 1] = 1
        #logging.debug("adverse iter %s : %s" % (idx, gradient))

    #logging.debug("adverse update %s" % pred_mask)
    return pred_mask
