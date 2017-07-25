import logging
import tensorflow as tf
from dvn.src.util.measures import oracle_score

# Number of iterations for inference
ITERS_TRAIN = 30
ITERS_TEST = 30
update_rate = 0.1

ITERS_PER_SAVE = 10


def inference(session, net, img, init_mask, data_update_rate, train, iterations):
    """
    executes inference on initial mask init_mask given an image img for given number of iterations. The inference comprises of
    determining the gradient of the output score on the mask and updating the mask by adding the gradient with a factor of data_update_rate
    :param session: session the inference shall run in
    :param net: network
    :param img: image
    :param init_mask: initial mask which is used for inference
    :param data_update_rate: factor which the gradient is multiplied with
    :param train: true for training mode, false for testing mode
    :param iterations: number of inference iterations
    :return: list of inferred masks. In training mode the list contains only the final mask, in testing mode a number of masks are saved
    """
    logging.debug('update rate : %s' % data_update_rate)
    pred_mask = init_mask

    result_masks = []
    for idx in range(iterations):
        feed_dict = {net.x: img, net.y: pred_mask}
        gradient = session.run(net.inference_grad, feed_dict=feed_dict)
        pred_mask += data_update_rate * gradient[0]
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 1] = 1
        logging.debug("infer iter %s : %s" % (idx, gradient))
        logging.debug("infer update %s" % pred_mask)

        if not train and idx == 0  and (ITERS_PER_SAVE != 0):
            result_masks.append((pred_mask.copy(), idx))
        if not train and ((idx+1) % ITERS_PER_SAVE == 0):
            result_masks.append((pred_mask.copy(), idx))

    if not train and (iterations % ITERS_PER_SAVE != 0):
        result_masks.append((pred_mask.copy(), iterations))

    if len(result_masks) == 0:
        result_masks.append((pred_mask.copy(), iterations))

    return result_masks

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
