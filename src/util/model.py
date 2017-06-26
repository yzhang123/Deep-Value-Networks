import logging

# Number of iterations for inference
ITERS = 30
learning_rate = 0.5
def inference(sess, graph, img, pred_mask, iterations=ITERS):

    for idx in range(iterations):
        feed_dict = {graph['x']: img, graph['y']: pred_mask}
        gradient = sess.run(graph['inference_grad'], feed_dict=feed_dict)
        pred_mask += learning_rate * gradient[0]
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 1] = 1

    #logging.debug("infer update %s" % pred_mask)
    return pred_mask

def adversarial(sess, graph, img, pred_mask, img_gt, iterations=ITERS):
    for i in range(iterations):
        feed_dict = {graph['x']: img, graph['y']: pred_mask, graph['y_gt']: img_gt}
        gradient = sess.run(graph['adverse_grad'], feed_dict=feed_dict)
        pred_mask += learning_rate * gradient[0]
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 1] = 1

    #logging.debug("infer update %s" % pred_mask)
    return pred_mask
