

# Number of iterations for inference
ITERS = 30

def inference(sess, graph, img, img_mask):

    feed_dict = {graph['x']: img, graph['y']: img_mask}

    inference_update = sess.run(graph['inference_update'], feed_dict=feed_dict)
    # print("generated data after first inference")
    # print(inference_update)
    for i in range(ITERS):
        feed_dict = {graph['x']: img}
        inference_update = sess.run(graph['inference_update'], feed_dict=feed_dict)
        # print(inference_update)

    return inference_update