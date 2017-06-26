import numpy as np
import random
from os.path import join, dirname, abspath
import logging


from dvn.src.util.loss import _oracle_score_cpu
from dvn.src.util.data import randomMask, blackMask, sampleExponential
from dvn.src.util.model import inference as infer, adversarial as adverse
from dvn.src.util.data import generate_similar_image


module_path = abspath(__file__)
dir_path = dirname(module_path)
root_path = join(dir_path, "../../")
SAVE_PATH = join(root_path, 'checkpoints/')

class DataGenerator(object):

    def __init__(self, sess, graph, data):
        self.sess = sess
        self.graph = graph
        self.data = data # (img, img_gt)
        self.generators = [self.gt, self.inference, self.sampling, self.adversarial]
        #self.generators = [self.gt, self.inference, self.random]

    def generate(self):
        functions = map(lambda x: x(), self.generators)
        while True:
            yield next(random.choice(functions))
        #return self.random()
        #return self.sampling()

    def gt(self):
        for img, img_gt in self.data:

            logging.info("gt")
            yield img, img_gt, img_gt

    def black(self):
        shape = (1, self.data.size[0], self.data.size[1], self.data.num_classes)
        black_batch = blackMask(shape)
        for img, img_gt in self.data:
            yield img, black_batch, img_gt


    def random(self):
        theta = 0.05
        for img, img_gt in self.data:

            logging.info("random")
            while True:
                shape = (1, self.data.size[0], self.data.size[1], self.data.num_classes)
                random_batch = randomMask(shape)
                sim = _oracle_score_cpu(random_batch, img_gt)
                norm_factor = 1./ (np.exp(theta) - 1)
                prob = norm_factor * np.exp(theta * sim[0]) - norm_factor
                rand = np.random.rand()
                if rand < prob:
                    yield img, random_batch, img_gt
                    break
                # else:
                #     print('fail')

    def sampling(self):
        teta = 0.05
        for img, img_gt in self.data:
            logging.info("sampling")
            diff_iou = sampleExponential(teta, 1.0)
            logging.info("iou diff %s " % diff_iou)
            new_mask = generate_similar_image(img_gt, 1-diff_iou)
            yield img, new_mask, img_gt


    def inference(self):
        shape = (1, self.data.size[0], self.data.size[1], self.data.num_classes)
        black_batch = blackMask(shape)

        for img, img_gt in self.data:
            logging.info("inference")
            inference_update = infer(self.sess, self.graph, img, black_batch)
            yield img, inference_update, img_gt

    def adversarial(self):
        shape = (1, self.data.size[0], self.data.size[1], self.data.num_classes)
        black_batch = blackMask(shape)

        for img, img_gt in self.data:
            logging.info("adversarial")
            adversarial_update = adverse(self.sess, self.graph, img, black_batch)
            yield img, adversarial_update, img_gt



if __name__=='__main__':
    img_path = join(dir_path, "../../", "data/weizmann_horse_db/rgb")
    test_img_path = join(dir_path, "../../", "data/weizmann_horse_db/gray")
    img_gt_path = join(dir_path, "../../", "data/weizmann_horse_db/figure_ground")
    print("img_dir %s" % img_path)
    print("img_gt_dir %s" % img_gt_path)

    classes = ['__background__', 'horse']
    from dvn.src.model.dvn import DvnNet
    from dvn.src.data.data_set import DataSet
    import tensorflow as tf
    net = DvnNet(classes=classes, batch_size = 1)

    graph = net.build_network(train=True)
    train_data = DataSet(classes, img_path, img_gt_path, batch_size=1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        generator = DataGenerator(sess, graph, train_data)
        for a, b, c in generator.generate():
            print("a")
            print(a)
            print("b")
            print(b)
            print("c")
            print(c)
            print("------------------------------")
