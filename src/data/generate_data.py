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
        #self.generators = [self.gt, self.inference, self.sampling, self.adversarial]
        #self.generators = [self.gt, self.inference, self.random]
        self.generators = [self.generate_examples(train=True)]

    def generate(self):
        while True:
            yield next(random.choice(self.generators))
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


    def generate_examples(self, train=True):
        shape = (self.data.batch_size, self.data.size[0], self.data.size[1], self.data.num_classes)

        for img, img_gt in self.data:
            init_mask = self.get_initialization(shape)
            if train and np.random.rand() >= 0.5:
                logging.info("adverse")
                gt_indices = np.random.rand(img_gt.shape[0]) > 0.5
                init_mask[gt_indices] = img_gt[gt_indices]
                pred_mask = adverse(self.sess, self.graph, img, img_gt, init_mask)
            else:
                logging.info("inference")
                pred_mask = infer(self.sess, self.graph, img, init_mask)

            yield img, pred_mask, img_gt

    def get_initialization(self, shape):
        black_batch = np.zeros(shape, dtype=np.float32)
        return black_batch

if __name__=='__main__':
    img_path = join(dir_path, "../../", "data/weizmann_horse_db/rgb_1")
    test_img_path = join(dir_path, "../../", "data/weizmann_horse_db/gray_1")
    img_gt_path = join(dir_path, "../../", "data/weizmann_horse_db/figure_ground_1")
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
