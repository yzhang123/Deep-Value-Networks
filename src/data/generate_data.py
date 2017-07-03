import numpy as np
import random
from os.path import join, dirname, abspath
import logging


from dvn.src.util.loss import _oracle_score_cpu
from dvn.src.util.data import randomMask, blackMask, sampleExponential, zeroMask, oneMask, left_upper_Mask, left_lower_Mask, right_upper_Mask, right_lower_Mask
from dvn.src.util.data import left_upper1_4_mask, left_upper2_4_mask, left_upper3_4_mask, left_upper4_4_mask, left_upper2_2_mask

from dvn.src.util.model import inference as infer, adversarial as adverse
from dvn.src.util.data import generate_random_sample


module_path = abspath(__file__)
dir_path = dirname(module_path)
root_path = join(dir_path, "../../")
SAVE_PATH = join(root_path, 'checkpoints/')

class DataGenerator(object):

    def __init__(self, sess, graph, data, train, data_update_rate):
        self.sess = sess
        self.graph = graph
        self.data = data # (img, img_gt)
        self.data_update_rate = data_update_rate
        self.generators = [self.generate_examples(train=train)]

    def generate(self):
        while True:
            yield next(random.choice(self.generators))
            #yield next(self.gt())

    def gt(self):
        for img, img_gt in self.data:
            logging.info("gt")
            yield img, img_gt, img_gt


    def helper(self):
        shape = (self.data.batch_size, self.data.size[0], self.data.size[1], self.data.num_classes)
        i = 0
        mask0 = left_upper1_4_mask(shape)
        mask1 = left_upper2_4_mask(shape)
        mask2 = left_upper3_4_mask(shape)
        mask3 = left_upper4_4_mask(shape)
        mask4 = zeroMask(shape)

        # mask0 = blackMask(shape)
        # mask1 = oneMask(shape)
        # mask2 = zeroMask(shape)
        # mask3 = left_upper_Mask(shape)
        # mask4 = right_lower_Mask(shape)


        while(True):
            img, _, img_gt = next(self.gt())
            logging.debug("img_gt %s" % img_gt[0, :, :, 0])
            if i % 5 == 0:
                yield img, mask0, img_gt
            elif i % 5 == 1:
                yield img, mask1, img_gt
            elif i % 5 == 2:
                yield img, mask2, img_gt
            elif i % 5 == 3:
                yield img, mask3, img_gt
            elif i % 5 == 4:
                yield img, mask4, img_gt
            i += 1
            i %= 5



    def generate_examples(self, train=False):
        shape = (self.data.batch_size, self.data.size[0], self.data.size[1], self.data.num_classes)

        for img, img_gt in self.data:
            init_mask = self.get_initialization(shape)
            rand = np.random.rand()
            if train:
                if rand > 0.55:
                    logging.info("adverse")
                    gt_indices = np.random.rand(img_gt.shape[0]) > 0.5
                    init_mask[gt_indices] = img_gt[gt_indices].copy()
                    pred_mask = adverse(self.sess, self.graph, img, init_mask, img_gt, data_update_rate=self.data_update_rate, train=train)
                elif rand > 0.15:
                    logging.info("inference")
                    pred_mask = infer(self.sess, self.graph, img, init_mask, data_update_rate=self.data_update_rate, train=train)
                else:
                    logging.info("rand")
                    teta = 0.05
                    pred_mask = generate_random_sample(shape, teta, img_gt)
            else:
                pred_mask = infer(self.sess, self.graph, img, init_mask, data_update_rate=self.data_update_rate, train=train)
            yield img, pred_mask, img_gt
            # yield img, init_mask, img_gt

    def get_initialization(self, shape):
        black_batch = zeroMask(shape)
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
