import numpy as np
import random
from os.path import join, dirname, abspath
import logging


from dvn.src.util.loss import _oracle_score_cpu
from dvn.src.util.data import randomMask, blackMask, sampleExponential, zeroMask, oneMask, left_upper_Mask, left_lower_Mask, right_upper_Mask, right_lower_Mask
from dvn.src.util.data import left_upper1_4_mask, left_upper2_4_mask, left_upper3_4_mask, left_upper4_4_mask, left_upper2_2_mask

from dvn.src.util.model import inference as infer, adversarial as adverse
from dvn.src.util.data import generate_random_sample

from nn_toolbox.src.datasets.iterators import BatchIterator



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

        # mask4 = right_lower_Mask(shape)
        def _get_mask(img_gt):
            shape = (self.data.size[0], self.data.size[1], self.data.num_classes)
            mask0 = img_gt
            mask1 = left_upper2_2_mask(shape)
            mask2 = left_upper1_4_mask(shape)
            mask3 = randomMask(shape)
            mask4 = zeroMask(shape)

            #logging.debug("img_gt %s" % img_gt[0, :, :, 0])
            idx = np.random.randint(0, 5)
            if idx == 0:
                print("mask: gt")
                return mask0
            elif idx == 1:
                print("mask: center crop")
                return mask1
            elif idx == 2:
                print("mask: left crop")
                return mask2
            elif idx == 3:
                print("mask: random")
                return mask3
            elif idx == 4:
                print("mask: zero")
                return mask4

        while(True):
            img, _, img_gt = next(self.gt())
            all_masks = []
            for i in range(img_gt.shape[0]):
                mask = _get_mask(img_gt[i])
                all_masks.append(mask)
            all_masks = np.stack(all_masks, axis=0)

            data = [img, all_masks, img_gt]
            yield data






    def generate_examples(self, train=False):
        shape = (self.data.batch_size, self.data.size[0], self.data.size[1], self.data.num_classes)

        for img, img_gt in self.data:
            init_mask = self.get_initialization(shape)
            rand = np.random.rand()
            if train:
                if rand > 0.20:
                    logging.info("adverse")
                    gt_indices = np.random.rand(img_gt.shape[0]) > 0.3
                    init_mask[gt_indices] = img_gt[gt_indices].copy()
                    pred_mask = adverse(self.sess, self.graph, img, init_mask, img_gt, data_update_rate=self.data_update_rate, train=train, iterations=3)
                elif rand > 0.10:
                    logging.info("inference")
                    pred_mask = infer(self.sess, self.graph, img, init_mask, data_update_rate=self.data_update_rate, train=train, iterations=20)
                else:
                    # logging.info("rand")
                    # teta = 0.05
                    # pred_mask = generate_random_sample(shape, teta, img_gt)

                    logging.info("inference + gt")
                    pred_mask = infer(self.sess, self.graph, img, init_mask, data_update_rate=self.data_update_rate, train=train, iterations=20)
                    number_elements = len(np.reshape(pred_mask, -1))
                    rand_positions = np.random.choice(number_elements, int(0.5 * number_elements))
                    for pos in rand_positions:
                        idx3 = pos % self.data.num_classes
                        pos  = pos // self.data.num_classes
                        idx2 = pos % self.data.size[1]
                        pos  = pos // self.data.size[1]
                        idx1 = pos % self.data.size[0]
                        pos  = pos // self.data.size[0]
                        idx0 = pos % self.data.batch_size
                        pred_mask[idx0][idx1][idx2][idx3] = img_gt[idx0][idx1][idx2][idx3]
            else:
                pred_mask = infer(self.sess, self.graph, img, init_mask, data_update_rate=self.data_update_rate, train=train, iterations=20)
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
    from dvn.src.util.loss import _oracle_score_cpu

    data = DataSet(classes, img_path, img_gt_path, batch_size=1)
    generator = DataGenerator(sess=None, graph=None, data=data, train=False, data_update_rate=0)
    for img, mask, img_gt in generator.helper():
        print("mask shape ")
        print( mask.shape)
        print("y_mean")
        print(np.mean(mask, (1,2))[..., 1])
        print("oracle score ")
        print(_oracle_score_cpu(mask, img_gt))
        print("\n")