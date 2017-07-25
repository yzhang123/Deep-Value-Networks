import numpy as np
import random
from os.path import join, dirname, abspath
import logging


from dvn.src.util.measures import oracle_score
from dvn.src.util.data import randomMask, blackMask, sampleExponential, zeroMask, oneMask
from dvn.src.util.data import left_upper1_4_mask, left_upper2_4_mask, left_upper3_4_mask, left_upper2_2_mask

from dvn.src.util.model import inference as infer, adversarial as adverse
from dvn.src.util.data import generate_random_sample
from dvn.src.util.input_output import write_mask




module_path = abspath(__file__)
dir_path = dirname(module_path)
root_path = join(dir_path, "../../")

class DataGenerator(object):

    def __init__(self, session, net, data, mode, data_update_rate):
        """

        :param session:
        :param net: network class object
        :param data: DataSet object
        :param train:
        :param data_update_rate:
        """
        self.session = session
        self.net = net
        self.data = data # (img, img_gt)
        self.data_update_rate = data_update_rate
        self.mode = mode

    # def generate(self):
    #     while True:
    #         yield next(random.choice(self.generators))

    def reset(self):
        self.data.reset()

    def gt(self):
        return self.data.__iter__()


    # def gen_batch(self):
    #     """
    #
    #     :return: images, input masks, and according ground truth masks as batches
    #     """
    #     shape = (self.data.height, self.data.width, self.data.num_classes)
    #     masks = list()
    #     masks.append(left_upper2_2_mask(shape))
    #     masks.append(left_upper1_4_mask(shape))
    #     masks.append(blackMask(shape))
    #     masks.append(left_upper2_4_mask(shape))
    #     masks.append(left_upper3_4_mask(shape))
    #     masks.append(zeroMask(shape))
    #
    #     def _get_mask(img_mask):
    #         rand_idx = np.random.randint(0, len(masks) + 1)
    #         logging.info("mask %s" %rand_idx)
    #         if rand_idx < len(masks):
    #             return masks[rand_idx]
    #         else:
    #             return img_mask
    #
    #     while(True):
    #         imgs, img_masks = next(self.gt())
    #         input_masks = list()
    #         for i in range(img_masks.shape[0]):
    #             mask = _get_mask(img_masks[i])
    #             input_masks.append(mask)
    #
    #         input_masks = np.stack(input_masks, axis=0)
    #
    #         assert imgs.shape > input_masks.shape, "imgs.shape : %s, input_masks.shape : %s" % (
    #         imgs.shape, input_masks.shape)
    #         assert img_masks.shape == input_masks.shape
    #
    #         yield imgs, input_masks, img_masks
    #

    def generate_batch(self):
        idx = 0
        for img, mask_gt in self.data:
            shape = mask_gt.shape
            init_mask = self.get_initialization(shape)
            rand = np.random.rand()
            if self.mode == 'train' or self.mode == 'trainval':
                if rand > 0.7:
                    logging.info("gt")
                    pred_mask = mask_gt
                    # gt_indices = np.random.rand(mask_gt.shape[0]) > 0.5
                    # init_mask[gt_indices] = mask_gt[gt_indices].copy()
                    # pred_mask = adverse(session=self.session, net=self.net, img=img, init_mask=init_mask,
                    #                     mask_gt=mask_gt, data_update_rate=self.data_update_rate, train=train, iterations=3)
                else:
                    logging.info("inference")
                    # pred_mask = infer(session=self.session, net=self.net, img=img, init_mask=init_mask,
                    #                   data_update_rate=self.data_update_rate, train=train, iterations=20)
                    pred_masks = infer(session=self.session, net=self.net, img=img, init_mask=init_mask,
                                      data_update_rate=100, train=True, iterations=100)
                    assert len(pred_masks) == 1
                    pred_mask = pred_masks[0][0]
                # elif rand > 0.1:
                #     logging.info("rand")
                #     teta = 0.05
                #     pred_mask = generate_random_sample(shape, teta, mask_gt)
                #
                #
                # else:
                #     logging.info("inference + gt")
                #     pred_mask = infer(session=self.session, net=self.net, img=img, init_mask=init_mask,
                #                       data_update_rate=self.data_update_rate, train=train, iterations=20)
                #     number_elements = len(np.reshape(pred_mask, -1))
                #     rand_positions = np.random.choice(number_elements, int(0.5 * number_elements))
                #     for pos in rand_positions:
                #         idx3 = pos % self.data.num_classes
                #         pos  = pos // self.data.num_classes
                #         idx2 = pos % self.data.width
                #         pos  = pos // self.data.width
                #         idx1 = pos % self.data.height
                #         pos  = pos // self.data.height
                #         idx0 = pos % self.data.batch_size
                #         pred_mask[idx0][idx1][idx2][idx3] = mask_gt[idx0][idx1][idx2][idx3]

                    # logging.info("adverse + gt")
                    # gt_indices = np.random.rand(mask_gt.shape[0]) > 0.5
                    # init_mask[gt_indices] = mask_gt[gt_indices].copy()
                    # pred_mask = adverse(session=self.session, net=self.net, img=img, init_mask=init_mask,
                    #                     mask_gt=mask_gt, data_update_rate=self.data_update_rate, train=train,
                    #                     iterations=20)
                    # number_elements = len(np.reshape(pred_mask, -1))
                    # rand_positions = np.random.choice(number_elements, int(0.5 * number_elements))
                    # for pos in rand_positions:
                    #     idx3 = pos % self.data.num_classes
                    #     pos = pos // self.data.num_classes
                    #     idx2 = pos % self.data.width
                    #     pos = pos // self.data.width
                    #     idx1 = pos % self.data.height
                    #     pos = pos // self.data.height
                    #     idx0 = pos % self.data.batch_size
                    # pred_mask[idx0][idx1][idx2][idx3] = mask_gt[idx0][idx1][idx2][idx3]
            elif self.mode == 'test' or self.mode == 'val':
                logging.info("inference")
                pred_masks = infer(session=self.session, net=self.net, img=img, init_mask=init_mask,
                                  data_update_rate=100, train=False, iterations=300)

                # visualize stages of inference and write out
                for mask, iter in pred_masks:
                    write_mask(mask=mask, mask_gt=mask_gt, output_dir='output', name=self.data.file_index_list[idx], iteration=iter)
                idx += img.shape[0]

                # only return final mask in inference process
                pred_mask = pred_masks[-1][0]
            yield img, pred_mask, mask_gt
            # yield img, init_mask, mask_gt

    def get_initialization(self, shape):
        black_batch = blackMask(shape)
        return black_batch


if __name__=='__main__':
    import os
    from dvn.src.data.data_set import DataSet
    import tensorflow as tf
    module_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(module_path)  # store dir_path for later us
    root_path = join(dir_path, "../../")
    log_dir = join(root_path, "logs")
    model_dir = join(root_path, "model")
    data_dir = join(root_path, "data/weizmann_horse_db")
    classes = ['__background__', 'horse']



    with tf.Session() as sess:
        data = DataSet(data_dir=data_dir, classes=classes, batch_size=10, height=48, width=48, mode='test')
        generator = DataGenerator(session=sess, net=None, data=data, mode='val', data_update_rate=100)

        idx = 0
        while(True):
            for img, img_gt in generator.gt():
                print(idx)
                idx +=1
            data.reset()

