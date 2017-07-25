import os
from os.path import join
from dkfz.convnet.generate_images import save_images
from dvn.src.util.data import pred_to_label, result_sample_mapping

module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)  # store dir_path for later use
root_path = join(dir_path, "../../")

def write_image(images, iteration, output_dir, name, mapping=True):
    if mapping:
        images = ((images + 1.) * (255.99 / 2)).astype('int32')
    else:
        images = images.astype('int32')
    write_path = join(root_path, output_dir, name + '-{}.png'.format(iteration+1))
    save_images(images, write_path, False)
    print("written to %s " %write_path)


def write_mask(mask, mask_gt, output_dir, name, iteration):
    #labels = pred_to_label(mask)
    mapp_pred = result_sample_mapping(mask_gt, mask)
    write_image(images=mapp_pred, iteration=iteration, output_dir=output_dir, name=name)
    #write_image(images=mask[..., 1], iteration=iteration, output_dir=output_dir, name=name)
