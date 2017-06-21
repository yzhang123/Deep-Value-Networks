import os
from os.path import join
from dkfz.convnet.generate_images import save_images

module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)  # store dir_path for later use
root_path = join(dir_path, "../")

def write_image(images, iteration, name, mapping=True):
    if mapping:
        images = ((images + 1.) * (255.99 / 2)).astype('int32')
    else:
        images = images.astype('int32')
    save_images(images, root_path + '/output/' + name + '{}.png'.format(iteration), False)
