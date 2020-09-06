import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from parameters import tag_image, tag_label, tag_name, label_folder_name

import random
import os
from typing import Union


class AugManager(object):
    def __init__(self, iaalist=None):
        if iaalist is None:
            iaalist = iaa.Sequential([
                iaa.Sometimes(0.3, iaa.Fliplr(0.4)),
                iaa.Sometimes(0.3, iaa.Flipud(0.4)),
                iaa.Sometimes(0.5, iaa.Rotate((-30, 30)))
            ], random_order=True)
        self.transformSet = iaalist
        self.outscale = random.choice([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    def __call__(self, image):
        image = np.array(image)

        # size measure
        y_max = image.shape[0]
        x_max = image.shape[1]

        # augmentation
        zoomset = iaa.OneOf([
            iaa.Identity(),  # do nothing
            iaa.Affine(scale=self.outscale),  # zoom out
            RandomCrop(y_max, x_max).cut()  # zoom in
        ])

        image = zoomset(image=image)
        image = self.transformSet(image=image)

        image = Image.fromarray(image)

        return image


class RandomCrop(object):
    def __init__(self, max_height, max_width):
        assert isinstance(max_height, int) and max_height >= 1, 'max_height must be positive integer type.'
        assert isinstance(max_width, int) and max_width >= 1, 'max_width must be positive integer type.'

        self.percent_limit = 0.7
        self.top, self.right, self.bottom, self.left = self.operate_location(max_height, max_width)

    def operate_location(self, max_height, max_width):
        import random
        max_height = max_height + 1
        max_width = max_width + 1

        min_height = int(self.percent_limit * max_height)
        min_width = int(self.percent_limit * max_width)

        fix_height = random.randint(min_height, max_height)
        fix_width = random.randint(min_width, max_width)

        left = random.randint(0, max_width - fix_width)
        up = random.randint(0, max_height - fix_height)

        right = max_width - fix_width - left
        down = max_height - fix_height - up

        return up, right, down, left

    def cut(self):
        return iaa.Crop(px=(self.top, self.right, self.bottom, self.left))





















