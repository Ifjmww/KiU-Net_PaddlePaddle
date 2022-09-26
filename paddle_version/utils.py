import numpy as np
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict

from paddle.vision.transforms import RandomCrop, ColorJitter, RandomRotate, crop, hflip,to_tensor


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, crop=(32, 32), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if self.crop:
            rc = RandomCrop(self.crop)

            i, j, h, w = RandomCrop._get_param(image, self.crop)
            image, mask = crop(image, i, j, h, w), crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = hflip(image), hflip(mask)

        # color transforms || ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            angle = RandomRotate(180)._get_params((-90, 90))
            translate = (1, 1)
            scale_ranges = (2, 2)
            shears = (-45, 45)
            affine_params = (angle, translate, scale_ranges, shears)

            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image = to_tensor(image)
        if not self.long_mask:
            mask = to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask
