import paddle
import os
import numpy


def train(args):
    if args.crop is not None:
        crop = (args.crop, args.crop)
    else:
        crop = None
    tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
