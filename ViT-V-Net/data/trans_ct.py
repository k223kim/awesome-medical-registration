import random
import collections
import numpy as np
import torch, sys, random, math
from scipy import ndimage

from .rand import Constant, Uniform, Gaussian
from scipy.ndimage import rotate
from skimage.transform import rescale, resize

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class RandomHVFlip(object):
    def __init__(self):
        self.axis = 0
    def __call__(self, data):    
        axis = np.random.randint(0, 3)
        data = [np.flip(d, axis) for d in data]
        return data
    def __str__(self):
        return "RandomHVFlip()"

class Random90Rotation(object):
    def __init__(self):
        self.num = 0 
        self.axes = (1, 2)
    def __call__(self, data):
        num = np.random.randint(0, 4)
        data = [np.rot90(d, axes=self.axes, k=num) for d in data]
        return data
    def __str__(self):
        return "Random90Rotation()"

class NumpyType(object):# types: ('float32', 'int64')
    def __init__(self, types):
        self.types = types # ('float32', 'int64')

    def __call__(self, data):
        # make this work with both Tensor and Numpy
        if len(self.types) == len(data):
            data = [d.astype(t) for d, t in zip(data, self.types)]
        else:
            factor = len(data) / len(self.types)
            self.types = np.repeat(np.array(self.types), factor)
            data = [d.astype(t) for d, t in zip(data, self.types)]
        return data
    def __str__(self):
        return "NumpyType()"