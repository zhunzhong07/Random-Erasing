from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        
        img = img.copy()
        h, w = img.shape
        
        area = h * w
       
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1/self.r1)

        erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

        if erase_h < h and erase_w < w:
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            
            img[x1 : x1 + erase_h, y1 : y1 + erase_w] = self.mean[0]

        return img

