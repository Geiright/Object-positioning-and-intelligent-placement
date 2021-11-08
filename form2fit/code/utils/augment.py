# Author:LiPu
import matplotlib.pyplot as plt
import cv2
import random
import math
import numpy as np


class RandomErasing(object):

    def __init__(self, probability=0.9, sl=0.02, sh=0.4, r1=0.3, mean=[0.914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        batchsize = img.shape[0]
        for i in range(batchsize):
            sp = img.shape
            # print(sp)
            area = sp[2] * sp[3]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < sp[3] and h < sp[2]:
                x1 = random.randint(0, sp[2] - h)
                y1 = random.randint(0, sp[3] - w)
                img[i, 0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                img[i, 1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                img[i, 2, x1:x1 + h, y1:y1 + w] = self.mean[2]
        return img