"""The suction dataset.
"""

import glob
import logging
import multiprocessing
import os
import pickle

import sys
sys.path.append('/usr/local/lib/python3.6/pyrealsense2')

import pyrealsense2 as rs
import time
#import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from walle.core import RotationMatrix
from form2fit import config
from form2fit.code.utils import misc
from form2fit.code.utils.mask import suction_background_substract
#from get_align_img1012 import initial_camera,get_curr_image


class SuctionDataset(Dataset):
    """The suction network dataset.
    """

    def __init__(self, root, sample_ratio, augment, background_subtract, num_channels, radius):
        """Initializes the dataset.

        Args:
            root: (str) Root directory path.
            sample_ratio: (float) The ratio of negative to positive
                labels.
            normalize: (bool) Whether to normalize the images by
                subtracting the mean and dividing by the std deviation.
            augment: (bool) Whether to apply data augmentation.
        """
        self._root = root
        self._sample_ratio = sample_ratio
        self._augment = augment
        self._background_subtract = background_subtract
        self._num_channels = num_channels
        self._radius = radius

        # figure out how many data samples we have
        self._get_filenames()

        

        stats = pickle.load(open(os.path.join(Path(self._root).parent, "mean_std.p"), "rb"))
        if self._num_channels == 4:#_num_channels非4即2
            self._c_norm = transforms.Normalize(mean=stats[0][0] * 3, std=stats[0][1] * 3)
        else:
            self._c_norm = transforms.Normalize(mean=stats[0][0], std=stats[0][1])
        self._d_norm = transforms.Normalize(mean=stats[1][0], std=stats[1][1])
        self._transform = transforms.ToTensor()

    def __len__(self):
        return len(self._filenames)

    def _get_filenames(self):
        self._filenames = glob.glob(os.path.join(self._root, "*/"))
        self._filenames.sort(key=lambda x: int(x.split("/")[-2]))

    def _load_state(self, name):
        """Loads the raw state variables.
        """
        # load heightmaps
        c_height_f = np.asarray(Image.open(os.path.join(name, "final_color_height.png")))
        d_height_f = np.asarray(Image.open(os.path.join(name, "final_depth_height.png")))
        # c_height_i = np.asarray(Image.open(os.path.join(name, "init_color_height.png")))
        # d_height_i = np.asarray(Image.open(os.path.join(name, "init_depth_height.png")))

        # convert depth to meters
        # d_height_f = (d_height_f * 1e-3).astype("float32")
        # d_height_i = (d_height_i * 1e-3).astype("float32")

        return (
            c_height_f,
            d_height_f
        )

    def _split_heightmap(self, height, source):
        """Splits a heightmap into a source and target.

        For suction, we just need the target heightmap.
        """
        half = height.shape[1] // 2
        self._half = half
        height_t = height[:, :half].copy()
        height_s = height[:, half:].copy()
        if source:
            return height_s
        return height_t

    def __getitem__(self, idx):
        name = self._filenames[idx]

        # load state
        c_height, d_height = self._load_state(name)
        # pipeline, align, clipping_distance = initial_camera()
        # c_height_i,d_height_i = get_curr_image(pipeline, align, clipping_distance)

        # split heightmap into source and target
        c_height_f = self._split_heightmap(c_height, False)#color_obj
        d_height_f = self._split_heightmap(d_height, False)#depth_obj
        c_height_i = self._split_heightmap(c_height, True)#color_kit
        d_height_i = self._split_heightmap(d_height, True)#depth_kit

        assert c_height_f.shape==(480,424) 
        assert d_height_f.shape==(480,424)
        assert c_height_i.shape==(480,424)
        assert d_height_i.shape==(480,424)
        # self._H, self._W = c_height_f.shape[:2]#
        
        if self._background_subtract :
            c_height_i,d_height_i,c_height_f,d_height_f = suction_background_substract(c_height_i,d_height_i,c_height_f,d_height_f)
            

        # convert depth to meters
        d_height_f = (d_height_f * 1e-3).astype("float32")
        d_height_i = (d_height_i * 1e-3).astype("float32")

        if self._num_channels == 2:
            c_height_i = c_height_i[..., np.newaxis]
            c_height_f = c_height_f[..., np.newaxis]
        else:  # clone the gray channel 3 times
            c_height_i = np.repeat(c_height_i[..., np.newaxis], 3, axis=-1)
            c_height_f = np.repeat(c_height_f[..., np.newaxis], 3, axis=-1)

        # convert heightmaps tensors
        c_height_i = self._c_norm(self._transform(c_height_i))#(W,H,1)or (W,H,3)
        c_height_f = self._c_norm(self._transform(c_height_f))
        d_height_i = self._d_norm(self._transform(d_height_i[..., np.newaxis]))#(W,H,1)
        d_height_f = self._d_norm(self._transform(d_height_f[..., np.newaxis]))
        assert c_height_f.shape==(1,480,424)
        assert d_height_f.shape==(1,480,424)
        assert c_height_i.shape==(1,480,424)
        assert d_height_i.shape==(1,480,424)
        # concatenate height and depth into a 4-channel tensor
        img_tensor_i = torch.cat([c_height_i, d_height_i], dim=0)#kit,shape==(2,W,H)
        img_tensor_f = torch.cat([c_height_f, d_height_f], dim=0)#obj,shape==(2,W,H)
        img_tensor = torch.stack([img_tensor_i, img_tensor_f], dim=0)
        #img_tensor = torch.stack([img_tensor_i, img_tensor_f], dim=0)#shape == (2,2,W,H)
        assert img_tensor.shape ==(2,2,480,424)

        
        return img_tensor


def get_suction_loader(
    foldername,
    dtype="train",
    batch_size=1,
    sample_ratio=1,
    shuffle=True,
    augment=False,
    num_channels=2,
    background_subtract=None,
    radius=1,
    num_workers=4,
    use_cuda=True,
):
    """Returns a dataloader over the `Suction` dataset.

    Args:
        foldername: (str) The name of the folder containing the data.
        dtype: (str) Whether to use the train, validation or test partition.
        batch_size: (int) The number of data samples in a batch.
        sample_ratio: (float) The ratio of negative to positive
            labels.
        shuffle: (bool) Whether to shuffle the dataset at the end
            of every epoch.
        augment: (bool) Whether to apply data augmentation.
        num_workers: (int) How many processes to use. Each workers
            is responsible for loading a batch.
        use_cuda: (bool) Whether to use the GPU.
    """

    def _collate_fn(batch):
        """A custom collate function.

        This is to support variable length suction labels.
        """
        imgs = [b for b in batch]
        #labels = [b[1] for b in batch]
        imgs = torch.cat(imgs, dim=0)
        #labels = [l for sublist in labels for l in sublist]
        return imgs

    num_workers = min(num_workers, multiprocessing.cpu_count())
    root = os.path.join(config.ml_data_dir, foldername, dtype)

    dataset = SuctionDataset(
        root,
        sample_ratio,
        augment,
        background_subtract,
        num_channels,
        radius,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )

    return loader


