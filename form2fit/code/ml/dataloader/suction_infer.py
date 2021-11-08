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
from skimage.measure import label
#from get_align_img1012 import initial_camera,get_curr_image

def remove_small_area(mask,area_th):
    print(type(mask),mask.shape)
    contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < area_th:
            cv2.drawContours(mask, [contours[i]], 0, 0, -1)
    return mask

def remove_surrounding_white(mask,visual):
    '''
    在mask中，去掉贴着图像边缘的白色部分（通常是背景）
    :param mask:
    :param visual: 可视化
    :return: mask：处理后的mask
    '''
    h,w = mask.shape
    labels = label(mask)
    if visual:
        cv2.imshow('labels going to remove_surrounding_white',(labels*40).astype('uint8'))
    num = np.max(labels)
    if visual:
        print('num in remove_surrounding_white',num)
    if num > 1:#如果只有一个连通域，不需要处理
        for i in range(num):
            domain = np.where(labels==i+1,1,0)
            if visual:
                cv2.imshow('domain in remove_surrounding_white',(domain.astype('uint8'))*255)
            rmin,rmax,cmin,cmax = mask2bbox(domain)
            if rmin ==0 or rmax == h-1 or cmin == 0 or cmax == w-1:
                labels = np.where(labels == i+1 , 0 , labels)
        mask = np.where(labels !=0,mask,0)
        if visual:
            cv2.imshow('mask in remove_surrounding_white',mask)
    return mask


def remove_inner_black(mask,visual):
    '''
    在mask中去掉白色部分中间的黑色
    :param mask:
    :param visual: 可视化
    :return: mask：处理后的mask
    '''
    h, w = mask.shape
    mask = 255 - mask
    labels = label(mask)
    if visual:
        cv2.imshow('labels going to remove_inner_black', (labels * 40).astype('uint8'))
    num = np.max(labels)
    if visual:
        print('num in remove_inner_black', num)
    for i in range(num):
        domain = np.where(labels == i + 1, 1, 0)
        if visual:
            cv2.imshow('domain in remove_inner_black', (domain.astype('uint8')) * 255)
        rmin, rmax, cmin, cmax = mask2bbox(domain)
        if not (rmin == 0 or rmax == h - 1 or cmin == 0 or cmax == w - 1):
            labels = np.where(labels == i + 1, 0, labels)
    mask = np.where(labels != 0, mask, 0)
    mask = 255 - mask
    if visual:
        cv2.imshow('mask in remove_inner_black', mask)
    return mask

def mask2bbox(mask):#寻找二值化图像的mask=1处的方框
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

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
        '''if self._background_subtract is not None:
            # 用深度图区分盒子和物体
            idxs = np.vstack(np.where(d_height_i > self._background_subtract[0])).T
            mask = np.zeros_like(d_height_i)
            mask[idxs[:, 0], idxs[:, 1]] = 1
            mask = misc.largest_cc(mask)
            idxs = np.vstack(np.where(mask == 1)).T
            mask = np.zeros_like(d_height_i)
            # 生成不包含白边的mask
            mask[idxs[:, 0].min():idxs[:, 0].max(), idxs[:, 1].min():idxs[:, 1].max()] = 1
            # mask = np.zeros_like(d_height_i)
            # mask[idxs[:, 0], idxs[:, 1]] = 1
            # mask = misc.largest_cc(np.logical_not(mask))
            idxs = np.vstack(np.where(mask == 0)).T
            # 把用深度图选出的白边置为0
            c_height_i[idxs[:, 0], idxs[:, 1]] = 0
            d_height_i[idxs[:, 0], idxs[:, 1]] = 0
            # 终止位置图片同上
            idxs = np.vstack(np.where(d_height_f > self._background_subtract[1])).T
            mask = np.zeros_like(d_height_f)
            mask[idxs[:, 0], idxs[:, 1]] = 1
            mask = misc.largest_cc(np.logical_not(mask))
            idxs = np.vstack(np.where(mask == 1)).T
            c_height_f[idxs[:, 0], idxs[:, 1]] = 0
            d_height_f[idxs[:, 0], idxs[:, 1]] = 0
        if self._num_channels == 2:
            c_height_i = c_height_i[..., np.newaxis]#(W,H,1)
            c_height_f = c_height_f[..., np.newaxis]
        else:  # clone the gray channel 3 times
            c_height_i = np.repeat(c_height_i[..., np.newaxis], 3, axis=-1)#(W,H,3)
            c_height_f = np.repeat(c_height_f[..., np.newaxis], 3, axis=-1)'''
        if self._background_subtract :#TODO:修改条件
            
            thre_otsu, img_otsu = cv2.threshold(d_height_i,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            bool_otsu = misc.largest_cc(img_otsu)#largest_cc返回值是一个bool类型的矩阵
            mask = np.zeros_like(d_height_i)
            # cv2.imshow("otsu",(bool_otsu*255).astype('uint8'))
            rmin,rmax,cmin,cmax = misc.mask2bbox(bool_otsu)
            mask[rmin:rmax,cmin:cmax] = 1
            c_height_i = np.where(mask, c_height_i, 0)
            d_height_i = np.where(mask, d_height_i, 0)
            # cv2.imshow('d_height_i', d_height_i)
            # cv2.imshow('c_height_i', c_height_i)
           
            seg_img_obj = cv2.adaptiveThreshold(d_height_f, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,401, 2)
            # seg_img_obj = remove_inner_black(seg_img_objs)
            # cv2.imshow('adap_gaus_inv',adap_thre_gaus_inv)
            seg_img_obj = 255 - seg_img_obj
            seg_img_obj = remove_small_area(seg_img_obj,700)
            # seg_img_obj = remove_surrounding_white(seg_img_obj)
            #本来是黑部分就设为黑色，然后把白色的桌面设成0
            # mask = np.logical_or(bg_mask,desk_mask)
            assert mask.shape == d_height_f.shape
            c_height_f = np.where(mask,0,c_height_f)
            d_height_f = np.where(mask,0,d_height_f)

        # convert depth to meters
        d_height_f = (d_height_f * 1e-3).astype("float32")
        d_height_i = (d_height_i * 1e-3).astype("float32")
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

        #第一个2是kit和obj；第二个2是color和depth
        ########################img_tensor is going to be returned##################################

        # # offset indices to adjust for splitting
        # pos_suction_i[:, 1] = pos_suction_i[:, 1] - self._half
        #
        # pos_f = []
        # for pos in pos_suction_f:  # len(pos_suction_f) == N
        #     rr, cc = circle(pos[0], pos[1], self._radius)
        #     pos_f.append(np.vstack([rr, cc]).T)
        # pos_suction_f = np.concatenate(pos_f)
        # pos_i = []
        # for pos in pos_suction_i:  # len(pos_suction_i) == 1
        #     rr, cc = circle(pos[0], pos[1], self._radius)
        #     pos_i.append(np.vstack([rr, cc]).T)
        # pos_suction_i = np.concatenate(pos_i)
        #
        # # add columns of 1 (positive labels)
        # pos_label_i = np.hstack((pos_suction_i, np.ones((len(pos_suction_i), 1))))
        # pos_label_f = np.hstack((pos_suction_f, np.ones((len(pos_suction_f), 1))))
        #
        # # generate negative labels
        # neg_suction_i = np.vstack(self._sample_negative(pos_label_i)).T
        # neg_label_i = np.hstack((neg_suction_i, np.zeros((len(neg_suction_i), 1))))
        # neg_suction_f = np.vstack(self._sample_negative(pos_label_f)).T
        # neg_label_f = np.hstack((neg_suction_f, np.zeros((len(neg_suction_f), 1))))
        #
        # # stack positive and negative into a single array
        # label_i = np.vstack((pos_label_i, neg_label_i))
        # label_f = np.vstack((pos_label_f, neg_label_f))
        #
        # # convert suction points to tensors
        # label_tensor_i = torch.LongTensor(label_i)
        # label_tensor_f = torch.LongTensor(label_f)
        # label_tensor = [label_tensor_i, label_tensor_f]
        
        # img1 = img_tensor_i[0,:,:]
        # img1 = np.expand_dims(img1, axis=2)
        # print(img1.shape)
        # cv2.imwrite("img1.jpg", img1)
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
        imgs = [b[0] for b in batch]
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


