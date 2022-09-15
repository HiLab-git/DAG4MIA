#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 22:02
# @Author  : Ran.Gu
'''
Define a dataset class for MS-CMRSeg challenge (.png, .jpg) dataset
'''
import os
import torch
import numpy as np
import itertools
import imageio
import cv2
import random
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
from scipy import misc
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

def itensity_normalize(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized n                                                                                                                                                                 d volume
    """

    # pixels = volume[volume > 0]
    mean = volume.mean()
    std = volume.std()
    out = (volume - mean) / std
    # out_random = np.random.normal(0, 1, size=volume.shape)
    # out[volume == 0] = out_random[volume == 0]

    return out

class CmrsegDataset(Dataset):
    def __init__(self, data_list_dir='./Datasets/MS_CMRSeg', data_dir='./Data/MSCMR_C0_45', 
                 train_type='train', image_type='image', percent=1.0, transform=None):
        self.transform = transform
        self.train_type = train_type
        self.data_list_dir = data_list_dir
        self.data_dir = data_dir
        self.image_type = image_type

        if self.train_type in ['train', 'valid', 'test']:
            # this is for cross validation
            with open(join(self.data_list_dir, self.train_type+'_list'),
                      'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            if percent < 1 and percent > 0:
                random.shuffle(self.image_list)
                self.image_list = self.image_list[:round(len(self.image_list)*percent)]
            self.data = [join(self.data_dir, self.image_type, x+'.png') for x in self.image_list]
            self.mask = [join(self.data_dir, 'label', x+'.png') for x in self.image_list]
            print("totoal {} {} samples".format(len(self.data), self.train_type))
        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")

        assert len(self.data) == len(self.mask)

    def __getitem__(self, item: int):
        '''
        scipy.misc reads the '.jpg' images, the read data's format is HxWxC
        '''
        slice_name = self.data[item].rsplit('/', maxsplit=1)[-1].split('.')[0]
        image = cv2.imread(self.data[item], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        image = itensity_normalize(image)
        label = cv2.imread(self.mask[item], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
        label[label == 128] = 1
        label[label == 255] = 2
        assert (label.any() > 2) == False 

        sample = {'slice_name': slice_name, 'image': image, 'label': label}

        if self.transform is not None:
            # TODO: transformation to argument datasets
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.data)


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'slice_name': sample['slice_name'], 'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'slice_name': sample['slice_name'], 'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image, label = image.transpose([2, 0, 1]), label.transpose([2, 0, 1])
        image, label = image.astype(np.float32), label.astype(np.float32)
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'slice_name': sample['slice_name'], 'image': torch.from_numpy(image),
                    'label': torch.from_numpy(label), 'onehot_label': torch.from_numpy(sample['onehot_label'])}
        else:
            return {'slice_name': sample['slice_name'], 'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)