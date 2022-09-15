#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 22:02
# @Author  : Ran.Gu
'''
Define a dataset class for REFUGE challenge (.png, .jpg) dataset
'''
import os
import torch
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
from scipy import misc
from torch.utils.data.dataset import Dataset

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

class RefugeDataset(Dataset):
    def __init__(self, data_list_dir='./Datasets/fundus', data_dir='./Data/REFUGE', 
                 train_type='train', image_type='image', transform=None):
        self.transform = transform
        self.train_type = train_type
        self.data_list_dir = data_list_dir
        self.data_dir = data_dir
        self.image_type = image_type

        if self.train_type in ['train', 'valid', 'test', 'data']:
            # this is for cross validation
            with open(join(self.data_list_dir, self.train_type+'_list'),
                      'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.data = [join(self.data_dir, 'Non-Glaucoma', self.image_type, x+'.png') for x in self.image_list]
            self.mask = [join(self.data_dir, 'Non-Glaucoma', 'label', x+'.png') for x in self.image_list]
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
        # plt.imshow(label)
        # plt.show()
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