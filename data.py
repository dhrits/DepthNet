import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import glob
import random
import torch
import numpy as np
import h5py

import os
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image

torch.manual_seed(13)
np.random.seed(13)
random.seed(13)


class H5Dataset(object):
    def __init__(self, dataroot, valid=False, img_transforms=None, depth_transforms=None):
        
        self.dataroot = dataroot
        self.valid = valid
        self.datafile = h5py.File(self.dataroot, mode='r')
        size = len(self.datafile['images'])
        self.indexes = np.arange(size)
        np.random.shuffle(self.indexes)
        self.train_indexes, self.valid_indexes = train_test_split(self.indexes, test_size=0.05, random_state=13)
        if self.valid:
            self.indexes = self.valid_indexes
        else:
            self.indexes = self.train_indexes
        self.img_transforms = img_transforms
        self.depth_transforms = depth_transforms

    def __len__(self):
        return self.indexes.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        lookup = self.indexes[idx]
        img = np.copy(self.datafile['images'][lookup])
        depth = np.copy(self.datafile['depths'][lookup])
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)
        depth = Image.fromarray(depth)
        img = img.transpose(Image.ROTATE_270)
        depth = depth.transpose(Image.ROTATE_270)
        
        if self.img_transforms:
            img = self.img_transforms(img)
        if self.depth_transforms:    
            depth = self.depth_transforms(depth)
        return img, depth
    
