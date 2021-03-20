import os
import time
import argparse

from typing import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from torchvision import datasets, transforms

from torch.optim import lr_scheduler
from torch import tensor
from tqdm import tqdm

from utils import *
from learner import *
from data import *
from models import *
from criterion import *
from config import *

from dataloaders.nyu import NYUDataset

def get_subset_databunch(datapath, config):
    train_img_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_depth_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    valid_img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_depth_transforsms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = H5Dataset(datapath,
                         img_transforms=train_img_transforms,
                         depth_transforms=train_depth_transforms)

    valid_ds = H5Dataset(datapath,
                         img_transforms=valid_img_transforms,
                         depth_transforms=valid_depth_transforsms, valid=True)

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=config.batch_size,
        shuffle=True,
    )

    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )
    data = DataBunch(train_dl, valid_dl)
    return data

def get_full_databunch(datapath, config):
    train_ds = NYUDataset(datapath, split='train', modality='rgb')

    valid_ds = NYUDataset(datapath, split='val', modality='rgb')

    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=config.batch_size,
        shuffle=True,
    )

    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )
    data = DataBunch(train_dl, valid_dl)
    return data


def get_databunch(datapath, config, is_subset=False):
    return get_subset_databunch(datapath, config) if is_subset else get_full_databunch(datapath, config)


def get_optimizer_function(config):
    def optimizer(params, lr):
        if config.optimizer == SGD:
            opt = torch.optim.SGD(filter(lambda p: p.requires_grad, params),
                            lr, momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr, weight_decay=config.weight_decay)
        return opt
    return optimizer



class StepCallback(Callback):
    def begin_fit(self):
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10, gamma=0.9, 
                                                            last_epoch=-1, verbose=True)
    def after_epoch(self):
        self.lr_scheduler.step()


if __name__ == '__main__':
    config = Config()
    config.write()
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    datapath = config.subset_datapath if config.train_subset else config.datapath
    data = get_databunch(datapath, config, config.train_subset)

    if config.loss_function == SMOOTH_L1:
        loss_fn = SmoothedL1Loss(preserve_edges=True)
    elif config.loss_function == L1:
        loss_fn = L1Loss(preserve_edges=True)
    else:
        loss_fn = L2Loss(preserve_edges=True)

    if config.arch == MOBILENET_V2:
        model = MobileNetV2SkipAdd(pretrained=True, interpolation='bilinear')
    else:
        model = MobileNetSkipAdd()


    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter('runs')

    # Set up checkpoint dir
    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)
    model_prefix = str(Path(config.checkpoint_dir)/config.modelname_prefix)

    learner = UnifiedLearner(model, data, loss_fn, opt_func=get_optimizer_function(config),
                             lr=config.lr, cbs=[Recorder(), AvgStatsCallback([delta1], tb),
                                                CudaCallback('cuda'), StepCallback(),
                                                ModelSavingCallback(model_prefix),
                                                DisplayResultsCallback()],
                             iter_wrapper=tqdm)
    
    learner.fit(config.epochs)
    
    
