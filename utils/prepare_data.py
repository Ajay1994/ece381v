import argparse
import os
import random
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.dataloader import *

def get_data_models(args):
    
    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    PATH_TO_IMAGES = "/data2/ajay_data/lesion_data/2019/"
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    # create train/val/test dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = LesionDataLoader(
        path_to_images=PATH_TO_IMAGES,
        mode='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = LesionDataLoader(
        path_to_images=PATH_TO_IMAGES,
        mode='val',
        transform=data_transforms['val'])
    transformed_datasets['test'] = LesionDataLoader(
        path_to_images=PATH_TO_IMAGES,
        mode='test',
        transform=data_transforms['val'])
    
    print('Total image in train: ', len(transformed_datasets['train']))
    print('Total image in valid: ', len(transformed_datasets['val']))
    print('Total image in test: ', len(transformed_datasets['test']))
    
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=8)
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=args.test_batch,
        shuffle=True,
        num_workers=8)
    dataloaders['test'] = torch.utils.data.DataLoader(
        transformed_datasets['test'],
        batch_size=args.test_batch,
        shuffle=True,
        num_workers=8)
    
    return dataloaders