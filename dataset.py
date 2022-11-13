import os
import sys
import argparse



import numpy as np
import glob
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset


class Sun_dataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        self.data_path = data_path
        self.img_path_list = glob.glob(os.path.join(data_path, '*', '*.jpg'))
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = cv2.imread(img_path)
        target_image = image

    def img_processing(img):
        pass

