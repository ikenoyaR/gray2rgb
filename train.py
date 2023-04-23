import os
import sys
import argparse


import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision


from dataset import Sun_dataset

def train(args):
    train_data = Sun_dataset(args.root, train=True, size=args.img_size)
    valid_data = Sun_dataset(args.root, train=True, size=args.img_size)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=args.pin_memory, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)