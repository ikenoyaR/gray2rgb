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
    def __init__(self, data_path, train = True, size=(224, 224), transform=None, target_transform=None):
        self.data_path = data_path
        self.train = train
        self.size = size
        if self.train:
            self.img_path_list = glob.glob(os.path.join(data_path, '*', '*', '*.jpg'))
        else:
            self.img_path_list = glob.glob(os.path.join(data_path, '*.jpg'))
        self.transform = transform
        self.target_transform = target_transform

    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        image = cv2.imread(img_path, 1)
        image = cv2.resize(image, dsize=(self.size[1], self.size[0])) # size = (h, w)
        if self.train:
            target_image = self.img_processing(image).transpose(2, 0, 1)
            gray_image = self.RGB2GRAY(target_image)
            target_image, gray_image = torch.tensor(target_image), torch.tensor(gray_image).unsqueeze()
            if self.transform:
                gray_image = self.transform(gray_image)
            if self.target_transform:
                target_image = self.target_transform(target_image)
            return gray_image, target_image

        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = torch.tensor(gray_image).unsqueeze()
            if self.transform:
                gray_image = self.transform(gray_image)
            return gray_image

    def img_processing(img): # Data augumentation will be added.
        img = img / 255
        return img[:, :, ::-1]

    def RGB2GRAY(img):
        if np.random.rand() >= 0.5:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            a, b = np.random.normal(loc=1.0, scale=0.05), np.random.normal(loc=1.0, scale=0.05)
            gray_img = 0.299 * a * img[:, :, 0] + 0.587 * b * img[:, :, 1] + (1-0.299*a-0.587*b) * img[:, :, 2]
            gray_img = gray_img.astype('np.uint8')
        return gray_img


def main(args):
    if args.mode == 'debug':
        dataset = Sun_dataset(args.source)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--mode', default='debug', choices=['train', 'test', 'predict', 'debug'], help='choose modes train or test or predict')
    args = parser.parse_args()
    main(args)
