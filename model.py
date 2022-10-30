import os
import sys
import argparse



from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        return x
