import os
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import torchvision
from data import *
from net import *
import numpy as np
import traceback
import trainHelper
import sys

MRI_series_this = sys.argv[1]
epoches = int(sys.argv[2])

basePath = os.path.dirname(__file__)
dataPath = os.path.join(basePath, 'data')
modelPath = os.path.join(basePath, 'model', MRI_series_this)
weightPath = os.path.join(modelPath, f'{MRI_series_this}_unet.pth')
savePath = os.path.join(dataPath, 'train_monitor_image_AX')
predictPath = os.path.join(dataPath, 'test_image_AX')

if not os.path.isdir(modelPath):
    os.mkdir(modelPath)


if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

def model_train():
