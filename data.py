import os
from statistics import mode
from PIL import Image

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

import pandas as pd

import platform
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import albumentations as albu
import sys

class NiiDataset():
    def __init__(self, img_path, msk_path, MRI_series='T1', mode='train', resize=None):
        self.img_path = img_path
        self.msk_path = msk_path
        self.MRI_series = MRI_series
        self.mode = mode
        self.resize = resize

        if self.mode == 'train':
            img_list = os.listdir(self.img_path)
            img_list = [img for img in img_list if img <='W5']
        elif self.mode == 'test':
            img_list = os.listdir(self.img_path)
            img_list = [img for img in img_list if img > 'W5']
        else:
            print('Wrong dataset')
            sys.exit(0)

        img_list = [img for img in img_list if MRI_series in img]

        self.img_list = img_list


    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.img_path, self.img_list[index])
        mask_path = os.path.join(self.msk_path, self.img_list[index].split('_')[0] + '.nii.gz')

        img = read_nii_image(image_path)
        msk = read_nii_image(mask_path)
        msk = gray2Binary(msk)
        if self.resize is not None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
            msk = cv2.resize(msk, self.resize, interpolation=cv2.INTER_NEAREST)

        return torch.tensor(img / sys.float_info.max), torch.tensor(msk / 255)

if __name__ == '__main__':
    IMGDataset = NiiDataset('./data/Images', './data/Masks', MRI_series='T1', mode='train', resize=None)
    print(IMGDataset[5])
    plt.imshow(IMGDataset[5][0][:,:,100])
    plt.show()

    plt.imshow(read_nii_image(os.path.join(IMGDataset.img_path, IMGDataset.img_list[5]))[:,:,100])
    plt.show()

    print(IMGDataset[5][0].dtype)
    print(IMGDataset[5][1].dtype)