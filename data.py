import os
from statistics import mode
from PIL import Image
from cv2 import resize

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
from scipy.ndimage import interpolation as itpl


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
            img = resize_3d_image(img, self.resize)
            original_msk = np.copy(msk)
            msk = resize_3d_image(msk, self.resize, mode='nearest')

        #msk = msk / 255
        #msk[msk > 0.9] = 1
        #msk[msk <= 0.9] = 0

        if self.resize is not None:
            return torch.FloatTensor(img / img.max()).unsqueeze(0), torch.FloatTensor(msk / 255).unsqueeze(0), torch.FloatTensor(original_msk / 255).unsqueeze(0)
        else:
            return torch.FloatTensor(img / img.max()).unsqueeze(0), torch.FloatTensor(msk / 255).unsqueeze(0)

if __name__ == '__main__':
    #print(sys.float_info.max)
    IMGDataset = NiiDataset('./data/Images', './data/Masks', MRI_series='T1', mode='train', resize=(256, 256, 128))
    #print(IMGDataset[5][0].shape)
    plt.imshow(resize_3d_image(IMGDataset[6][1], IMGDataset[6][2].shape, mode='nearest')[:,:,70])
    plt.show()

    plt.imshow(IMGDataset[6][2][:,:,70])
    plt.show()


    #TestDataset = NiiDataset('./data/Images', './data/Masks', MRI_series='T1', mode='train')
    #plt.imshow(TestDataset[5][1][:,:,90])
    #plt.show()

    #plt.imshow(IMGDataset[5][1][:,:,70])
    #plt.show()