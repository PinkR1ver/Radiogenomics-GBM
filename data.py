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


transform = transforms.Compose([
    transforms.ToTensor()
])

class ImageDataset(Dataset):
    def __init__(self, path, Dataset_file, axis='AX', MRI_series='T1', n_classes=2, label_value=[0,255], mode='train', resize=None, augmentation=False, canny_edge=False):
        self.path = path
        self.MRI_series = MRI_series
        self.axis = axis
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        self.Info = self.Info[self.Info['Plane'] == self.axis]
        if not MRI_series == 'ALL':
            self.Info = self.Info[self.Info['MRISeries'] == self.MRI_series]
        self.Info = self.Info[self.Info['Slice'] > 20]
        self.Info = self.Info[self.Info['Slice'] < 150]
        if mode == 'train':
            self.Info = self.Info[self.Info['Patient'] <= 'W5']
        elif mode == 'test':
            self.Info = self.Info[self.Info['Patient'] > 'W5']
        self.Info = self.Info.reset_index(drop=True)
        self.n_classes = n_classes
        self.label_value = label_value
        self.mode = mode
        self.augmentation = augmentation
        self.resize = resize
        self.canny_edge = canny_edge
        if self.augmentation:
            self.augs = albu.OneOf([albu.ElasticTransform(p=1.0, alpha=120, sigma=280 * 0.05, alpha_affine=120 * 0.03),
                                albu.GridDistortion(p=1.0, border_mode=cv2.BORDER_CONSTANT, distort_limit=0.2),
                                albu.Rotate(p=1.0, limit=(-5, 5), interpolation=1, border_mode=cv2.BORDER_CONSTANT),
                                ],p=1.0)
    def __len__(self):
        return len(self.Info)

    def class_weights(self):
        counts = defaultdict(lambda : 0)
        for i in range(len(self)):
            if platform.system() == 'Windows':
                maskPath = os.path.join(self.path, (((self.Info).iloc[i]).MaskPath))
            elif platform.system() == 'Linux' or platform.system() == 'Darwin':
                maskPath = os.path.join(self.path, ((((self.Info).iloc[i]).MaskPath).replace('\\', '/')))
            msk = cv2.imread(maskPath, -1)
            for c in range(self.n_classes):
                counts[c] += np.sum(msk == self.label_value[c])

        counts = dict(sorted(counts.items()))
        weights = [1 - (x/sum(list(counts.values()))) for x in counts.values()]

        return torch.FloatTensor(weights)
    
    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.Info).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.Info).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.Info).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.Info).iloc[index]).ImagePath).replace('\\', '/')))

        img = cv2.imread(imagePath, 0)
        msk = cv2.imread(maskPath, 0)
        msk = gray2Binary(msk)
        if self.resize is not None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
            msk = cv2.resize(msk, self.resize, interpolation=cv2.INTER_NEAREST)

        if self.augmentation:
            augmented = self.augs(image=img, mask=msk)
            img_aug, msk_aug = augmented['image'], augmented['mask']


        if self.canny_edge:
            canny = cv2.Canny(img, 10, 100)
            canny = np.asarray(canny, np.float32)
            canny /= 255.0

        if self.canny_edge:
            #return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(canny).unsqueeze(0), torch.LongTensor(msk), torch.FloatTensor(canny)
            if self.augmentation:
                return transform(img), transform(msk), transform(img_aug), transform(msk_aug),transform(canny)
            else:
                return transform(img), transform(msk), transform(canny)
        
        elif self.augmentation:
            return transform(img), transform(msk), transform(img_aug), transform(msk_aug)
        else:
            return transform(img), transform(msk)

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
        print(img_list)

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

        return transform(img), transform(msk)


if __name__ == '__main__':
    IMGDataset = NiiDataset('./data/Images', './data/Masks', MRI_series='T1', mode='train', resize=None)
    print(IMGDataset[5])
    plt.imshow(IMGDataset[5][1][100,:,:].cpu().detach().numpy())
    plt.show()

    plt.imshow(read_nii_image(os.path.join(IMGDataset.img_path, IMGDataset.img_list[5]))[:,:,100])
    plt.show()