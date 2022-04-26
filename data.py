import os
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
        if MRI_series == 'Stack':
            FLAIR_Info = self.Info[self.Info['MRISeries'] == 'FLAIR']
            FLAIR_Info = FLAIR_Info.reset_index(drop=True)
            FLAIR_Info = FLAIR_Info.rename(columns={'MRISeries':'MRISeries3', 'ImagePath':'ImagePath3'})

            T2_Info = self.Info[self.Info['MRISeries'] == 'T2']
            T2_Info = T2_Info.reset_index(drop=True)
            T2_Info = T2_Info.rename(columns={'MRISeries':'MRISeries2', 'ImagePath':'ImagePath2'})

            T1_Info = self.Info[self.Info['MRISeries'] == 'T1']
            T1_Info = T1_Info.reset_index(drop=True)

            self.Info = FLAIR_Info.merge(T2_Info.merge(T1_Info))

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
        if self.MRI_series != 'Stack':
            if platform.system() == 'Windows':
                maskPath = os.path.join(self.path, (((self.Info).iloc[index]).MaskPath))
                imagePath = os.path.join(self.path, (((self.Info).iloc[index]).ImagePath))
            elif platform.system() == 'Linux' or platform.system() == 'Darwin':
                maskPath = os.path.join(self.path, ((((self.Info).iloc[index]).MaskPath).replace('\\', '/')))
                imagePath = os.path.join(self.path, ((((self.Info).iloc[index]).ImagePath).replace('\\', '/')))
        else:
            if platform.system() == 'Windows':
                maskPath = os.path.join(self.path, (((self.Info).iloc[index]).MaskPath))
                T1_imagePath = os.path.join(self.path, (((self.Info).iloc[index]).ImagePath))
                T2_imagePath = os.path.join(self.path, (((self.Info).iloc[index]).ImagePath2))
                FLAIR_imagePath = os.path.join(self.path, (((self.Info).iloc[index]).ImagePath3))
            elif platform.system() == 'Linux' or platform.system() == 'Darwin':
                maskPath = os.path.join(self.path, ((((self.Info).iloc[index]).MaskPath).replace('\\', '/')))
                T1_imagePath = os.path.join(self.path, ((((self.Info).iloc[index]).ImagePath).replace('\\', '/')))
                T2_imagePath = os.path.join(self.path, ((((self.Info).iloc[index]).ImagePath2).replace('\\', '/')))
                FLAIR_imagePath = os.path.join(self.path, ((((self.Info).iloc[index]).ImagePath3).replace('\\', '/')))

        
        if self.MRI_series != 'Stack':
            img = cv2.imread(imagePath, 0)
        else:
            T1_image = cv2.imread(T1_imagePath, 0)
            T2_image = cv2.imread(T2_imagePath, 0)
            FLAIR_image = cv2.imread(FLAIR_imagePath, 0)
            img = np.stack((T1_image, T2_image, FLAIR_image), axis=-1)

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



if __name__ == '__main__':
    IMGDataset = ImageDataset('./data', 'GBM_MRI_Dataset.csv', n_classes=2, label_value=[0, 255], augmentation=True, canny_edge=True)
    #plt.imshow(IMGDataset[20][1])
    #plt.show()

    print(IMGDataset.class_weights())

    print(IMGDataset[20][0].size())
    print(IMGDataset[20][1].size())
    print(IMGDataset[20][1].max())

    plt.imshow(IMGDataset[20][1].squeeze(0))
    plt.show()

    plt.imshow(IMGDataset[20][3].squeeze(0))
    plt.show()

    plt.imshow(IMGDataset[20][1].squeeze(0) - IMGDataset[20][3].squeeze(0))
    plt.show()

    plt.imshow(IMGDataset[20][0].squeeze(0))
    plt.show()

    plt.imshow(IMGDataset[20][2].squeeze(0))
    plt.show()

    plt.imshow(IMGDataset[20][2].squeeze(0) - IMGDataset[20][0].squeeze(0))
    plt.show()

    plt.imshow(IMGDataset[20][4].squeeze(0))
    plt.show()