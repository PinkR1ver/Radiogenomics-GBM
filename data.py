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
        self.Info = self.Info[(self.Info)['MRISeries'] == self.MRI_series]
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

class FeatureExtractionDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        self.AXInfo = self.Info[self.Info['Plane'] == 'AX']
    
    def __len__(self):
        return len(self.AXInfo)
    
    def __getitem__(self, index):
        return pd.DataFrame([self.AXInfo.iloc[index].values.flatten().tolist()], columns=self.AXInfo.columns)


def image_location_transfer(rootdir):
    for root, dirs, files in os.walk(os.path.join(rootdir)):
        for file in files:
            imageSavePath = os.path.join('data', 'Images')
            maskSavePath = os.path.join('data', 'Masks')
            if "1_Images_MNI" in root and '.png' in file:
                imagePath = os.path.join(root, file)
                print(f'Image:{imagePath}')

                dirTmp = root.split('\\')[-1]
                dirTmp = dirTmp.split('_')
                dirNeed = dirTmp[0] + '_' + dirTmp[-2] + '_' + dirTmp[-1]

                fileNeed = dirTmp[0] + '_' + dirTmp[-2] + '_' + dirTmp[-1] + '_' + file

                # print(dirNeed)
                # print(fileNeed)

                if not os.path.isdir(os.path.join(imageSavePath, dirNeed)):
                    os.mkdir(os.path.join(imageSavePath, dirNeed))


                image = Image.open(imagePath)
                imageSavePath = os.path.join(imageSavePath, dirNeed, fileNeed)
                image.save(imageSavePath)
            elif "3_Annotations_MNI" in root and '.png' in file:
                maskPath = os.path.join(root, file)
                print(f'Mask:{maskPath}')

                dirTmp = root.split('\\')[-1]
                dirTmp = dirTmp.split('_')
                dirNeed = dirTmp[0] + '_' + dirTmp[-1] 

                fileNeed = dirTmp[0] + '_' + dirTmp[-1] + '_' + file

                if not os.path.isdir(os.path.join(maskSavePath, dirNeed)):
                    os.mkdir(os.path.join(maskSavePath, dirNeed))

                mask = Image.open(maskPath)
                maskSavePath = os.path.join(maskSavePath, dirNeed, fileNeed)
                mask.save(maskSavePath)

def built_Dataset_csv(path):
    ImageDatasetTable = pd.DataFrame(
        {
            "Patient":[],
            "Plane":[],
            "MRISeries":[],
            "Slice":[],
            "ImagePath":[],
            "MaskPath":[]
        }
    )
    for root, dirs, files in os.walk(path):
        for file in files:
            if "Images" in root and ".png" in file:
                ImagePathTmp = root.split('\\')
                ImagePath = ImagePathTmp[-2] + '\\' + ImagePathTmp[-1] + '\\' + file
                Info = file.split('_')
                Patient = Info[0]
                MRISeries = Info[1]
                Plane = Info[2]
                Slice = Info[3].replace('.png','')
                MaskPath = ImagePath.replace('Images','Masks')
                MaskPath = MaskPath.replace('_' + MRISeries, '')
                ImageDatasetTableRow = pd.DataFrame([[Patient, Plane, MRISeries, Slice, ImagePath, MaskPath]], columns=ImageDatasetTable.columns)
                # print(ImageDatasetTable)
                ImageDatasetTable = ImageDatasetTable.append(ImageDatasetTableRow, ignore_index=True)
    
    ImageDatasetTable.to_csv(os.path.join(path,'GBM_MRI_Dataset.csv'), index=False)



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