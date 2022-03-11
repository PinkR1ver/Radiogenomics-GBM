import os
from PIL import Image

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
import SimpleITK as sitk

import pandas as pd

import platform

import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import albumentations as albu

transform = transforms.Compose([
    transforms.ToTensor()
])


class Train_T1_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        T1_AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T1']
        T1_AXInfo = T1_AXInfo[T1_AXInfo['Patient'] < 'W5']
        self.T1_AXInfo = T1_AXInfo.reset_index(drop=True)

    def __len__(self):
        return len(self.T1_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.T1_AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.T1_AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.T1_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.T1_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Train_T2_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        T2_AXInfo = AXInfo[AXInfo['MRISeries'] == 'T2']
        T2_AXInfo = T2_AXInfo[T2_AXInfo['Patient'] < 'W5']
        self.T2_AXInfo = T2_AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.T2_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.T2_AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.T2_AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.T2_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.T2_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Train_FLAIR_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        FLAIR_AXInfo = AXInfo[AXInfo['MRISeries'] == 'FLAIR']
        FLAIR_AXInfo = FLAIR_AXInfo[FLAIR_AXInfo['Patient'] < 'W5']
        self.FLAIR_AXInfo = FLAIR_AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.FLAIR_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.FLAIR_AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.FLAIR_AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.FLAIR_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.FLAIR_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Train_Stack_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        FLAIR_AXInfo = AXInfo[AXInfo['MRISeries'] == 'FLAIR']
        FLAIR_AXInfo = FLAIR_AXInfo.reset_index(drop=True)
        FLAIR_AXInfo = FLAIR_AXInfo.rename(columns={'MRISeries':'MRISeries3', 'ImagePath':'ImagePath3'})
        
        T1_AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T1']
        T1_AXInfo = T1_AXInfo.reset_index(drop=True)
        
        T2_AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T2']
        T2_AXInfo = T2_AXInfo.reset_index(drop=True)
        T2_AXInfo = T2_AXInfo.rename(columns={'MRISeries':'MRISeries2', 'ImagePath':'ImagePath2'})

        Stack_AXInfo = T1_AXInfo.merge(T2_AXInfo.merge(FLAIR_AXInfo))
        Stack_AXInfo = Stack_AXInfo[Stack_AXInfo['Patient'] < 'W5']
        self.Stack_AXInfo = Stack_AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.Stack_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.Stack_AXInfo).iloc[index]).MaskPath))
            T1_imagePath = os.path.join(self.path, (((self.Stack_AXInfo).iloc[index]).ImagePath))
            T2_imagePath = os.path.join(self.path, (((self.Stack_AXInfo).iloc[index]).ImagePath2))
            FLAIR_imagePath = os.path.join(self.path, (((self.Stack_AXInfo).iloc[index]).ImagePath3))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.Stack_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            T1_imagePath = os.path.join(self.path, ((((self.Stack_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
            T2_imagePath = os.path.join(self.path, ((((self.Stack_AXInfo).iloc[index]).ImagePath2).replace('\\', '/')))
            FLAIR_imagePath = os.path.join(self.path, ((((self.Stack_AXInfo).iloc[index]).ImagePath3).replace('\\', '/')))
        T1_image = np.array(keep_image_size_open_gray(T1_imagePath))
        T2_image = np.array(keep_image_size_open_gray(T2_imagePath))
        FLAIR_image = np.array(keep_image_size_open_gray(FLAIR_imagePath))
        image = np.stack((T1_image, T2_image, FLAIR_image), axis=-1)
        # I don't know why i need to transpose numpy array before I create Image from array
        #print(image_stack.T.shape) (256, 256, 3)
        #print(image_stack.shape) (3, 256, 256)
        #image = Image.fromarray(image_stack.T, mode='RGB')
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Train_Messy_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        Messy_AXInfo = AXInfo[AXInfo['Patient'] < 'W5']
        self.Messy_AXInfo = Messy_AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.Messy_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.Messy_AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.Messy_AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.Messy_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.Messy_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Test_T1_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        T1_AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T1']
        T1_AXInfo = T1_AXInfo[T1_AXInfo['Patient'] >= 'W5']
        self.T1_AXInfo = T1_AXInfo.reset_index(drop=True)

    def __len__(self):
        return len(self.T1_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.T1_AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.T1_AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.T1_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.T1_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Test_T2_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        T2_AXInfo = AXInfo[AXInfo['MRISeries'] == 'T2']
        T2_AXInfo = T2_AXInfo[T2_AXInfo['Patient'] >= 'W5']
        self.T2_AXInfo = T2_AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.T2_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.T2_AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.T2_AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.T2_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.T2_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Test_FLAIR_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        FLAIR_AXInfo = AXInfo[AXInfo['MRISeries'] == 'FLAIR']
        FLAIR_AXInfo = FLAIR_AXInfo[FLAIR_AXInfo['Patient'] >= 'W5']
        self.FLAIR_AXInfo = FLAIR_AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.FLAIR_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.FLAIR_AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.FLAIR_AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.FLAIR_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.FLAIR_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Test_Stack_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        FLAIR_AXInfo = AXInfo[AXInfo['MRISeries'] == 'FLAIR']
        FLAIR_AXInfo = FLAIR_AXInfo.reset_index(drop=True)
        FLAIR_AXInfo = FLAIR_AXInfo.rename(columns={'MRISeries':'MRISeries3', 'ImagePath':'ImagePath3'})
        
        T1_AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T1']
        T1_AXInfo = T1_AXInfo.reset_index(drop=True)
        
        T2_AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T2']
        T2_AXInfo = T2_AXInfo.reset_index(drop=True)
        T2_AXInfo = T2_AXInfo.rename(columns={'MRISeries':'MRISeries2', 'ImagePath':'ImagePath2'})

        Stack_AXInfo = T1_AXInfo.merge(T2_AXInfo.merge(FLAIR_AXInfo))
        Stack_AXInfo = Stack_AXInfo[Stack_AXInfo['Patient'] >= 'W5']
        self.Stack_AXInfo = Stack_AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.Stack_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.Stack_AXInfo).iloc[index]).MaskPath))
            T1_imagePath = os.path.join(self.path, (((self.Stack_AXInfo).iloc[index]).ImagePath))
            T2_imagePath = os.path.join(self.path, (((self.Stack_AXInfo).iloc[index]).ImagePath2))
            FLAIR_imagePath = os.path.join(self.path, (((self.Stack_AXInfo).iloc[index]).ImagePath3))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.Stack_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            T1_imagePath = os.path.join(self.path, ((((self.Stack_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
            T2_imagePath = os.path.join(self.path, ((((self.Stack_AXInfo).iloc[index]).ImagePath2).replace('\\', '/')))
            FLAIR_imagePath = os.path.join(self.path, ((((self.Stack_AXInfo).iloc[index]).ImagePath3).replace('\\', '/')))
        T1_image = np.array(keep_image_size_open_gray(T1_imagePath))
        T2_image = np.array(keep_image_size_open_gray(T2_imagePath))
        FLAIR_image = np.array(keep_image_size_open_gray(FLAIR_imagePath))
        image = np.stack((T1_image, T2_image, FLAIR_image), axis=-1)
        # I don't know why i need to transpose numpy array before I create Image from array
        #print(image_stack.T.shape) (256, 256, 3)
        #print(image_stack.shape) (3, 256, 256)
        #image = Image.fromarray(image_stack.T, mode='RGB')
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Test_Messy_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        Messy_AXInfo = AXInfo[AXInfo['Patient'] >= 'W5']
        self.Messy_AXInfo = Messy_AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.Messy_AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.Messy_AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.Messy_AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.Messy_AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.Messy_AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

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

# New version dataset, you can choose MRI axis and MRI series and can do resize or not, also support data augmentation
class MSRf_Net_Dataset(Dataset):
    def __init__(self, path, Dataset_file, axis, MRI_series, n_classes, mode='train', augmentation=True, resize=None):
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
        self.mode = mode
        self.augmentation = augmentation
        self.resize = resize
        if self.augmentation:
            self.augs = albu.OneOf([albu.ElasticTransform(p=0.5, alpha=120, sigma=280 * 0.05, alpha_affine=120 * 0.03),
                                albu.GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT, distort_limit=0.2),
                                albu.Rotate(p=0.5, limit=(-5, 5), interpolation=1, border_mode=cv2.BORDER_CONSTANT),
                                ],)
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
                counts[c] += np.sum(msk == c)

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
        if self.resize is not None:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
            msk = cv2.resize(msk, self.resize, interpolation=cv2.INTER_NEAREST)

        if self.augmentation:
            augmented = self.augs(image=img, mask=msk)
            img = augmented['image']
            msk = augmented['mask']
        
        canny = cv2.Canny(img, 10, 100)
        canny = np.asarray(canny, np.float32)
        canny /= 255.0

        return torch.FloatTensor(img).unsqueeze(0), torch.FloatTensor(canny).unsqueeze(0), torch.LongTensor(msk), torch.FloatTensor(canny)



if __name__ == '__main__':
    exp = MSRf_Net_Dataset('data', 'GBM_MRI_Dataset.csv', 'AX', 'T1', 4, 'train', True, (256, 256))
    img, canny, msk, canny_label = exp[20]
    plt.imshow(img.detach().numpy()[0])
    plt.show()
    plt.imshow(canny.detach().numpy()[0])
    plt.show()
    plt.imshow(msk.detach().numpy())
    plt.show()
    plt.imshow(canny_label.detach().numpy())
    plt.show()
    print(exp.class_weights())