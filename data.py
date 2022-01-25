import os
from PIL import Image
from cv2 import selectROIs

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
import SimpleITK as sitk

import pandas as pd

import platform

import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor()
])


class Train_T1_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        T1AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T1']
        T1AXInfo = T1AXInfo[T1AXInfo['Patient'] < 'W5']
        self.T1AXInfo = T1AXInfo.reset_index(drop=True)

    def __len__(self):
        return len(self.T1AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.T1AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.T1AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.T1AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.T1AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Train_T2_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        T2AXInfo = AXInfo[AXInfo['MRISeries'] == 'T2']
        T2AXInfo = T2AXInfo[T2AXInfo['Patient'] < 'W5']
        self.T2AXInfo = T2AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.T2AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.T2AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.T2AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.T2AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.T2AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
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
        
        T1AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T1']
        T1AXInfo = T1AXInfo.reset_index(drop=True)
        
        T2AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T2']
        T2AXInfo = T2AXInfo.reset_index(drop=True)
        T2AXInfo = T2AXInfo.rename(columns={'MRISeries':'MRISeries2', 'ImagePath':'ImagePath2'})

        Stack_AXInfo = T1AXInfo.merge(T2AXInfo.merge(FLAIR_AXInfo))
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

class Test_T1_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        T1AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T1']
        T1AXInfo = T1AXInfo[T1AXInfo['Patient'] >= 'W5']
        self.T1AXInfo = T1AXInfo.reset_index(drop=True)

    def __len__(self):
        return len(self.T1AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.T1AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.T1AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.T1AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.T1AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)

class Test_T2_AX_ImageDataset(Dataset):
    def __init__(self, path, Dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, Dataset_file))
        AXInfo = self.Info[self.Info['Plane'] == 'AX']
        T2AXInfo = AXInfo[AXInfo['MRISeries'] == 'T2']
        T2AXInfo = T2AXInfo[T2AXInfo['Patient'] >= 'W5']
        self.T2AXInfo = T2AXInfo.reset_index(drop=True)


    def __len__(self):
        return len(self.T2AXInfo)

    def __getitem__(self, index):
        if platform.system() == 'Windows':
            maskPath = os.path.join(self.path, (((self.T2AXInfo).iloc[index]).MaskPath))
            imagePath = os.path.join(self.path, (((self.T2AXInfo).iloc[index]).ImagePath))
        elif platform.system() == 'Linux' or platform.system() == 'Darwin':
            maskPath = os.path.join(self.path, ((((self.T2AXInfo).iloc[index]).MaskPath).replace('\\', '/')))
            imagePath = os.path.join(self.path, ((((self.T2AXInfo).iloc[index]).ImagePath).replace('\\', '/')))
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
        
        T1AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T1']
        T1AXInfo = T1AXInfo.reset_index(drop=True)
        
        T2AXInfo = AXInfo[(AXInfo)['MRISeries'] == 'T2']
        T2AXInfo = T2AXInfo.reset_index(drop=True)
        T2AXInfo = T2AXInfo.rename(columns={'MRISeries':'MRISeries2', 'ImagePath':'ImagePath2'})

        Stack_AXInfo = T1AXInfo.merge(T2AXInfo.merge(FLAIR_AXInfo))
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
    #image_location_transfer(r'C:\Users\83549\OneDrive\Documents\Research Data\Multi-institutional Paired Expert Segmentations MNI images-atlas-annotations')
    # built_Dataset_csv(r'C:\Users\RTX 3090\Desktop\WangYichong\U-net for Ivy Gap\data')
    GBMDataset = Train_T1_AX_ImageDataset(r'data', 'GBM_MRI_Dataset.csv')
    GBMDataset2 = Train_T2_AX_ImageDataset('data', 'GBM_MRI_Dataset.csv')
    GBMDataset3 = Train_FLAIR_AX_ImageDataset('data', 'GBM_MRI_Dataset.csv')

    print(len(GBMDataset))
    print(len(GBMDataset2))
    print(len(GBMDataset3))

    GBMDataset = Test_T1_AX_ImageDataset(r'data', 'GBM_MRI_Dataset.csv')
    GBMDataset2 = Test_T2_AX_ImageDataset('data', 'GBM_MRI_Dataset.csv')
    GBMDataset3 = Test_FLAIR_AX_ImageDataset('data', 'GBM_MRI_Dataset.csv')
    
    print(len(GBMDataset))
    print(len(GBMDataset2))
    print(len(GBMDataset3))


    GBMDataset4 = Train_Stack_AX_ImageDataset('data', 'GBM_MRI_Dataset.csv')
    print(len(GBMDataset4))
    print(GBMDataset4[3][0].shape)

    for i in range(len(GBMDataset4)):
        print(GBMDataset4.Stack_AXInfo.iloc[i])
    plt.imshow(GBMDataset4[3][0].detach().numpy()[1])
    plt.show()
    plt.imshow(GBMDataset4[3][1].detach().numpy()[0])
    plt.show()

    plt.imshow(GBMDataset[3][0].detach().numpy()[0])
    plt.show()
    plt.imshow(GBMDataset2[3][0].detach().numpy()[0])
    plt.show()
    plt.imshow(GBMDataset3[3][0].detach().numpy()[0])
    plt.show()
    
    # print(GBMDataset[5])
    # pass
    #FeatureDataset = FeatureExtractionDataset(r'/home/pinkr1ver/Documents/Github Projects/Radiogenemics--on-Ivy-Gap/data', 'GBM_MRI_Dataset.csv')
    #df = FeatureDataset[5631]
    #mask = sitk.ReadImage(os.path.join(r'/home/pinkr1ver/Documents/Github Projects/Radiogenemics--on-Ivy-Gap/data', (df['MaskPath'].loc[0]).replace('\\', '/')))
    #sitk.Show(mask)
    #df = df.to_frame()
    #df2 = FeatureDataset[8]
    #df2 = df2.to_frame()
    #df = df.append(df2)
    #print(df.values.flatten().tolist())
    #print(df)
    #print(df2)
    #df = df.append(df2, ignore_index=True)
    #print(df)