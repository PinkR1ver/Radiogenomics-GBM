import os
from PIL import Image
from cv2 import selectROIs

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

import pandas as pd

transform = transforms.Compose([
    transforms.ToTensor()
])


class ImageDataSet(Dataset):
    def __init__(self, path, dataset_file):
        self.path = path
        self.Info = pd.read_csv(os.path.join(self.path, dataset_file))
        self.AxInfo = ((self.Info).loc[(self.Info)['Plane'] == 'AX'])


    def __len__(self):
        return len(self.AxInfo)

    def __getitem__(self, index):
        maskPath = os.path.join(self.path, ((self.AxInfo).iloc[index]).MaskPath)
        imagePath = os.path.join(self.path, ((self.AxInfo).iloc[index]).ImagePath)
        image = keep_image_size_open_gray(imagePath)
        mask = keep_image_size_open_gray(maskPath)
        mask = gray2Binary(mask)
        return transform(image), transform(mask)



def image_location_transfer(rootdir):
    for root, dirs, files in os.walk(os.path.join(rootdir)):
        for file in files:
            imageSavePath = r'C:\Users\83549\Github Projects\Radiogenemics\Radiogenemics--on-Ivy-Gap\data\Images'
            maskSavePath = r'C:\Users\83549\Github Projects\Radiogenemics\Radiogenemics--on-Ivy-Gap\data\Masks'
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

def built_dataset_csv(path):
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
    image_location_transfer(r'C:\Users\83549\OneDrive\Documents\Research Data\Multi-institutional Paired Expert Segmentations MNI images-atlas-annotations')
    # built_dataset_csv(r'C:\Users\RTX 3090\Desktop\WangYichong\U-net for Ivy Gap\data')
    # GBMDataset = ImageDataSet(r'C:\Users\RTX 3090\Desktop\WangYichong\U-net for Ivy Gap\data', 'GBM_MRI_Dataset.csv')
    # print(GBMDatase
    # t.AxInfo)
    # print(len(GBMDataset))
    # print(GBMDataset[5])
    # pass