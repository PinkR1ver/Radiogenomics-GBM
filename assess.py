import os
from data import *
from utils import *
from net import *
import os
import torch
from torch.utils.data import DataLoader, sampler
import torchvision
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from matplotlib import pyplot as plt
import gc

if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

basePath = r''
dataPath = os.path.join(basePath, 'data')
weightPath = os.path.join(basePath, r'model/unet.pth')
assessmentPath = os.path.join(dataPath, 'assessment')
cmatPath = os.path.join(assessmentPath, 'confusion_matrix')
predictMaskPath = os.path.join(assessmentPath, r'predict_mask')

batch_size = 1
sensitivity_all = 0
specificity_all = 0
iters = 0
cmat_all = [
    [0, 0],
    [0, 0]
]

if __name__ == '__main__':
    PredictDataset = ImageDataSet(dataPath, 'GBM_MRI_Dataset.csv')
    PredictDataLoader = DataLoader(
        PredictDataset, batch_size=batch_size, shuffle=False)

    f = open(os.path.join(assessmentPath, 'log.txt'), "w")
    f.close()

    net = UNet().to(device)
    if os.path.exists(weightPath):
        if device == 'cpu':
            net.load_state_dict(torch.load(
                weightPath, map_location=torch.device('cpu')))
            print("Loading Weight Successful")
        else:
            net.load_state_dict(torch.load(weightPath))
            print("Loading Weight Successful")
    else:
        print("Loading Weight Failed")

    for i, (image, mask) in enumerate(PredictDataLoader):
        image, mask = image.to(device), mask.to(device)

        predictMask = net(image)

        for j in range(batch_size):
            predictMask_arr = torch.squeeze(
                predictMask[j].cpu()).detach().numpy()
            predictMask_arr[predictMask_arr > 0.5] = 1
            predictMask_arr[predictMask_arr <= 0.5] = 0
            mask_arr = torch.squeeze(mask[j].cpu()).detach().numpy()
            #torchvision.utils.save_image(mask[j], os.path.join(predictMaskPath, f'{i}_{j}.png'))

            if np.any(predictMask_arr) or np.any(mask_arr):
                # print(predictMask.sum())

                FP = len(np.where(predictMask_arr - mask_arr == -1)[0])
                FN = len(np.where(predictMask_arr - mask_arr == 1)[0])
                TP = len(np.where(predictMask_arr + mask_arr == 2)[0])
                TN = len(np.where(predictMask_arr + mask_arr == 0)[0])
                cmat = [
                    [TN, FN],
                    [FP, TP]
                ]
                cmat_all += cmat

                # print(f'sensitivity:{TP/(TP+FN)}')
                # print(np.any(torch.squeeze(PredictDataset[i*batch_size + j][1]).detach().numpy() - mask_arr))

                # print(cmat)

                if TP + FN != 0:
                    sensitivity = TP / (TP + FN)
                else:
                    sensitivity = np.nan
                if TN + FP != 0:
                    specificity = FP / (TN + FP)
                else:
                    specificity = np.nan
                iters += 1
                sensitivity_all += sensitivity
                specificity_all += specificity

                fig_series = PredictDataset.AxInfo.iloc[i*batch_size + j]
                fig_name = fig_series.Patient + '_' + fig_series.MRISeries + \
                    '_' + fig_series.Plane + '_' + \
                    str(fig_series.Slice) + '.png'

                f = open(os.path.join(assessmentPath, 'log.txt'), "a")
                f.write(
                    f'{fig_name} sensitivity:{sensitivity}, specificity:{specificity}\n')
                f.close()

                fig = plt.figure(figsize=(6, 6))
                sns.heatmap(cmat / np.sum(cmat), cmap="Reds",
                            annot=True, fmt='.2%', square=1, linewidth=2.)
                plt.xlabel("predictions")
                plt.ylabel("real values")
                plt.savefig(os.path.join(cmatPath, fig_name))
                print(fig_name)
                plt.close(fig)
                gc.collect()

                _image = image[0]
                _segmentImage = mask[0]
                _outImage = predictMask[0]

                savePredictImage = torch.stack([_image, _segmentImage, _outImage], dim=0)
                torchvision.utils.save_image(savePredictImage, f'{predictMaskPath}\{fig_name}.png')
    
    fig = plt.figure(figsize=(6, 6))
    sns.heatmap(cmat_all / np.sum(cmat_all), cmap="Reds",
                annot=True, fmt='.2%', square=1, linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.savefig(os.path.join(cmatPath, 'ALL.png'))
    print('Done')
    plt.close(fig)

    f = open(os.path.join(assessmentPath, 'log.txt'), "a")
    f.write(
        f'sensitivity_mean:{sensitivity_all/iters}, specificity_mean:{specificity/iters}')
    f.close()
