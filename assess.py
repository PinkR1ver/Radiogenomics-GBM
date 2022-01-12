import os

from cv2 import pencilSketch
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

basePath = r'C:\Users\83549\Github Projects\Radiogenemics\Radiogenemics--on-Ivy-Gap\data'
dataPath = os.path.join(basePath, 'data')
weightPath = os.path.join(basePath, r'model\unet.pth')
predictMaskPath = os.path.join(dataPath, r'predict_mask')
assessmentPath = os.path.join(dataPath, 'assessment')
cmatPath = os.path.join(assessmentPath, 'confusion_matrix')

batch_size = 1
sensitivity_all = 0
iters = 0

if __name__ == '__main__':
    PredictDataset = ImageDataSet(dataPath, 'GBM_MRI_Dataset.csv')
    PredictDataLoader = DataLoader(PredictDataset, batch_size=batch_size, shuffle=False)

    f = open(os.path.join(assessmentPath, 'log.txt'), "w")
    f.close()

    net = UNet().to(device)
    if os.path.exists(weightPath):
        if device == 'cpu':
            net.load_state_dict(torch.load(weightPath, map_location=torch.device('cpu')))
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
            predictMask_arr = torch.squeeze(predictMask[j]).detach().numpy()
            predictMask_arr[predictMask_arr > 0.5] = 1
            predictMask_arr[predictMask_arr <= 0.5] = 0
            mask_arr = torch.squeeze(mask[j]).detach().numpy()
            #torchvision.utils.save_image(mask[j], os.path.join(predictMaskPath, f'{i}_{j}.png'))

            if np.any(predictMask_arr):
                #print(predictMask.sum())

                FP = len(np.where(predictMask_arr - mask_arr == -1)[0])
                FN = len(np.where(predictMask_arr - mask_arr == 1)[0])
                TP = len(np.where(predictMask_arr + mask_arr == 2)[0])
                TN = len(np.where(predictMask_arr + mask_arr == 0)[0])
                cmat = [
                    [TN, FN],
                    [FP, TP]
                ]

                # print(f'sensitivity:{TP/(TP+FN)}')
                # print(np.any(torch.squeeze(PredictDataset[i*batch_size + j][1]).detach().numpy() - mask_arr))

                # print(cmat)

                sensitivity = TP / (TP+FN)
                iters +=1
                sensitivity_all += sensitivity


                fig_series = PredictDataset.AxInfo.iloc[i*batch_size + j]
                fig_name = fig_series.Patient + '_' + fig_series.MRISeries + '_' + fig_series.Plane + '_' + str(fig_series.Slice) +'.png'

                f = open(os.path.join(assessmentPath, 'log.txt'), "a")
                f.write(f'{fig_name} sensitivity:{sensitivity}\n')
                f.close()

                
                fig = plt.figure(figsize = (6, 6))
                sns.heatmap(cmat / np.sum(cmat), cmap = "Reds", annot = True, fmt = '.2%', square = 1, linewidth = 2.)
                plt.xlabel("predictions")
                plt.ylabel("real values")
                plt.savefig(os.path.join(cmatPath, fig_name))
                print(fig_name)
                plt.close(fig)
                gc.collect()
    
    f = open(os.path.join(assessmentPath, 'log.txt'), "a")
    f.write(f'sensitivity_mean:{sensitivity_all/iters}\n')
    f.close()