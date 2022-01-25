from cProfile import label
import os
from sys import path
from turtle import title
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from data import *
from net import *
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

basePath = r''
dataPath = os.path.join(basePath, 'data')
T1WeightPath = os.path.join(basePath, 'model', 'T1_unet.pth')
savePath = os.path.join(dataPath, 'train_monitor_image_AX')
predictPath = os.path.join(dataPath, 'test_image_AX')

if __name__ == '__main__':
    fullTrainDataset = Train_T1_AX_ImageDataset(dataPath,'GBM_MRI_Dataset.csv')
    trainingDataSize = 0.8

    trainSize = int(trainingDataSize * len(fullTrainDataset))
    testSize = len(fullTrainDataset) - trainSize

    trainDataset, testDataset = torch.utils.data.random_split(fullTrainDataset, [trainSize, testSize])

    batchSize = 1

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    net = UNet().to(device)
    if os.path.exists(T1WeightPath):
        net.load_state_dict(torch.load(T1WeightPath))
        print("Loading Weight Successful")
    else:
        print("Loading Weight Failed")

    opt = optim.Adam(net.parameters())  # stochastic gradient descent
    lossFunction = nn.BCELoss()

    epoch = 1
    trainLossList = np.array([])
    trainSensitivityList = np.array([])
    trainSpecificityList = np.array([])
    while True:

        if epoch % 2 == 0:
            trainLossList_x = len(trainLossList)
            fig = plt.figure(name="Train_Loss", figsize=(30, 30))
            plt.title(f'epoch{epoch-1} - epoch{epoch}: Train Loss')
            plt.xlabel('Train Times')
            plt.ylabel('Train Loss')
            plt.plot(trainLossList_x, trainLossList)
            plt.legend(title='train loss', loc='upper right', labels='train loss')
            plt.savefig(os.path.join(savePath, 'train_loss_monitor', f'epoch{epoch-1} - epoch{epoch}: Train Loss' + '.png'))

            trainSensitivityList_x = len(trainSensitivityList)
            fig = plt.figure(name="Train_Loss", figsize=(30, 30))
            plt.title(f'epoch{epoch-1} - epoch{epoch}: Sensitivity')
            plt.xlabel('Train Times')
            plt.ylabel('Sensitivity')
            plt.plot(trainSensitivityList_x, trainSensitivityList)
            plt.legend(title='sensitivity', loc='upper right', labels='sensitivity')
            plt.savefig(os.path.join(savePath, 'train_sensitivity_monitor', f'epoch{epoch-1} - epoch{epoch}: Sensitivity' + '.png'))

            trainSpecificityList_x = len(trainSpecificityList)
            fig = plt.figure(name="Train_Loss", figsize=(30, 30))
            plt.title(f'epoch{epoch-1} - epoch{epoch}: Specificity')
            plt.xlabel('Train Times')
            plt.ylabel('Specificity')
            plt.plot(trainSpecificityList_x, trainSpecificityList)
            plt.legend(title='specificity', loc='upper right', labels='specificty')
            plt.savefig(os.path.join(savePath, 'train_specificity_monitor', f'epoch{epoch-1} - epoch{epoch}: Specificity' + '.png'))



        for i, (image, segmentImage) in enumerate(trainLoader):
            image, segmentImage = image.to(device), segmentImage.to(device)

            outImage = net(image)
            trainLoss = lossFunction(outImage, segmentImage)
            trainLossList = np.append(trainLossList, trainLoss.cpu().detach().numpy())

            sensitivity = 0
            specificity = 0
            iters = 0

            for j in range(batchSize):
                
                predictMask_arr = torch.squeeze(outImage[j].cpu()).detach().numpy()
                predictMask_arr[predictMask_arr > 0.5] = 1
                predictMask_arr[predictMask_arr <= 0.5] = 0
                mask_arr = torch.squeeze(segmentImage[j].cpu()).detach().numpy()

                if np.any(predictMask_arr) and np.any(mask_arr):
                    FP = len(np.where(predictMask_arr - mask_arr == -1)[0])
                    FN = len(np.where(predictMask_arr - mask_arr == 1)[0])
                    TP = len(np.where(predictMask_arr + mask_arr == 2)[0])
                    TN = len(np.where(predictMask_arr + mask_arr == 0)[0])
                    cmat = np.array([[TN, FN],[FP, TP]])

                    sensitivity += TP / (TP + FN)
                    specificity += TN / (TN + FP)
                    iters += 1

            if sensitivity != 0 and specificity != 0:
                trainSensitivityList = np.append(trainSensitivityList, np.array(sensitivity/iters))
                trainSpecificityList = np.append(trainSpecificityList, np.array(specificity/iters))

            opt.zero_grad()
            trainLoss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch}-{i}_train loss=====>>{trainLoss.item()}')
                if sensitivity != 0 and specificity != 0:
                    print(f'{epoch}-{i}_sensitivity=====>>{sensitivity/iters}')
                    print(f'{epoch}-{i}_sensitivity=====>>{specificity/iters}')

            if i % 50 == 0:
                torch.save(net.state_dict(), T1WeightPath)

            _image = image[0]
            _segmentImage = segmentImage[0]
            _outImage = outImage[0]

            testImage = torch.stack([_image, _segmentImage, _outImage], dim=0)
            torchvision.utils.save_image(testImage, os.path.join(savePath, 'T1', f'{i}.png'))

        epoch += 1

        if epoch % 5 == 0:
            print("\n-------------------------------------------------------\n")
            with torch.no_grad():
                for i, (image, segmentImage) in enumerate(testLoader):
                    image, segmentImage = image.to(device), segmentImage.to(device)

                    outImage = net(image)
                    trainLoss = lossFunction(outImage, segmentImage)

                    _image = image[0]
                    _segmentImage = segmentImage[0]
                    _outImage = outImage[0]

                    print(f'{int(epoch / 5)}_{i}_test_loss=====>>{trainLoss.item()}')

                    testImage = torch.stack([_image, _segmentImage, _outImage], dim=0)
                    torchvision.utils.save_image(testImage, os.path.join(predictPath, 'T1', f'{i}.png'))

            print("\n-------------------------------------------------------\n")