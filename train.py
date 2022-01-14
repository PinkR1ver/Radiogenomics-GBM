import os
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from data import *
from net import *
import numpy as np


if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

basePath = r'C:\Users\83549\Github Projects\Radiogenemics\Radiogenemics--on-Ivy-Gap'
dataPath = os.path.join(basePath, 'data')
weightPath = os.path.join(basePath, r'model\unet.pth')
savePath = os.path.join(basePath, r'data\train_monitor_image_AX')
predictPath = os.path.join(basePath, r'data\test_image_AX')

if __name__ == '__main__':
    fullDataSet = ImageDataSet(dataPath,'GBM_MRI_Dataset.csv')
    trainingDataSize = 0.8

    trainSize = int(trainingDataSize * len(fullDataSet))
    testSize = len(fullDataSet) - trainSize

    trainDataset, testDataset = torch.utils.data.random_split(fullDataSet, [trainSize, testSize])

    batchSize = 4

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    net = UNet().to(device)
    if os.path.exists(weightPath):
        net.load_state_dict(torch.load(weightPath))
        print("Loading Weight Successful")
    else:
        print("Loading Weight Failed")

    opt = optim.Adam(net.parameters())  # stochastic gradient descent
    lossFunction = nn.BCELoss()

    epoch = 1
    while True:

        for i, (image, segmentImage) in enumerate(trainLoader):
            image, segmentImage = image.to(device), segmentImage.to(device)

            outImage = net(image)
            trainLoss = lossFunction(outImage, segmentImage)

            opt.zero_grad()
            trainLoss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch}-{i}_train loss=====>>{trainLoss.item()}')

            if i % 50 == 0:
                torch.save(net.state_dict(), weightPath)

            _image = image[0]
            _segmentImage = segmentImage[0]
            _outImage = outImage[0]

            testImage = torch.stack([_image, _segmentImage, _outImage], dim=0)
            torchvision.utils.save_image(testImage, f'{savePath}\{i}.png')

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
                    torchvision.utils.save_image(testImage, f'{predictPath}\{i}.png')

            print("\n-------------------------------------------------------\n")