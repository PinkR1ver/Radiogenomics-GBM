import os
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import torchvision
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
modelPath = os.path.join(basePath, 'model')
FLAIRWeightPath = os.path.join(modelPath, 'FLAIR_unet.pth')
savePath = os.path.join(dataPath, 'train_monitor_image_AX')
predictPath = os.path.join(dataPath, 'test_image_AX')

if __name__ == '__main__':
    fullTrainDataset = Train_FLAIR_AX_ImageDataset(dataPath,'GBM_MRI_Dataset.csv')
    trainingDataSize = 0.8

    trainSize = int(trainingDataSize * len(fullTrainDataset))
    testSize = len(fullTrainDataset) - trainSize

    trainDataset, testDataset = torch.utils.data.random_split(fullTrainDataset, [trainSize, testSize])

    batchSize = 8

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    net = UNet().to(device)
    if os.path.exists(FLAIRWeightPath):
        net.load_state_dict(torch.load(FLAIRWeightPath))
        print("Loading Weight Successful")
    else:
        print("Loading Weight Failed")

    opt = optim.Adam(net.parameters())  # stochastic gradient descent
    lossFunction = nn.BCELoss()

    f = open(os.path.join(modelPath, 'FLAIR_epoch.txt'), "r") 
    epoch = int(f.read())
    f.close()


    trainLossList = np.array([])
    trainSensitivityList = np.array([])
    trainSpecificityList = np.array([])

    averageTrainLossList = np.array([])
    averageTrainSensitivityList = np.array([])
    averageTrainSpecificityList = np.array([])

    testLossList = np.array([])
    testSensitivityList = np.array([])
    testSpecificityList = np.array([])

    averageTestLossList = np.array([])
    averageTestSensitivityList = np.array([])
    averageTestSpecificityList = np.array([])

    for iter_out in range(300):

        for i, (image, segmentImage) in enumerate(trainLoader):
            image, segmentImage = image.to(device), segmentImage.to(device)

            outImage = net(image)
            trainLoss = lossFunction(outImage, segmentImage)
            trainLossList = np.append(trainLossList, trainLoss.item())

            sensitivity = 0
            specificity = 0
            iters = 0

            for j in range(outImage.size(dim=0)):
                
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

            if sensitivity != 0:
                trainSensitivityList = np.append(trainSensitivityList, np.array(sensitivity/iters))
            else:
                trainSensitivityList = np.append(trainSensitivityList, np.NaN)
            if specificity != 0:
                trainSpecificityList = np.append(trainSpecificityList, np.array(specificity/iters))
            else:
                trainSpecificityList = np.append(trainSpecificityList, np.NaN)

            opt.zero_grad()
            trainLoss.backward()
            opt.step()

            if i % 5 == 0:
                print(f'{epoch}-{i}_train loss=====>>{trainLoss.item()}')
                if sensitivity != 0:
                    print(f'{epoch}-{i}_sensitivity=====>>{sensitivity/iters}')
                else:
                    print(f'{epoch}-{i}_sensitivity=====>>{np.NaN}')
                if specificity != 0:
                    print(f'{epoch}-{i}_specificity=====>>{specificity/iters}')
                else:
                    print(f'{epoch}-{i}_specificity=====>>{np.NaN}')

            if i % 50 == 0:
                torch.save(net.state_dict(), FLAIRWeightPath)

            _image = image[0]
            _segmentImage = segmentImage[0]
            _outImage = outImage[0]

            testImage = torch.stack([_image, _segmentImage, _outImage], dim=0)
            torchvision.utils.save_image(testImage, os.path.join(savePath, 'FLAIR', f'{i}.png')) 

        if epoch % 5 == 0:
            print("\n-------------------------------------------------------\n")
            with torch.no_grad():
                for i, (image, segmentImage) in enumerate(testLoader):
                    image, segmentImage = image.to(device), segmentImage.to(device)

                    outImage = net(image)
                    testLoss = lossFunction(outImage, segmentImage)
                    testLossList = np.append(testLossList, testLoss.item())

                    sensitivity = 0
                    specificity = 0
                    iters = 0

                    for j in range(outImage.size(dim=0)):
                        
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

                    if sensitivity != 0:
                        testSensitivityList = np.append(testSensitivityList, np.array(sensitivity/iters))
                    else:
                        testSensitivityList = np.append(testSensitivityList, np.NaN)
                    if specificity != 0:
                        testSpecificityList = np.append(testSpecificityList, np.array(specificity/iters))
                    else:
                        testSpecificityList = np.append(testSpecificityList, np.NaN)

                    _image = image[0]
                    _segmentImage = segmentImage[0]
                    _outImage = outImage[0]

                    print(f'test-{int(epoch / 5)}_{i}_test_loss=====>>{testLoss.item()}')

                    if sensitivity != 0:
                        print(f'test-{int(epoch / 5)}-{i}_sensitivity=====>>{sensitivity/iters}')
                    else:
                        print(f'test-{int(epoch / 5)}-{i}_sensitivity=====>>{np.NaN}')
                    if specificity != 0:
                        print(f'test-{int(epoch / 5)}-{i}_specificity=====>>{specificity/iters}')
                    else:
                        print(f'test-{int(epoch / 5)}-{i}_specificity=====>>{np.NaN}')



                    testImage = torch.stack([_image, _segmentImage, _outImage], dim=0)
                    torchvision.utils.save_image(testImage, os.path.join(predictPath, 'FLAIR', f'{i}.png'))

            print("\n-------------------------------------------------------\n")

            #-----------------------------------------------------------------------------
            # Test Monitor Data Plot

            testLossList_x = np.arange(len(testLossList))
            fig = plt.figure(num="Test_Loss", figsize=(30, 30))
            plt.title(f'test times {int(epoch/5)}: Test Loss')
            plt.xlabel('Test Times')
            plt.ylabel('Test Loss')
            plt.plot(testLossList_x, testLossList)
            plt.legend(title='test loss', loc='upper right', labels='test loss')
            plt.savefig(os.path.join(predictPath, 'FLAIR', 'test_loss_monitor', f'test_times_{int(epoch/5)}_TestLoss.png'))
            testSensitivityList_x = np.arange(len(testSensitivityList))
            fig = plt.figure(num="Test_Sensitivity", figsize=(30, 30))
            plt.title(f'test times {int(epoch/5)}: Sensitivity')
            plt.xlabel('Test Times')
            plt.ylabel('Sensitivity')
            plt.plot(testSensitivityList_x, testSensitivityList)
            plt.legend(title='sensitivity', loc='upper right', labels='sensitivity')
            plt.savefig(os.path.join(predictPath, 'FLAIR', 'test_sensitivity_monitor', f'test_times_{int(epoch/5)}_Sensitivity.png'))

            testSpecificityList_x = np.arange(len(testSpecificityList))
            fig = plt.figure(num="Test_Specificity", figsize=(30, 30))
            plt.title(f'test times {int(epoch/5)}: Specificity')
            plt.xlabel('Test Times')
            plt.ylabel('Specificity')
            plt.plot(testSpecificityList_x, testSpecificityList)
            plt.legend(title='specificity', loc='upper right', labels='specificty')
            plt.savefig(os.path.join(predictPath, 'FLAIR', 'test_specificity_monitor', f'test_times_{int(epoch/5)}_Specificity.png'))

            plt.close('all')

            f = open(os.path.join(predictPath, 'FLAIR', 'test_loss_monitor', 'log.txt'), "a")
            f.write(f'test times {int(epoch/5)}:\n')
            for t_loss in testLossList:
                f.write(f'{t_loss}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            f = open(os.path.join(predictPath, 'FLAIR', 'test_sensitivity_monitor', 'log.txt'), "a")
            f.write(f'test times {int(epoch/5)}:\n')
            for t_sen in testSensitivityList:
                f.write(f'{t_sen}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            f = open(os.path.join(predictPath, 'FLAIR', 'test_specificity_monitor', 'log.txt'), "a")
            f.write(f'test times {int(epoch/5)}:\n')
            for t_spec in testSpecificityList:
                f.write(f'{t_spec}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            averageTestLossList = np.append(averageTestLossList, testLossList.sum()/len(testLossList))
            averageTestSensitivityList = np.append(averageTestSensitivityList, testSensitivityList.sum()/len(testSensitivityList))
            averageTestSpecificityList = np.append(averageTestSpecificityList, testSpecificityList.sum()/len(testSpecificityList))
            
            testLossList = np.array([])
            testSensitivityList = np.array([])
            testSpecificityList = np.array([])

    

        #-----------------------------------------------------------------------------
        # Train Monitor Data Plot

        trainLossList_x = np.arange(len(trainLossList))
        fig = plt.figure(num="Train_Loss", figsize=(30, 30))
        plt.title(f'epoch{epoch}: Train Loss')
        plt.xlabel('Train Times')
        plt.ylabel('Train Loss')
        plt.plot(trainLossList_x, trainLossList)
        plt.legend(title='train loss', loc='upper right', labels='train loss')
        plt.savefig(os.path.join(savePath, 'FLAIR', 'train_loss_monitor', f'epoch{epoch}_TrainLoss.png'))

        trainSensitivityList_x = np.arange(len(trainSensitivityList))
        fig = plt.figure(num="Train_Sensitivity", figsize=(30, 30))
        plt.title(f'epoch{epoch}: Sensitivity')
        plt.xlabel('Train Times')
        plt.ylabel('Sensitivity')
        plt.plot(trainSensitivityList_x, trainSensitivityList)
        plt.legend(title='sensitivity', loc='upper right', labels='sensitivity')
        plt.savefig(os.path.join(savePath, 'FLAIR', 'train_sensitivity_monitor', f'epoch{epoch}_Sensitivity.png'))

        trainSpecificityList_x = np.arange(len(trainSpecificityList))
        fig = plt.figure(num="Train_Specificity", figsize=(30, 30))
        plt.title(f'epoch{epoch}: Specificity')
        plt.xlabel('Train Times')
        plt.ylabel('Specificity')
        plt.plot(trainSpecificityList_x, trainSpecificityList)
        plt.legend(title='specificity', loc='upper right', labels='specificty')
        plt.savefig(os.path.join(savePath, 'FLAIR', 'train_specificity_monitor', f'epoch{epoch}_Specificity.png'))

        plt.close('all')

        f = open(os.path.join(savePath, 'FLAIR', 'train_loss_monitor', 'log.txt'), "a")
        f.write(f'epoch{epoch}:\n')
        for t_loss in trainLossList:
            f.write(f'{t_loss}\n')
        f.write('--------------------***--------------------\n\n\n')
        f.close()

        f = open(os.path.join(savePath, 'FLAIR', 'train_sensitivity_monitor', 'log.txt'), "a")
        f.write(f'epoch{epoch}:\n')
        for t_sen in trainSensitivityList:
            f.write(f'{t_sen}\n')
        f.write('--------------------***--------------------\n\n\n')
        f.close()

        f = open(os.path.join(savePath, 'FLAIR', 'train_specificity_monitor', 'log.txt'), "a")
        f.write(f'epoch{epoch}:\n')
        for t_spec in trainSpecificityList:
            f.write(f'{t_spec}\n')
        f.write('--------------------***--------------------\n\n\n')
        f.close()


        averageTrainLossList = np.append(averageTrainLossList, trainLossList.sum()/len(trainLossList))
        averageTrainSensitivityList = np.append(averageTrainSensitivityList, trainSensitivityList.sum()/len(trainSensitivityList))
        averageTrainSpecificityList = np.append(averageTrainSpecificityList, trainSpecificityList.sum()/len(trainSpecificityList))

        trainLossList = np.array([])
        trainSensitivityList = np.array([])
        trainSpecificityList = np.array([])

        if epoch % 20 == 0:
            averageTrainLossList_x = np.arange(len(averageTrainLossList))
            fig = plt.figure(num="Train_Loss", figsize=(30, 30))
            plt.title(f'epoch{epoch-19} - epoch{epoch}: Train Loss')
            plt.xlabel('epoch')
            plt.ylabel('Train Loss')
            plt.plot(averageTrainLossList_x, averageTrainLossList)
            plt.legend(title='train loss', loc='upper right', labels='train loss')
            plt.savefig(os.path.join(savePath, 'FLAIR', 'train_loss_monitor', f'epoch{epoch-19}_epoch{epoch}:_TrainLoss.png'))

            averageTrainSensitivityList_x = np.arange(len(averageTrainSensitivityList))
            fig = plt.figure(num="Train_Sensitivity", figsize=(30, 30))
            plt.title(f'epoch{epoch-19} - epoch{epoch}: Sensitivity')
            plt.xlabel('epoch')
            plt.ylabel('Sensitivity')
            plt.plot(averageTrainSensitivityList_x, averageTrainSensitivityList)
            plt.legend(title='sensitivity', loc='upper right', labels='sensitivity')
            plt.savefig(os.path.join(savePath, 'FLAIR', 'train_sensitivity_monitor', f'epoch{epoch-19}_epoch{epoch}_Sensitivity.png'))

            averageTrainSpecificityList_x = np.arange(len(averageTrainSpecificityList))
            fig = plt.figure(num="Train_Specificity", figsize=(30, 30))
            plt.title(f'epoch{epoch-19} - epoch{epoch}: Specificity')
            plt.xlabel('epoch')
            plt.ylabel('Specificity')
            plt.plot(averageTrainSpecificityList_x, averageTrainSpecificityList)
            plt.legend(title='specificity', loc='upper right', labels='specificty')
            plt.savefig(os.path.join(savePath, 'FLAIR', 'train_specificity_monitor', f'epoch{epoch-19}_epoch{epoch}_Specificity.png'))
            
            plt.close('all')

            f = open(os.path.join(savePath, 'FLAIR', 'train_loss_monitor', 'log.txt'), "a")
            f.write('***AVERAGE***\n')
            f.write(f'epoch{epoch-19} - epoch{epoch}:\n')
            for t_loss in averageTrainLossList:
                f.write(f'{t_loss}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            f = open(os.path.join(savePath, 'FLAIR', 'train_sensitivity_monitor', 'log.txt'), "a")
            f.write('***AVERAGE***\n')
            f.write(f'epoch{epoch-19} - epoch{epoch}:\n')
            for t_sen in averageTrainSensitivityList:
                f.write(f'{t_sen}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            f = open(os.path.join(savePath, 'FLAIR', 'train_specificity_monitor', 'log.txt'), "a")
            f.write('***AVERAGE***\n')
            f.write(f'epoch{epoch-19} - epoch{epoch}:\n')
            for t_spec in averageTrainSpecificityList:
                f.write(f'{t_spec}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            averageTrainLossList = np.array([])
            averageTrainSensitivityList = np.array([])
            averageTrainSpecificityList = np.array([])

        if epoch % 100 == 0:
            averageTestLossList_x = np.arange(len(averageTestLossList))
            fig = plt.figure(num="Test_Loss", figsize=(30, 30))
            plt.title(f'Already epoch{epoch}:Test Loss')
            plt.xlabel('epoch')
            plt.ylabel('Test Loss')
            plt.plot(averageTestLossList_x, averageTestLossList)
            plt.legend(title='test loss', loc='upper right', labels='test loss')
            plt.savefig(os.path.join(predictPath, 'FLAIR', 'test_loss_monitor', f'Already_epoch{epoch}:_TestLoss.png'))

            averageTestSensitivityList_x = np.arange(len(averageTestSensitivityList))
            fig = plt.figure(num="Test_Sensitivity", figsize=(30, 30))
            plt.title(f'Already epoch{epoch}: Sensitivity')
            plt.xlabel('epoch')
            plt.ylabel('Sensitivity')
            plt.plot(averageTestSensitivityList_x, averageTestSensitivityList)
            plt.legend(title='sensitivity', loc='upper right', labels='sensitivity')
            plt.savefig(os.path.join(predictPath, 'FLAIR', 'test_sensitivity_monitor', f'Already epoch{epoch}_Sensitivity.png'))

            averageTestSpecificityList_x = np.arange(len(averageTestSpecificityList))
            fig = plt.figure(num="Test_Specificity", figsize=(30, 30))
            plt.title(f'Already epoch{epoch}: Specificity')
            plt.xlabel('epoch')
            plt.ylabel('Specificity')
            plt.plot(averageTestSpecificityList_x, averageTestSpecificityList)
            plt.legend(title='specificity', loc='upper right', labels='specificty')
            plt.savefig(os.path.join(predictPath, 'FLAIR', 'test_specificity_monitor', f'Already epoch{epoch}_Specificity.png'))
            
            plt.close('all')

            f = open(os.path.join(predictPath, 'FLAIR', 'test_loss_monitor', 'log.txt'), "a")
            f.write('***AVERAGE***\n')
            f.write(f'Already epoch{epoch}:\n')
            for t_loss in averageTestLossList:
                f.write(f'{t_loss}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            f = open(os.path.join(predictPath, 'FLAIR', 'test_sensitivity_monitor', 'log.txt'), "a")
            f.write('***AVERAGE***\n')
            f.write(f'Already epoch{epoch}:\n')
            for t_sen in averageTestSensitivityList:
                f.write(f'{t_sen}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            f = open(os.path.join(predictPath, 'FLAIR', 'test_specificity_monitor', 'log.txt'), "a")
            f.write('***AVERAGE***\n')
            f.write(f'Already epoch{epoch}:\n')
            for t_spec in averageTestSpecificityList:
                f.write(f'{t_spec}\n')
            f.write('--------------------***--------------------\n\n\n')
            f.close()

            averageTestLossList = np.array([])
            averageTestSensitivityList = np.array([])
            averageTestSpecificityList = np.array([])


        epoch += 1

        f = open(os.path.join(modelPath, 'FLAIR_epoch.txt'), "w")
        f.write(f'{epoch}')
        f.close()