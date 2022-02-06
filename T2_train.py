import os
import string
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import torchvision
from data import *
from net import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


MRI_series_this = 'T2'

basePath = r''
dataPath = os.path.join(basePath, 'data')
modelPath = os.path.join(basePath, 'model')
weightPath = os.path.join(modelPath, f'{MRI_series_this}_unet.pth')
savePath = os.path.join(dataPath, 'train_monitor_image_AX')
predictPath = os.path.join(dataPath, 'test_image_AX')

def sendmail(content, subject):
    # setting mail server information
    mail_host = 'smtp.gmail.com'

    # password
    mail_pass = '*******'

    # sender mail
    sender = 'fakelspwang@gmail.com'

    # receiver mail, you can set a lots of receiver mails in a list
    receivers = ['pinkr1veroops@gmail.com']

    # message information

    for i in range(len(receivers)):

        message = MIMEMultipart()
        message.attach(MIMEText(content, 'plain', 'utf-8'))

        message['Subject'] = subject

        message['From'] = sender

        message['To'] = receivers[i]

        # try send mail
        try:
            # login and send
            smtpObj = smtplib.SMTP_SSL(mail_host, 465)

            smtpObj.login(sender, mail_pass)
            smtpObj.sendmail(
                sender, receivers, message.as_string())

            smtpObj.quit()
            
            print('send success')
        except smtplib.SMTPException as e:
            print('sending error', e)  
            smtpObj.quit()


def sensitivity_and_specificity_calculation(groundtruth: torch.Tensor, predictImage: torch.Tensor):
    sensitivity = 0
    specificity = 0
    iters = 0

    for i in range(predictImage.size(dim=0)):

        predictMask_arr = torch.squeeze(
            predictImage[i].cpu()).detach().numpy()
        predictMask_arr[predictMask_arr > 0.5] = 1
        predictMask_arr[predictMask_arr <= 0.5] = 0
        mask_arr = torch.squeeze(
            groundtruth[i].cpu()).detach().numpy()

        if np.any(predictMask_arr) and np.any(mask_arr):
            FP = len(np.where(predictMask_arr - mask_arr == -1)[0])
            FN = len(np.where(predictMask_arr - mask_arr == 1)[0])
            TP = len(np.where(predictMask_arr + mask_arr == 2)[0])
            TN = len(np.where(predictMask_arr + mask_arr == 0)[0])

            sensitivity += TP / (TP + FN)
            specificity += TN / (TN + FP)
            iters += 1

    if sensitivity != 0:
        sensitivity = sensitivity / iters
    else:
        sensitivity = np.nan
    if specificity != 0:
        specificity = specificity / iters
    else:
        specificity = np.nan

    return sensitivity, specificity


# TrainHelper type is to store processing parameters and plot them
# such as loss, sensitivity, specificity
class trainHelper():
    def __init__(self, list: np.ndarray, averageList: np.ndarray, monitor_para: str, train_or_test: str, begin: int):
        self.list = list
        self.averageList = averageList
        self.monitor_para = monitor_para
        self.train_or_test = train_or_test
        self.begin = begin

    def list_pushback(self, num):
        self.list = np.append(self.list, num)

    def averageList_pushback(self):
        self.delete_nan_list()
        self.averageList = np.append(
            self.averageList, self.list.sum() / len(self.list))

    def clear_list(self):
        self.list = np.array([])

    def clear_averageList(self):
        self.averageList = np.array([])

    def delete_nan_list(self):
        self.list = self.list[~(np.isnan(self.list))]

    def list_plot(self, epoch):
        fig = plt.figure(figsize=(6, 6))
        sns.histplot(self.list, stat="probability", kde=True)
        plt.title(
            f'{self.train_or_test}_epoch{epoch}:{self.monitor_para.capitalize()}')
        plt.xlabel(self.monitor_para.capitalize())
        plt.savefig(os.path.join(savePath if self.train_or_test == 'train' else predictPath, MRI_series_this,
                    f'{self.train_or_test}_{self.monitor_para}_monitor', f'epoch{epoch}_{self.monitor_para.capitalize()}.png'))
        plt.close(fig)

    def averageList_plot(self, epoch):
        if self.train_or_test == 'test':
            averageList_x = np.arange(
                start=self.begin, stop=self.begin + 20 * len(self.averageList), step=5)
        else:
            averageList_x = np.arange(
                start=self.begin, stop=self.begin + len(self.averageList))
        fig = plt.figure(figsize=(6, 6))
        plt.title(
            f'epoch{self.begin} - epoch{epoch}: {self.monitor_para.capitalize()}')
        plt.xlabel('epoch')
        plt.ylabel(self.monitor_para)
        plt.plot(averageList_x, self.averageList)
        plt.savefig(os.path.join(savePath if self.train_or_test == 'train' else predictPath, MRI_series_this, f'{self.train_or_test}_{self.monitor_para}_monitor',
                    f'epoch{self.begin}_epoch{epoch}_{self.monitor_para.capitalize()}.png'))
        plt.close(fig)

    def list_write_into_log(self, epoch):
        if not os.path.isfile(os.path.join(savePath if self.train_or_test == 'train' else predictPath, MRI_series_this, f'{self.train_or_test}_{self.monitor_para}_monitor', f'log_epoch{epoch}.txt')):
            f = open(os.path.join(savePath if self.train_or_test == 'train' else predictPath, MRI_series_this,
                     f'{self.train_or_test}_{self.monitor_para}_monitor', f'log_epoch{epoch}.txt'), "x")
            f.close()
        f = open(os.path.join(savePath if self.train_or_test == 'train' else predictPath, MRI_series_this,
                              f'{self.train_or_test}_{self.monitor_para}_monitor', f'log_epoch{epoch}.txt'), "w")
        f.write(f'epoch{epoch}:\n')
        for monitor_para_ in self.list:
            f.write(f'{monitor_para_}\n')
        f.write('\n')
        f.close()

    def averageList_write_into_log(self, epoch):
        if self.train_or_test == 'test':
            averageList_x = np.arange(
                start=self.begin, stop=self.begin + 20 * len(self.averageList), step=5)
        else:
            averageList_x = np.arange(
                start=self.begin, stop=self.begin + len(self.averageList))
        if not os.path.isfile(os.path.join(savePath if self.train_or_test == 'train' else predictPath, MRI_series_this, f'{self.train_or_test}_{self.monitor_para}_monitor', f'log.txt')):
            f = open(os.path.join(savePath if self.train_or_test == 'train' else predictPath, MRI_series_this,
                     f'{self.train_or_test}_{self.monitor_para}_monitor', 'log.txt'), "x")
            f.close()
        f = open(os.path.join(savePath if self.train_or_test == 'train' else predictPath, MRI_series_this,
                              f'{self.train_or_test}_{self.monitor_para}_monitor', 'log.txt'), "a")
        f.write(f'epoch{self.begin}-epoch{epoch}:\n')
        for i, monitor_para_ in enumerate(self.averageList):
            f.write(f'epoch{averageList_x[i]}:{monitor_para_}\n')
        f.write('\n')
        f.close()


if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

if __name__ == '__main__':


    #--------------------------------------------------------------#
    #   Load data and init some list to monitor processing         #
    #                                                              #
    #                                                              #
    #--------------------------------------------------------------#

    fullTrainDataset = Train_T2_AX_ImageDataset(
        dataPath, 'GBM_MRI_Dataset.csv')
    trainingDataSize = 0.8

    trainSize = int(trainingDataSize * len(fullTrainDataset))
    testSize = len(fullTrainDataset) - trainSize

    trainDataset, testDataset = torch.utils.data.random_split(
        fullTrainDataset, [trainSize, testSize])

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

    f = open(os.path.join(modelPath, f'{MRI_series_this}_epoch.txt'), "r")
    epoch = int(f.read())
    f.close()

    # ---------------------------------------------------------------
    # Init some list to monitor processing

    trainLossList = trainHelper(list=np.array([]), averageList=np.array(
        []), monitor_para="loss", train_or_test="train", begin=epoch)
    trainSensitivityList = trainHelper(list=np.array([]), averageList=np.array(
        []), monitor_para="sensitivity", train_or_test="train", begin=epoch)
    trainSpecificityList = trainHelper(list=np.array([]), averageList=np.array(
        []), monitor_para="specificity", train_or_test="train", begin=epoch)

    testLossList = trainHelper(list=np.array([]), averageList=np.array(
        []), monitor_para="loss", train_or_test="test", begin=epoch)
    testSensitivityList = trainHelper(list=np.array([]), averageList=np.array(
        []), monitor_para="sensitivity", train_or_test="test", begin=epoch)
    testSpecificityList = trainHelper(list=np.array([]), averageList=np.array(
        []), monitor_para="specificity", train_or_test="test", begin=epoch)

    #---------------------------------------------------------------------------------------------------#
    #   train processing                                                                                #
    #   save model every epoch                                                                          #
    #   plot epoch monitor image every epoch                                                            #
    #   plot wrong information every break                                                              #
    #   plot average epoch's parameters (loss, sensitivity, specificity) every 100 epoch or something   #
    #---------------------------------------------------------------------------------------------------#

    try:
        for iter_out in range(1001):

            for i, (image, mask) in enumerate(trainLoader):

                #---------------------#
                # The main train loop #
                #---------------------#

                image, mask = image.to(device), mask.to(device)

                outImage = net(image)
                trainLoss = lossFunction(outImage, mask)
                trainLossList.list_pushback(trainLoss.item())

                sensitivity, specificity = sensitivity_and_specificity_calculation(
                    groundtruth=mask, predictImage=outImage)
                trainSensitivityList.list_pushback(sensitivity)
                trainSpecificityList.list_pushback(specificity)

                opt.zero_grad()
                trainLoss.backward()
                opt.step()

                if i % 5 == 0:
                    print(f'{epoch}-{i}_train loss=====>>{trainLoss.item()}')
                    if sensitivity != 0:
                        print(
                            f'{epoch}-{i}_sensitivity=====>>{sensitivity}')
                    else:
                        print(f'{epoch}-{i}_sensitivity=====>>{np.NaN}')
                    if specificity != 0:
                        print(
                            f'{epoch}-{i}_specificity=====>>{specificity}')
                    else:
                        print(f'{epoch}-{i}_specificity=====>>{np.NaN}')

                _image = image[0]
                _mask = mask[0]
                _outImage = outImage[0]

                testImage = torch.stack([_image, _mask, _outImage], dim=0)
                torchvision.utils.save_image(
                    testImage, os.path.join(savePath, MRI_series_this, f'{i}.png'))

            # -----------------------------------------------------------------------------
            # After a epoch, we will plot some histgram to monitor this epoch performance
            # And write trainning data into epoch

            trainLossList.list_plot(epoch)
            trainSensitivityList.list_plot(epoch)
            trainSpecificityList.list_plot(epoch)

            trainLossList.list_write_into_log(epoch)
            trainSensitivityList.list_write_into_log(epoch)
            trainSpecificityList.list_write_into_log(epoch)

            trainLossList.averageList_pushback()
            trainSensitivityList.averageList_pushback()
            trainSpecificityList.averageList_pushback()

            # Init list again for next epoch

            trainLossList.clear_list()
            trainSensitivityList.clear_list()
            trainSpecificityList.clear_list()

            # -------------------------------------------------------------------------------
            # for every 5 epoches, do a test epoch

            if iter_out % 5 == 0:
                print("\n-------------------------------------------------------\n")
                with torch.no_grad():
                    for i, (image, mask) in enumerate(testLoader):
                        image, mask = image.to(device), mask.to(device)

                        outImage = net(image)
                        testLoss = lossFunction(outImage, mask)
                        testLossList.list_pushback(testLoss.item())

                        sensitivity, specificity = sensitivity_and_specificity_calculation(
                            groundtruth=mask, predictImage=outImage)

                        testSensitivityList.list_pushback(sensitivity)
                        testSpecificityList.list_pushback(specificity)

                        _image = image[0]
                        _mask = mask[0]
                        _outImage = outImage[0]

                        print(
                            f'test-{int(iter_out / 5) + 1}_{i}_test_loss=====>>{testLoss.item()}')

                        if sensitivity != 0:
                            print(
                                f'test-{int(iter_out / 5) + 1}-{i}_sensitivity=====>>{sensitivity}')
                        else:
                            print(
                                f'test-{int(iter_out / 5) + 1}-{i}_sensitivity=====>>{np.NaN}')
                        if specificity != 0:
                            print(
                                f'test-{int(iter_out / 5) + 1}-{i}_specificity=====>>{specificity}')
                        else:
                            print(
                                f'test-{int(iter_out / 5) + 1}-{i}_specificity=====>>{np.NaN}')

                        testImage = torch.stack(
                            [_image, _mask, _outImage], dim=0)
                        torchvision.utils.save_image(
                            testImage, os.path.join(predictPath, MRI_series_this, f'{i}.png'))

                print("\n-------------------------------------------------------\n")

                # -----------------------------------------------------------------------------
                # Test Monitor Data Plot and write into epoch

                testLossList.list_plot(epoch)
                testSensitivityList.list_plot(epoch)
                testSpecificityList.list_plot(epoch)

                testLossList.list_write_into_log(epoch)
                testSensitivityList.list_write_into_log(epoch)
                testSpecificityList.list_write_into_log(epoch)

                testLossList.averageList_pushback()
                testSensitivityList.averageList_pushback()
                testSpecificityList.averageList_pushback()

                # Init list again for next epoch

                testLossList.clear_list()
                testSensitivityList.clear_list()
                testSpecificityList.clear_list()

            # ---------------------------------------------------------------
            # for every 100 epoches, we will plot the parameter change

            if iter_out != 0 and iter_out % 100 == 0:

                trainLossList.averageList_plot(epoch)
                trainSensitivityList.averageList_plot(epoch)
                trainSpecificityList.averageList_plot(epoch)

                trainLossList.averageList_write_into_log(epoch)
                trainSensitivityList.averageList_write_into_log(epoch)
                trainSpecificityList.averageList_write_into_log(epoch)

                trainLossList.clear_averageList()
                trainSensitivityList.clear_averageList()
                trainSpecificityList.clear_averageList()

                # also for the test part

                testLossList.averageList_plot(epoch)
                testSensitivityList.averageList_plot(epoch)
                testSpecificityList.averageList_plot(epoch)

                testLossList.averageList_write_into_log(epoch)
                testSensitivityList.averageList_write_into_log(epoch)
                testSpecificityList.averageList_write_into_log(epoch)

                testLossList.clear_averageList()
                testSensitivityList.clear_averageList()
                testSpecificityList.clear_averageList()

            # for every epoch end, save model and add epoch

            torch.save(net.state_dict(), weightPath)

            epoch += 1

            f = open(os.path.join(
                modelPath, f'{MRI_series_this}_epoch.txt'), "w")
            f.write(f'{epoch}')
            f.close()

    except:
        print('Exception!!!')
        if not os.path.isfile(os.path.join(basePath, 'exception_in_trainning', f'{MRI_series_this}_log.txt')):
            f = open(os.path.join(basePath, 'exception_in_trainning', f'{MRI_series_this}_log.txt'), 'x')
            f.close()
        f = open(os.path.join(basePath, 'exception_in_trainning',
                 f'{MRI_series_this}_log.txt'), 'a')
        f.write(f'exception in epoch{epoch}\n')
        f.write(traceback.format_exc())
        f.write('\n')
        f.close()

        if trainLossList.averageList.size != 0:
            trainLossList.averageList_plot(epoch)
            trainSensitivityList.averageList_plot(epoch)
            trainSpecificityList.averageList_plot(epoch)

            trainLossList.averageList_write_into_log(epoch)
            trainSensitivityList.averageList_write_into_log(epoch)
            trainSpecificityList.averageList_write_into_log(epoch)
        
        if testLossList.averageList.size != 0:
            testLossList.averageList_plot(epoch)
            testSensitivityList.averageList_plot(epoch)
            testSpecificityList.averageList_plot(epoch)

            testLossList.averageList_write_into_log(epoch)
            testSensitivityList.averageList_write_into_log(epoch)
            testSpecificityList.averageList_write_into_log(epoch)

        sendmail(content=r'Your train.py went something wrong', subject=r'train.py go wrong')
    
    else:
        print("Train finishing")
        if trainLossList.averageList.size != 0:
            trainLossList.averageList_plot(epoch)
            trainSensitivityList.averageList_plot(epoch)
            trainSpecificityList.averageList_plot(epoch)

            trainLossList.averageList_write_into_log(epoch)
            trainSensitivityList.averageList_write_into_log(epoch)
            trainSpecificityList.averageList_write_into_log(epoch)
        
        if testLossList.averageList.size != 0:
            testLossList.averageList_plot(epoch)
            testSensitivityList.averageList_plot(epoch)
            testSpecificityList.averageList_plot(epoch)

            testLossList.averageList_write_into_log(epoch)
            testSensitivityList.averageList_write_into_log(epoch)
            testSpecificityList.averageList_write_into_log(epoch)

        sendmail(content=r'Your train.py run success', subject=r'train.py finished')
