import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

__basePath = r''
__dataPath = os.path.join(__basePath, 'data')
__modelPath = os.path.join(__basePath, 'model')
__savePath = os.path.join(__dataPath, 'train_monitor_image_AX')
__predictPath = os.path.join(__dataPath, 'test_image_AX')

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
    def __init__(self, list: np.ndarray, averageList: np.ndarray, MRI_series_this: str, monitor_para: str, train_or_test: str, begin: int):
        self.list = list
        self.averageList = averageList
        self.monitor_para = monitor_para
        self.train_or_test = train_or_test
        self.begin = begin
        self.MRI_series_this = MRI_series_this

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
        plt.savefig(os.path.join(__savePath if self.train_or_test == 'train' else __predictPath, self.MRI_series_this,
                    f'{self.train_or_test}_{self.monitor_para}_monitor', f'epoch{epoch}_{self.monitor_para.capitalize()}.png'))
        plt.close(fig)

    def averageList_plot(self, epoch):
        if self.train_or_test == 'test':
            averageList_x = np.arange(
                start=self.begin, stop=self.begin + 5 * len(self.averageList), step=5)
        else:
            averageList_x = np.arange(
                start=self.begin, stop=self.begin + len(self.averageList))
        fig = plt.figure(figsize=(6, 6))
        plt.title(
            f'epoch{self.begin} - epoch{epoch}: {self.monitor_para.capitalize()}')
        plt.xlabel('epoch')
        plt.ylabel(self.monitor_para)
        plt.plot(averageList_x, self.averageList)
        plt.savefig(os.path.join(__savePath if self.train_or_test == 'train' else __predictPath, self.MRI_series_this, f'{self.train_or_test}_{self.monitor_para}_monitor',
                    f'epoch{self.begin}_epoch{epoch}_{self.monitor_para.capitalize()}.png'))
        plt.close(fig)

    def list_write_into_log(self, epoch):
        if not os.path.isfile(os.path.join(__savePath if self.train_or_test == 'train' else __predictPath, self.MRI_series_this, f'{self.train_or_test}_{self.monitor_para}_monitor', f'log_epoch{epoch}.txt')):
            f = open(os.path.join(__savePath if self.train_or_test == 'train' else __predictPath, self.MRI_series_this,
                     f'{self.train_or_test}_{self.monitor_para}_monitor', f'log_epoch{epoch}.txt'), "x")
            f.close()
        f = open(os.path.join(__savePath if self.train_or_test == 'train' else __predictPath, self.MRI_series_this,
                              f'{self.train_or_test}_{self.monitor_para}_monitor', f'log_epoch{epoch}.txt'), "w")
        f.write(f'epoch{epoch}:\n')
        for monitor_para_ in self.list:
            f.write(f'{monitor_para_}\n')
        f.write('\n')
        f.close()

    def averageList_write_into_log(self, epoch):
        if self.train_or_test == 'test':
            averageList_x = np.arange(
                start=self.begin, stop=self.begin + 5 * len(self.averageList), step=5)
        else:
            averageList_x = np.arange(
                start=self.begin, stop=self.begin + len(self.averageList))
        if not os.path.isfile(os.path.join(__savePath if self.train_or_test == 'train' else __predictPath, self.MRI_series_this, f'{self.train_or_test}_{self.monitor_para}_monitor', f'log.txt')):
            f = open(os.path.join(__savePath if self.train_or_test == 'train' else __predictPath, self.MRI_series_this,
                     f'{self.train_or_test}_{self.monitor_para}_monitor', 'log.txt'), "x")
            f.close()
        f = open(os.path.join(__savePath if self.train_or_test == 'train' else __predictPath, self.MRI_series_this,
                              f'{self.train_or_test}_{self.monitor_para}_monitor', 'log.txt'), "a")
        f.write(f'epoch{self.begin}-epoch{epoch}:\n')
        for i, monitor_para_ in enumerate(self.averageList):
            f.write(f'epoch{averageList_x[i]}:{monitor_para_}\n')
        f.write('\n')
        f.close()

if __name__ == '__main__':
    pass