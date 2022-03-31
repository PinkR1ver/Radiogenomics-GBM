import os
from sklearn.feature_extraction import grid_to_graph
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from torch import nn

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, 'data', 'UNet_evaluation')
model_path = os.path.join(base_path, 'model')
train_path = os.path.join(data_path, 'train_result')
validation_path = os.path.join(data_path, 'validation_result')
test_path = os.path.join(data_path, 'test_result')

def sendmail(content, subject):
    # setting mail server information
    mail_host = 'smtp.gmail.com'

    # password
    mail_pass = '*********'

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

def evalution_result(predict_image: torch.Tensor, ground_truth: torch.Tensor):
    accuracy = 0
    balance_accuracy = 0
    IoU = 0
    precision = 0
    f1score = 0
    sensitivity = 0
    specificity = 0
    iters = 0
    FPR = 0

    for i in range(predict_image.size(dim=0)):

        predict_arr= torch.squeeze(
            predict_image[i].cpu()).detach().numpy()
        predict_arr[predict_arr > 0.5] = 1
        predict_arr[predict_arr <= 0.5] = 0
        truth_arr = torch.squeeze(
            ground_truth[i].cpu()).detach().numpy()

        if np.any(predict_arr) and np.any(truth_arr):
            FP = len(np.where(predict_arr - truth_arr == -1)[0])
            FN = len(np.where(predict_arr - truth_arr == 1)[0])
            TP = len(np.where(predict_arr + truth_arr == 2)[0])
            TN = len(np.where(predict_arr + truth_arr == 0)[0])

            accuracy += (TP + TN) / (TP + TN + FN + FP)
            sensitivity += TP / (TP + FN)
            specificity += TN / (TN + FP)
            precision += TP / (TP + FP)
            balance_accuracy += (TP / (TP + FN) + TN / (TN + FP)) / 2
            IoU += TP / (TP + FN + FP)
            f1score += (2 * TP) / (2 * TP + FP + FN)
            FPR += FP / (FP + TN)
            



            iters += 1

    if accuracy != 0:
        accuracy = accuracy / iters
    else:
        accuracy = np.nan 

    if precision != 0:
        precision = precision / iters
    else:
        precision = np.nan

    if sensitivity != 0:
        sensitivity = sensitivity / iters
    else:
        sensitivity = np.nan
    
    if specificity != 0:
        specificity = specificity / iters
    else:
        specificity = np.nan 

    if balance_accuracy != 0:
        balance_accuracy = balance_accuracy / iters
    else:
        balance_accuracy = np.nan 

    if IoU != 0:
        IoU = IoU / iters
    else:
        IoU = np.nan

    if f1score != 0:
        f1score = f1score / iters
    else:
        f1score = np.nan
    
    if FPR != 0:
        FPR = FPR / iters
    else:
        FPR = np.nan

    return accuracy, precision, sensitivity, specificity, balance_accuracy, IoU, f1score, FPR


# TrainHelper type is to store processing parameters and plot them
# such as loss, sensitivity, specificity
class trainHelper():
    def __init__(self, MRI_series_this='T1', mode='train', begin=1):
        self.loss_list = np.array([])
        self.loss_average_list = np.array([])
        
        self.accuracy_list = np.array([])
        self.accuracy_average_list = np.array([])
        
        self.balance_accuracy_list = np.array([])
        self.balance_accuracy_average_list = np.array([])
        
        self.IoU_list = np.array([])
        self.IoU_average_list = np.array([])

        self.precision_list = np.array([])
        self.precision_average_list = np.array([])

        self.sensitivity_list = np.array([])
        self.sensitivity_average_list = np.array([])

        self.specificity_list = np.array([])
        self.specificity_average_list = np.array([])
        
        self.f1score_list = np.array([])
        self.f1score_average_list = np.array([])

        self.FPR_list = np.array([])
        self.FPR_average_list = np.array([])
        
        self.mode = mode
        self.begin = begin
        self.MRI_series_this = MRI_series_this
        
        if not os.path.isdir(data_path):
            os.mkdir(data_path)   
        if not os.path.isdir(os.path.join(data_path, self.MRI_series_this)):
            os.mkdir(os.path.join(data_path, self.MRI_series_this))
        if not os.path.isdir(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result')):
            os.mkdir(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result'))
        if not os.path.isdir(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'detail')):
            os.mkdir(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'detail'))

        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']
        for i in evalution_params:
            if not os.path.isdir(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'detail', i)):
                os.mkdir(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'detail', i))


    def list_pushback(self, predict_image, groud_truth, loss):
        self.loss_list = np.append(self.loss_list, loss)

        accuracy, precision, sensitivity, specificity, balance_accuracy, IoU, f1score, FPR = evalution_result(predict_image, groud_truth)

        self.accuracy_list = np.append(self.accuracy_list, accuracy)
        self.precision_list = np.append(self.precision_list, precision)
        self.sensitivity_list = np.append(self.sensitivity_list, sensitivity)
        self.specificity_list = np.append(self.specificity_list, specificity)
        self.balance_accuracy_list = np.append(self.balance_accuracy_list, balance_accuracy)
        self.IoU_list = np.append(self.IoU_list, IoU)
        self.f1score_list = np.append(self.f1score_list, f1score)
        self.FPR_list = np.append(self.FPR_list, FPR)

    def average_list_pushback(self):
        
        self.delete_nan_list()
        self.loss_average_list = np.append(
            self.loss_average_list, self.loss_list.sum() / len(self.loss_list))

        self.accuracy_average_list = np.append(
            self.accuracy_average_list, self.accuracy_list.sum() / len(self.accuracy_list))

        self.precision_average_list = np.append(
            self.precision_average_list, self.precision_list.sum() / len(self.precision_list))

        self.sensitivity_average_list = np.append(
            self.sensitivity_average_list, self.sensitivity_list.sum() / len(self.sensitivity_list))
        
        self.specificity_average_list = np.append(
            self.specificity_average_list, self.specificity_list.sum() / len(self.specificity_list))
        
        self.balance_accuracy_average_list = np.append(
            self.balance_accuracy_average_list, self.balance_accuracy_list.sum() / len(self.balance_accuracy_list))
        
        self.IoU_average_list = np.append(
            self.IoU_average_list, self.IoU_list.sum() / len(self.IoU_list))
        
        self.f1score_average_list = np.append(
            self.f1score_average_list, self.f1score_list.sum() / len(self.f1score_list))

        self.FPR_average_list = np.append(
            self.FPR_average_list, self.FPR_list.sum() / len(self.FPR_list))

    def clear_list(self):
        self.loss_list = np.array([])
        self.accuracy_list = np.array([])
        self.precision_list = np.array([])
        self.sensitivity_list = np.array([])
        self.specificity_list = np.array([])
        self.balance_accuracy_list = np.array([])
        self.IoU_list = np.array([])
        self.f1score_list = np.array([])
        self.FPR_list = np.array([])


    def clear_average_list(self):
        self.loss_average_list = np.array([])
        self.accuracy_average_list = np.array([])
        self.precision_average_list = np.array([])
        self.sensitivity_average_list = np.array([])
        self.specificity_average_list = np.array([])
        self.balance_accuracy_average_list = np.array([])
        self.IoU_average_list = np.array([])
        self.f1score_average_list = np.array([])
        self.FPR_average_list = np.array([])

    def delete_nan_list(self):
        self.loss_list = self.loss_list[~(np.isnan(self.loss_list))]
        self.accuracy_list = self.accuracy_list[~(np.isnan(self.accuracy_list))]
        self.precision_list = self.precision_list[~(np.isnan(self.precision_list))]
        self.sensitivity_list = self.sensitivity_list[~(np.isnan(self.sensitivity_list))]
        self.specificity_list = self.specificity_list[~(np.isnan(self.specificity_list))]
        self.balance_accuracy_list = self.balance_accuracy_list[~(np.isnan(self.balance_accuracy_list))]
        self.IoU_list = self.IoU_list[~(np.isnan(self.IoU_list))]
        self.f1score_list = self.f1score_list[~(np.isnan(self.f1score_list))]
        self.FPR_list = self.FPR_list[~(np.isnan(self.FPR_list))]

    def list_plot(self, epoch):
        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']
        for i in evalution_params:
            fig = plt.figure(figsize=(6, 6))
            sns.histplot(eval(f'self.{i}_list'), stat="probability", kde=True)
            plt.title(f'{self.mode}_epoch{epoch}')
            plt.ylabel(i)
            plt.savefig(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'detail', i, f'epoch{epoch}_{i}.png'))
            plt.close(fig)


    def average_list_plot(self, epoch, step=1):
        average_list_x = np.arange(
            start=epoch - step * len(self.loss_average_list) + step, stop=epoch + step, step=step)
        
        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']
        for i in evalution_params:
            fig = plt.figure(figsize=(6, 6))
            plt.title(f'epoch{average_list_x[0]} - epoch{epoch} {i}')
            plt.xlabel('epoch')
            plt.ylabel(i)
            plt.plot(average_list_x, eval(f'self.{i}_average_list'))
            plt.savefig(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', f'epoch{average_list_x[0]}_epoch{epoch}_{i}.png'))
            plt.close(fig)

    def list_write_into_log(self, epoch):
        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']
        for i in evalution_params:
            if not os.path.isfile(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'detail', i, f'log_epoch{epoch}.txt')):
                f = open(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'detail', i, f'log_epoch{epoch}.txt'), "x")
                f.close()

            f = open(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'detail', i, f'log_epoch{epoch}.txt'), "w")
            for val in eval(f'self.{i}_list'):
                f.write(f'{val}\n')
            f.write('\n')
            f.close()

    
    def average_list_write_into_log(self, epoch, step=1):
        average_list_x = np.arange(
            start=epoch - step * len(self.loss_average_list) + step, stop=epoch + step, step=step)
        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']
        if not os.path.isfile(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', f'log.txt')):
            f = open(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'log.txt'), "x")
            f.close()

        f = open(os.path.join(data_path, self.MRI_series_this,  f'{self.mode}_result', 'log.txt'), "a")

        f.write(f'epoch{average_list_x[0]}-epoch{epoch}:\n\n\n')
        for i in range(len(average_list_x)):
            f.write(f'epoch{average_list_x[i]}\n')
            for eval_para in evalution_params:
                val = eval(f'self.{eval_para}_average_list[i]')
                f.write(f'{eval_para}:{val}\n')
            f.write('\n\n')
        f.close()

def compare_train_validation_test(trainHelpers, epoch, step=1):
    average_list_x = np.arange(
        start=epoch - step * len(trainHelpers[0].loss_average_list) + step, stop=epoch + step, step=step)
    evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']
    for i in evalution_params:
        fig = plt.figure(figsize=(6, 6))
        for trainHelper in trainHelpers:
            plt.plot(average_list_x, eval(f'trainHelper.{i}_average_list'))
        
        plt.title(f'epoch{average_list_x[0]} - epoch{epoch} {i}')
        plt.xlabel('epoch')
        plt.ylabel(i)
        
        plt.legend(title='Lines', loc='upper right', labels=['train', 'validation', 'test'])
        plt.savefig(os.path.join(data_path, trainHelpers[0].MRI_series_this, f'epoch{average_list_x[0]}_epoch{epoch}_{i}.png'))
        plt.close(fig)

def ROC_curve(trainHelpers):
    fig = plt.figure(figsize=(6, 6))
    for trainHelper in trainHelpers:
        plt.plot(trainHelper.FPR_average_list, trainHelper.sensitivity_average_list)

    plt.title('ROC Curve')
    plt.xlabel('Flase Positive Rate')
    plt.ylabel('True Postive Rate')
    plt.legend(title='Lines', loc='upper right', labels=['train', 'validation', 'test'])
    plt.savefig(os.path.join(data_path, trainHelpers[0].MRI_series_this, f'ROC_Curve.png'))
    plt.close(fig)


if __name__ == '__main__':
    s = trainHelper("Stack", "train", 1)
    t = trainHelper("Stack", "train", 1)
    x = torch.rand(4, 1, 64, 256, 256)
    y = torch.rand(4, 1, 64, 256, 256)

    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    s.list_pushback(x, y, 0.05)
    s.list_pushback(x, y, 0.04)
    s.list_pushback(x, y, 0.03)
    s.list_pushback(x, y, 0.02)
    s.list_plot(1)
    s.list_write_into_log(1)
    s.average_list_pushback()

    
    t.list_pushback(x, y, 0.05)
    t.list_pushback(x, y, 0.04)
    t.list_pushback(x, y, 0.03)
    t.list_pushback(x, y, 0.02)
    t.average_list_pushback()

    s.average_list_plot(20)

    compare_train_validation_test([s,t,t], 10)
    ROC_curve([s,t,t])

