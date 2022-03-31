import os
from cv2 import sqrt
from sklearn.feature_extraction import grid_to_graph
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import math
import cv2
import gc
from tabulate import tabulate
import pandas as pd

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, 'data', 'result')

def sendmail(content, subject):
    # setting mail server information
    mail_host = 'smtp.gmail.com'

    # password
    mail_pass = '***********'

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


def ROC_to_calculate_thresold(pred_path, truth_path, save_path=None, save_or_not=False):
    preds = []
    truths = []
    for i in os.listdir(pred_path):
        pred = cv2.imread(os.path.join(pred_path, i), cv2.IMREAD_GRAYSCALE)
        preds.append(pred)

        truth = cv2.imread(os.path.join(truth_path, i), cv2.IMREAD_GRAYSCALE)
        truths.append(truth)
    

    FPR = np.array([])
    TPR = np.array([])
    G_mean_flag = (0, 0)
    G_mean_max = 0
    threshold_flag = 0
    for i in len(preds):
        FP, FN, TP, TN = 0
        for threshold in np.arange(0, 255, 1):
            tmp_pred = preds[i]
            tmp_pred = tmp_pred[tmp_pred >= threshold] = 255
            tmp_pred = tmp_pred[tmp_pred < threshold] = 0

            tmp_truth = truths[i]
            tmp_truth = tmp_truth[tmp_truth >= threshold] = 255
            tmp_truth = tmp_truth[tmp_truth < threshold] = 0

            FP += len(np.where(tmp_pred - tmp_truth == -255)[0])
            FN += len(np.where(tmp_pred - tmp_truth == 255)[0])
            TP += len(np.where(tmp_pred + tmp_truth == 510)[0])
            TN += len(np.where(tmp_pred + tmp_truth == 0)[0])
        
        FPR = np.append(FPR, FP / (FP + TN))
        TPR = np.append(TPR, TP / (TP + FN))
        recall = TP / (TP + TN)
        specificity = TN / (TN + FP)
        G_mean = math.sqrt(recall * specificity)
        if G_mean > G_mean_max:
            G_mean_max = G_mean
            G_mean_flag = (FPR[-1], TPR[-1])
            threshold_flag = threshold

    if save_or_not:
        fig = plt.figure()
        plt.plot(FPR, TPR)
        plt.title('ROC Curve')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.xlabel('False Positive Rate')
        plt.ylabel('Ture Positeve Rate')
        plt.annotate(f'Max G-mean with {G_mean_max}, threshold={threshold_flag}', G_mean_flag)
        plt.savefig(save_path)

    return threshold_flag

def f1score_to_calculate_thresold(pred_path, truth_path, save_path=None, save_or_not=False):
    preds = []
    truths = []
    for i in os.listdir(pred_path):
        pred = cv2.imread(os.path.join(pred_path, i), cv2.IMREAD_GRAYSCALE)
        preds.append(pred)

        truth = cv2.imread(os.path.join(truth_path, i), cv2.IMREAD_GRAYSCALE)
        truths.append(truth)
    

    f1score = np.array([])
    f1score_flag = (0, 0)
    f1score_max = 0
    threshold_flag = 0
    for i in len(preds):
        FP, FN, TP, TN = 0
        for threshold in np.arange(0, 255, 1):
            tmp_pred = preds[i]
            tmp_pred = tmp_pred[tmp_pred >= threshold] = 255
            tmp_pred = tmp_pred[tmp_pred < threshold] = 0

            tmp_truth = truths[i]
            tmp_truth = tmp_truth[tmp_truth >= threshold] = 255
            tmp_truth = tmp_truth[tmp_truth < threshold] = 0

            FP += len(np.where(tmp_pred - tmp_truth == -255)[0])
            FN += len(np.where(tmp_pred - tmp_truth == 255)[0])
            TP += len(np.where(tmp_pred + tmp_truth == 510)[0])
            TN += len(np.where(tmp_pred + tmp_truth == 0)[0])
        

        f1score = np.append(f1score, (2 * TP) / (2 * TP + FP + FN))
        if f1score[-1] > f1score_max:
            f1score_max = f1score[-1]
            f1score_flag = (threshold, f1score[-1])
            threshold_flag = threshold

    if save_or_not:
        fig = plt.figure()
        plt.plot(np.arange(0.0, 1.0, 0.0001), f1score)
        plt.title('Threshold Tuning Curve')
        plt.xlim((0, 255))
        plt.ylim((0, 1))
        plt.xlabel('Threshold')
        plt.ylabel('F1-score')
        plt.annotate(f'Max F1-score with {f1score_max }, threshold={threshold_flag}', f1score_flag)
        plt.savefig(save_path)

    return threshold_flag


def evalation_all(train_path, validation_path, test_path, ROC_cruve_save_path):
    thresold = ROC_to_calculate_thresold(pred_path=train_path[0], truth_path=train_path[1], save_path=ROC_cruve_save_path, save_or_not=True)

    train_preds = []
    train_truths = []
    for i in os.listdir(train_path[0]):
        train_pred = cv2.imread(os.path.join(train_path[0], i), cv2.IMREAD_GRAYSCALE)
        train_pred = train_pred[train_pred >= thresold] = 255
        train_pred = train_pred[train_pred < thresold] = 0
        train_preds.append(train_pred)

        train_truth = cv2.imread(os.path.join(train_path[1], i), cv2.IMREAD_GRAYSCALE)
        train_truth = train_truth[train_truth >= thresold] = 255
        train_truth = train_truth[train_truth < thresold] = 0
        train_truths.append(train_truth)

    for i in len(train_preds):
        FP, FN, TP, TN = 0

        FP += len(np.where(train_preds[i] - train_truth[i] == -255)[0])
        FN += len(np.where(train_preds[i] - train_truth[i] == 255)[0])
        TP += len(np.where(train_preds[i] + train_truth[i] == 510)[0])
        TN += len(np.where(train_preds[i] + train_truth[i] == 0)[0])


    train_accuracy = (TP + TN) / (TP + TN + FN + FP)
    train_sensitivity = TP / (TP + FN)
    train_specificity = TN / (TN + FP)
    train_precision = TP / (TP + FP)
    train_balance_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
    train_IoU = TP / (TP + FN + FP)
    train_f1score = (2 * TP) / (2 * TP + FP + FN)

    del train_pred
    del train_truth
    gc.collect()


    validation_preds = []
    validation_truths = []
    for i in os.listdir(validation_path[0]):
        validation_pred = cv2.imread(os.path.join(validation_path[0], i), cv2.IMREAD_GRAYSCALE)
        validation_pred = validation_pred[validation_pred >= thresold] = 255
        validation_pred = validation_pred[validation_pred < thresold] = 0
        validation_preds.append(validation_pred)

        validation_truth = cv2.imread(os.path.join(validation_path[1], i), cv2.IMREAD_GRAYSCALE)
        validation_truth = validation_truth[validation_truth >= thresold] = 255
        validation_truth = validation_truth[validation_truth < thresold] = 0
        validation_truths.append(validation_truth)

    for i in len(validation_preds):
        FP, FN, TP, TN = 0

        FP += len(np.where(validation_preds[i] - validation_truth[i] == -255)[0])
        FN += len(np.where(validation_preds[i] - validation_truth[i] == 255)[0])
        TP += len(np.where(validation_preds[i] + validation_truth[i] == 510)[0])
        TN += len(np.where(validation_preds[i] + validation_truth[i] == 0)[0])


    validation_accuracy = (TP + TN) / (TP + TN + FN + FP)
    validation_sensitivity = TP / (TP + FN)
    validation_specificity = TN / (TN + FP)
    validation_precision = TP / (TP + FP)
    validation_balance_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
    validation_IoU = TP / (TP + FN + FP)
    validation_f1score = (2 * TP) / (2 * TP + FP + FN)

    del validation_pred
    del validation_truth
    gc.collect()

    test_preds = []
    test_truths = []
    for i in os.listdir(test_path[0]):
        test_pred = cv2.imread(os.path.join(test_path[0], i), cv2.IMREAD_GRAYSCALE)
        test_pred = test_pred[test_pred >= thresold] = 255
        test_pred = test_pred[test_pred < thresold] = 0
        test_preds.append(test_pred)

        test_truth = cv2.imread(os.path.join(test_path[1], i), cv2.IMREAD_GRAYSCALE)
        test_truth = test_truth[test_truth >= thresold] = 255
        test_truth = test_truth[test_truth < thresold] = 0
        test_truths.append(test_truth)

    for i in len(test_preds):
        FP, FN, TP, TN = 0

        FP += len(np.where(test_preds[i] - test_truth[i] == -255)[0])
        FN += len(np.where(test_preds[i] - test_truth[i] == 255)[0])
        TP += len(np.where(test_preds[i] + test_truth[i] == 510)[0])
        TN += len(np.where(test_preds[i] + test_truth[i] == 0)[0])


    test_accuracy = (TP + TN) / (TP + TN + FN + FP)
    test_sensitivity = TP / (TP + FN)
    test_specificity = TN / (TN + FP)
    test_precision = TP / (TP + FP)
    test_balance_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
    test_IoU = TP / (TP + FN + FP)
    test_f1score = (2 * TP) / (2 * TP + FP + FN)

    del test_pred
    del test_truth
    gc.collect()

    evaluation_param = {
        "accuracy": [train_accuracy, validation_accuracy, test_accuracy],
        "sensitivity": [train_accuracy, validation_sensitivity, test_sensitivity],
        "specificity": [train_specificity, validation_specificity, test_specificity],
        "precision": [train_precision, validation_precision, test_precision],
        "balance_accuracy": [train_balance_accuracy, validation_balance_accuracy, test_balance_accuracy],
        "IoU": [train_IoU, validation_IoU, test_IoU],
        "f1score": [train_f1score, validation_f1score, test_f1score]
    }
    
    return evaluation_param
    
class evaluation_list():
    def __init__(self, MRI_series_this='T1', begin=1):
        self.train_list = {} 
        self.validation_list = {}
        self.test_list = {}

        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']
        for i in evalution_params:
            self.train_list[i] = np.array([])
            self.validation_list[i] = np.array([])
            self.test_list[i] = np.array([])
        
        self.begin = begin
        self.MRI_series_this = MRI_series_this

        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        if not os.path.isdir(os.path.join(data_path, self.MRI_series_this)):
            os.mkdir(os.path.join(data_path, self.MRI_series_this))
        for i in ['train', 'validation', 'test', 'ROC']:
            if not os.path.isdir(os.path.join(data_path, self.MRI_series_this, i)):
                os.mkdir(os.path.join(data_path, self.MRI_series_this, i))
    
    def push(self, evaluation_dict):

        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']

        for i in evalution_params:
            self.train_list[i] = np.append(self.train_list[i], evaluation_dict[i][0])
            self.validation_list[i] = np.append(self.validation_list[i], evaluation_dict[i][1])
            self.test_list[i] = np.append(self.test_list[i], evaluation_dict[i][2])

    def single_plot(self, epoch):

        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']

        for mode in ['train', 'validation', 'test']:
            for i in evalution_params:
                param = '\'' + i + '\''
                length = len(eval(f'self.{mode}_list[{param}]'))
                x = np.arange(
                    start=epoch - length + 1, stop=epoch + 1, step=1)
                fig = plt.figure()
                plt.plot(x, eval(f'self.{mode}_list[{param}]'))
                plt.title(f'epoch{x[0]} - epoch{epoch} {i}')
                plt.ylabel(i)
                plt.xlabel('epoch')
                plt.savefig(os.path.join(data_path, self.MRI_series_this, mode, f'epoch{x[0]}_epoch{epoch}_{i}.png'))
                plt.close(fig)

    def all_plot(self, epoch):

        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']

        for i in evalution_params:
            for mode in ['train', 'validation', 'test']:
                param = '\'' + i + '\''
                length = len(eval(f'self.{mode}_list[{param}]'))
                x = np.arange(
                    start=epoch - length + 1, stop=epoch + 1, step=1)
                fig = plt.figure()
                plt.plot(x, eval(f'self.{mode}_list[{param}]'))

            plt.title(f'epoch{x[0]} - epoch{epoch} {i}')
            plt.ylabel(i)
            plt.xlabel('epoch')
            plt.legend(title='Lines', loc='upper right', labels=['train', 'validation', 'test'])
            plt.savefig(os.path.join(data_path, self.MRI_series_this, f'epoch{x[0]}_epoch{epoch}_{i}.png'))
            plt.close(fig)
        
    def single_log(self, epoch):

        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']

        for mode in ['train', 'validation', 'test']:
            for i in evalution_params:
                if not os.path.isfile(os.path.join(data_path, self.MRI_series_this, mode, f'log_{i}.txt')):
                    f = open(os.path.join(data_path, self.MRI_series_this, mode, f'log_{i}.txt'), "x")
                    f.close()

                f = open(os.path.join(data_path, self.MRI_series_this, mode, f'log_{i}.txt'), "a")

                param = '\'' + i + '\''
                length = len(eval(f'self.{mode}_list[{param}]'))
                x = np.arange(
                    start=epoch - length + 1, stop=epoch + 1, step=1)
                f.write(f'{i} epoch{x[0]} - epoch{epoch}: \n\n')
                for j in range(len(x)):
                    val = eval(f'self.{mode}_list[{param}')[j]
                    f.write(f'epoch{x[j]}: {val}\n')
                f.write('-------------------------------------------------------------------------------\n\n')
                f.close()

    def all_log(self, epoch):

        evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']

        if not os.path.isfile(os.path.join(data_path, self.MRI_series_this, f'log.txt')):
            f = open(os.path.join(data_path, self.MRI_series_this, f'log.txt'), "x")
            f.close()

        f = open(os.path.join(data_path, self.MRI_series_this, f'log.txt'), "a")

        length = len(self.train_list['loss'])
        x = np.arange(
            start=epoch - length + 1, stop=epoch + 1, step=1)

        f.write(f'{i} epoch{x[0]} - epoch{epoch}: \n\n')

        for i, epoch_now in enumerate(x):
            f.write('epoch{epoch_now}:\n')
            toWrite = {
                'loss':[], 
                'accuracy': [], 
                'precision': [],
                'sensitivity': [],
                'specificity': [],
                'balance_accuracy': [],
                'IoU': [],
                'f1score': []
                }
            for param in evalution_params:
                for mode in ['train', 'validation', 'test']:
                    val = eval(f'self.{mode}_list[{param}')[i]
                    toWrite[param].append(val)

            toWrite = pd.DataFrame(data=toWrite, index=['train', 'validation', 'test'])
            f.write(toWrite.to_string())
            f.write('\n\n')

        f.write('-------------------------------------------------------------------------------\n\n')
        f.close()

def updata_monitor(train_path, validation_path, test_path):
    thresold = ROC_to_calculate_thresold(pred_path=train_path[2], truth_path=train_path[1])

    for mode in ['train', 'validation', 'test']:
        for i in os.listdir(eval('f{mode}_path[0]')):
            img = cv2.imread(os.path.join(eval('f{mode}_path[0]'), i), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(os.path.join(eval('f{mode}_path[1]'), i), cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(os.path.join(eval('f{mode}_path[2]'), i), cv2.IMREAD_GRAYSCALE)
            pred[pred >= thresold] = 255
            pred[pred < thresold] = 0

            monitor_img = cv2.hconcat([img, msk, pred])
            cv2.imwrite(os.path.join(eval('f{mode}_path[3]'), i), monitor_img)



if __name__ == '__main__':
    pass