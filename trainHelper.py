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
import pandas as pd
from tqdm import tqdm

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, 'data', 'result')
train_path = os.path.join(data_path, 'tmp', 'train')
test_path = os.path.join(data_path, 'tmp', 'test')
validation_path = os.path.join(data_path, 'tmp', 'validation')

def sendmail(content, subject):
    # setting mail server information
    mail_host = 'smtp.gmail.com'

    # password
    mail_pass = '585Goo521'

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


def ROC_to_calculate_thresold(preds, truths, save_path=None, save_or_not=False):

    # print(torch.unique(preds))

    FPR = np.array([])
    TPR = np.array([])
    G_mean_flag = (0, 0)
    G_mean_max = 0
    threshold_flag = 0

    for threshold in tqdm(np.arange(0, 1, 0.002), desc='Calculating ROC curve'):
        FP, FN, TP, TN = 0, 0, 0, 0

        for i in range(preds.size(dim=0)):
            tmp_pred = torch.squeeze(preds[i].cpu()).detach().clone().numpy()
            tmp_pred[tmp_pred >= threshold] = 1
            tmp_pred[tmp_pred < threshold] = 0

            tmp_truth = torch.squeeze(truths[i].cpu()).detach().clone().numpy()


            FP += len(np.where(tmp_pred - tmp_truth == 1)[0])
            FN += len(np.where(tmp_pred - tmp_truth == -1)[0])
            TP += len(np.where(tmp_pred + tmp_truth == 2)[0])
            TN += len(np.where(tmp_pred + tmp_truth == 0)[0])
        
        # print(threshold)
        # print(FP, FN, TP, TN)
        
        if FP + TN != 0 and TP + FN != 0 and TP + TN != 0 and TN + FP != 0:
            FPR = np.append(FPR, FP / (FP + TN))
            TPR = np.append(TPR, TP / (TP + FN))
            recall = TP / (TP + FN)
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
        plt.xlim((-0.02, 1.02))
        plt.ylim((-0.02, 1.02))
        plt.xlabel('False Positive Rate')
        plt.ylabel('Ture Positeve Rate')
        plt.annotate(f'Max G-mean with {G_mean_max}, threshold={threshold_flag}', G_mean_flag)
        plt.savefig(save_path)
        plt.close(fig)

    return threshold_flag

def f1score_to_calculate_thresold(preds, truths, save_path=None, save_or_not=False):

    # print(torch.unique(preds))

    f1score = np.array([])
    f1score_flag = (0, 0)
    f1score_max = 0
    threshold_flag = 0
    x = np.array([])

    for threshold in tqdm(np.arange(0, 1, 0.002), desc='Calculating ROC curve'):
        FP, FN, TP, TN = 0, 0, 0, 0

        for i in range(preds.size(dim=0)):
            tmp_pred = torch.squeeze(preds[i].cpu()).detach().clone().numpy()
            tmp_pred[tmp_pred >= threshold] = 1
            tmp_pred[tmp_pred < threshold] = 0

            tmp_truth = torch.squeeze(truths[i].cpu()).detach().clone().numpy()


            FP += len(np.where(tmp_pred - tmp_truth == 1)[0])
            FN += len(np.where(tmp_pred - tmp_truth == -1)[0])
            TP += len(np.where(tmp_pred + tmp_truth == 2)[0])
            TN += len(np.where(tmp_pred + tmp_truth == 0)[0])
        
        # print(threshold)
        # print(FP, FN, TP, TN)
        
        if 2 * TP + FP + FN != 0:
            x = np.append(x, threshold)
            f1score = np.append(f1score, (2 * TP) / (2 * TP + FP + FN))
            if f1score[-1] > f1score_max:
                f1score_max = f1score[-1]
                f1score_flag = (threshold, f1score[-1])
                threshold_flag = threshold

    if save_or_not:
        fig = plt.figure()
        plt.plot(x, f1score)
        plt.title('Threshold Tuning Curve')
        plt.xlim((-0.02, 1.02))
        plt.ylim((-0.02, 1.02))
        plt.xlabel('Threshold')
        plt.ylabel('F1-score')
        plt.annotate(f'Max F1-score with {f1score_max}, \nthreshold={threshold_flag}', f1score_flag)
        plt.savefig(save_path)

    return threshold_flag


def evalation_all(train_preds, train_truths, validation_preds, validation_truths, test_preds, test_truths, threshold):
    train_preds_copy = train_preds.clone()

    train_preds_copy[train_preds_copy >= threshold] = 1
    train_preds_copy[train_preds_copy < threshold] = 0


    FP, FN, TP, TN = 0, 0, 0, 0

    for i in range(len(train_preds)):
        train_pred_arr = torch.squeeze(train_preds_copy[i].cpu()).detach().clone().numpy()
        train_truth_arr = torch.squeeze(train_truths[i].cpu()).detach().clone().numpy()

        FP += len(np.where(train_pred_arr - train_truth_arr == 1)[0])
        FN += len(np.where(train_pred_arr - train_truth_arr == -1)[0])
        TP += len(np.where(train_pred_arr + train_truth_arr == 2)[0])
        TN += len(np.where(train_pred_arr + train_truth_arr == 0)[0])


    train_accuracy = (TP + TN) / (TP + TN + FN + FP)
    train_sensitivity = TP / (TP + FN)
    train_specificity = TN / (TN + FP)
    train_precision = TP / (TP + FP)
    train_balance_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
    train_IoU = TP / (TP + FN + FP)
    train_f1score = (2 * TP) / (2 * TP + FP + FN)

    gc.collect()

    validation_preds_copy = validation_preds.clone()

    validation_preds_copy[validation_preds_copy >= threshold] = 1
    validation_preds_copy[validation_preds_copy < threshold] = 0


    FP, FN, TP, TN = 0, 0, 0, 0

    for i in range(len(validation_preds)):
        validation_pred_arr = torch.squeeze(validation_preds_copy[i].cpu()).detach().clone().numpy()
        validation_truth_arr = torch.squeeze(validation_truths[i].cpu()).detach().clone().numpy()

        FP += len(np.where(validation_pred_arr - validation_truth_arr == 1)[0])
        FN += len(np.where(validation_pred_arr - validation_truth_arr == -1)[0])
        TP += len(np.where(validation_pred_arr + validation_truth_arr == 2)[0])
        TN += len(np.where(validation_pred_arr + validation_truth_arr == 0)[0])


    validation_accuracy = (TP + TN) / (TP + TN + FN + FP)
    validation_sensitivity = TP / (TP + FN)
    validation_specificity = TN / (TN + FP)
    validation_precision = TP / (TP + FP)
    validation_balance_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
    validation_IoU = TP / (TP + FN + FP)
    validation_f1score = (2 * TP) / (2 * TP + FP + FN)

    gc.collect()

    test_preds_copy = test_preds.clone()

    test_preds_copy[test_preds_copy >= threshold] = 1
    test_preds_copy[test_preds_copy < threshold] = 0


    FP, FN, TP, TN = 0, 0, 0, 0

    for i in range(len(test_preds)):
        test_pred_arr = torch.squeeze(test_preds_copy[i].cpu()).detach().clone().numpy()
        test_truth_arr = torch.squeeze(test_truths[i].cpu()).detach().clone().numpy()

        FP += len(np.where(test_pred_arr - test_truth_arr == 1)[0])
        FN += len(np.where(test_pred_arr - test_truth_arr == -1)[0])
        TP += len(np.where(test_pred_arr + test_truth_arr == 2)[0])
        TN += len(np.where(test_pred_arr + test_truth_arr == 0)[0])


    test_accuracy = (TP + TN) / (TP + TN + FN + FP)
    test_sensitivity = TP / (TP + FN)
    test_specificity = TN / (TN + FP)
    test_precision = TP / (TP + FP)
    test_balance_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
    test_IoU = TP / (TP + FN + FP)
    test_f1score = (2 * TP) / (2 * TP + FP + FN)

    gc.collect()

    evaluation_param = {
        "accuracy": [train_accuracy, validation_accuracy, test_accuracy],
        "sensitivity": [train_sensitivity, validation_sensitivity, test_sensitivity],
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
        for i in ['train', 'validation', 'test']:
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
            fig = plt.figure()
            for mode in ['train', 'validation', 'test']:
                param = '\'' + i + '\''
                length = len(eval(f'self.{mode}_list[{param}]'))
                x = np.arange(
                    start=epoch - length + 1, stop=epoch + 1, step=1)
                plt.plot(x, eval(f'self.{mode}_list[{param}]'))

            plt.title(f'epoch{x[0]} - epoch{epoch} {i}')
            plt.ylabel(i)
            plt.xlabel('epoch')
            plt.legend(title='Lines', loc='best', labels=['train', 'validation', 'test'])
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
                    val = eval(f'self.{mode}_list[{param}]')[j]
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

        f.write(f'epoch{x[0]} - epoch{epoch}: \n\n')

        for i, epoch_now in enumerate(x):
            f.write(f'epoch{epoch_now}:\n')
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
                param_str = '\'' + param + '\''
                for mode in ['train', 'validation', 'test']:
                    val = eval(f'self.{mode}_list[{param_str}]')[i]
                    toWrite[param].append(val)

            toWrite = pd.DataFrame(data=toWrite, index=['train', 'validation', 'test'])
            f.write(toWrite.to_string())
            f.write('\n\n')

        f.write('-------------------------------------------------------------------------------\n\n')
        f.close()

'''
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
'''



if __name__ == '__main__':
    evl_dict = evalation_all(train_path=[os.path.join(train_path, 'Preds'), os.path.join(train_path, 'Masks')], validation_path=[os.path.join(validation_path, 'Preds'), os.path.join(validation_path, 'Masks')], test_path=[os.path.join(test_path, 'Preds'), os.path.join(test_path, 'Masks')], ROC_cruve_save_path='./ROC.png')
    print(evl_dict)