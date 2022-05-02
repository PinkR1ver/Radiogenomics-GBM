import os
from re import S
from cv2 import threshold
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
import torchvision
from data import *
from unet import *
import numpy as np
import traceback
import trainHelper
import sys
from tqdm import tqdm
import gc

MRI_series_this = sys.argv[1]
epoches = int(sys.argv[2])

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, 'data')
result_path = os.path.join(data_path, 'result', MRI_series_this)
model_path = os.path.join(base_path, 'model', MRI_series_this)
weight_path = os.path.join(model_path, f'{MRI_series_this}_unet.pth')
ROC_path = os.path.join(result_path, 'ROC_curve')
monitor_path = os.path.join(result_path, 'monitor')
image_path = os.path.join(data_path, 'Images')
mask_path = os.path.join(data_path, 'Masks')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if  not os.path.isdir(os.path.join(data_path, 'result')):
    os.mkdir(os.path.join(data_path, 'result'))

if not os.path.isdir(result_path):
    os.mkdir(result_path)

if not os.path.isdir(ROC_path):
    os.mkdir(ROC_path)

if not os.path.isdir(monitor_path):
    os.mkdir(monitor_path)

for mode in ['train', 'validation', 'test']:
    if not os.path.isdir(os.path.join(monitor_path, mode)):
        os.mkdir(os.path.join(monitor_path, mode))

if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")


if __name__ == '__main__':

    train_dataset = NiiDataset(img_path=image_path, msk_path=mask_path, MRI_series=MRI_series_this, mode='train', resize=(256, 256, 128))

    # training_data_size = 0.8

    # train_size = int(training_data_size * len(GBM_Dataset))
    # validation_size = len(GBM_Dataset) - train_size

    # train_dataset, validation_dataset = torch.utils.data.random_split(
    #   GBM_Dataset, [train_size, validation_size])

    test_dataset = NiiDataset(img_path=image_path, msk_path=mask_path, MRI_series=MRI_series_this, mode='test', resize=(256, 256, 128))

    batch_size = 2

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    net = UNet_3D().to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("Loading Weight Successful")
    else:
        print("Loading Weight Failed")

    opt = optim.Adam(net.parameters())  # stochastic gradient descent
    loss_function = nn.BCELoss()

    f = open(os.path.join(model_path, f'{MRI_series_this}_epoch.txt'), "r")
    epoch = int(f.read())
    f.close()

    evl_list = trainHelper.evaluation_list(MRI_series_this, epoch)
    loss_list = np.array([])
    average_loss_list = np.array([])

    try:
        threshold = 0.5
        for iter_out in range(1, epoches + 1):

            # train_preds = torch.tensor([])
            # train_truths = torch.tensor([])
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for i, (image, mask, original_mask) in tqdm(enumerate(train_loader), desc=f"train_{epoch}", total=len(train_loader)):
                image, mask, original_mask = image.to(device), mask.to(device), original_mask.to(device)

                predict_image = net(image)
                train_loss = loss_function(predict_image, mask)
                loss_list = np.append(loss_list, train_loss.item())

                opt.zero_grad()
                train_loss.backward()
                opt.step()


                resize_predict_image = np.empty((0, original_mask[0].shape[1], original_mask[0].shape[2], original_mask[0].shape[3]))
                # print(resize_predict_image.shape)

                for j in range(predict_image.shape[0]):
                    # print(original_mask[i].shape[1:])
                    # print(torch.squeeze(predict_image[i].clone()).cpu().detach().numpy().shape)
                    resize_predict_image = np.append(resize_predict_image, np.expand_dims(resize_3d_image(torch.squeeze(predict_image[j].clone()).cpu().detach().numpy(), original_mask[j].shape[1:], mode='nearest'), 0), axis=0)
                    # print(resize_predict_image.shape)

                resize_predict_image[resize_predict_image >= threshold] = 1
                resize_predict_image[resize_predict_image < threshold] = 0

                original_mask = torch.squeeze(original_mask).cpu().detach().numpy()


                # print(resize_predict_image.shape)
                # print(original_mask.shape)


                FP += len(np.where(resize_predict_image - original_mask == 1)[0])
                FN += len(np.where(resize_predict_image - original_mask == -1)[0])
                TP += len(np.where(resize_predict_image + original_mask == 2)[0])
                TN += len(np.where(resize_predict_image + original_mask == 0)[0])


                # print(resize_predict_image.shape)

                # train_preds = torch.cat((train_preds, resize_predict_image.cpu().detach()), 0)
                # train_truths = torch.cat((train_truths, original_mask.cpu().detach()), 0)


            train_accuracy = (TP + TN) / (TP + TN + FN + FP)
            train_sensitivity = TP / (TP + FN)
            train_specificity = TN / (TN + FP)
            train_precision = TP / (TP + FP)
            train_balance_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
            train_IoU = TP / (TP + FN + FP)
            train_f1score = (2 * TP) / (2 * TP + FP + FN)

            average_loss_list = np.append(
                average_loss_list, loss_list.sum() / len(loss_list))
            loss_list = np.array([])

            gc.collect()


            TP = 0
            FP = 0
            TN = 0
            FN = 0
            # test_preds = torch.tensor([])
            # test_truths = torch.tensor([])
            for i, (image, mask, original_mask) in tqdm(enumerate(test_loader), desc=f"test_{epoch}", total=len(test_loader)):
                image, mask, original_mask = image.to(device), mask.to(device), original_mask.to(device)

                predict_image = net(image)
                train_loss = loss_function(predict_image, mask)
                loss_list = np.append(loss_list, train_loss.item())

                resize_predict_image = np.empty((0, original_mask[0].shape[1], original_mask[0].shape[2], original_mask[0].shape[3]))
                # print(resize_predict_image.shape)

                for j in range(predict_image.shape[0]):
                    # print(original_mask[i].shape[1:])
                    # print(torch.squeeze(predict_image[i].clone()).cpu().detach().numpy().shape)
                    resize_predict_image = np.append(resize_predict_image, np.expand_dims(resize_3d_image(torch.squeeze(predict_image[j].clone()).cpu().detach().numpy(), original_mask[j].shape[1:], mode='nearest'), 0), axis=0)
                    # print(resize_predict_image.shape)

                resize_predict_image[resize_predict_image >= threshold] = 1
                resize_predict_image[resize_predict_image < threshold] = 0

                original_mask = torch.squeeze(original_mask).cpu().detach().numpy()


                # print(resize_predict_image.shape)
                # print(original_mask.shape)


                FP += len(np.where(resize_predict_image - original_mask == 1)[0])
                FN += len(np.where(resize_predict_image - original_mask == -1)[0])
                TP += len(np.where(resize_predict_image + original_mask == 2)[0])
                TN += len(np.where(resize_predict_image + original_mask == 0)[0])

                # test_preds = torch.cat((test_preds, resize_predict_image.cpu().detach()), 0)
                # test_truths = torch.cat((test_truths, original_mask.cpu().detach()), 0)


            test_accuracy = (TP + TN) / (TP + TN + FN + FP)
            test_sensitivity = TP / (TP + FN)
            test_specificity = TN / (TN + FP)
            test_precision = TP / (TP + FP)
            test_balance_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2
            test_IoU = TP / (TP + FN + FP)
            test_f1score = (2 * TP) / (2 * TP + FP + FN)

            average_loss_list = np.append(
                average_loss_list, loss_list.sum() / len(loss_list))
            loss_list = np.array([])

            gc.collect()
            

            # print(torch.unique(train_preds))
            # evl_dict = trainHelper.evalation_all(train_preds, train_truths, test_preds, test_truths, threshold)
            
            evl_dict = {
                "accuracy": [train_accuracy, test_accuracy],
                "sensitivity": [train_sensitivity, test_sensitivity],
                "specificity": [train_specificity, test_specificity],
                "precision": [train_precision, test_precision],
                "balance_accuracy": [train_balance_accuracy, test_balance_accuracy],
                "IoU": [train_IoU, test_IoU],
                "f1score": [train_f1score, test_f1score]
            }
    
            evl_dict['loss'] = average_loss_list
            evl_list.push(evl_dict)

            average_loss_list = np.array([])


            if epoch % 25 == 0:
                torch.save(net.state_dict(), os.path.join(
                    model_path, f'{MRI_series_this}_epoch_{epoch}.pth'))

            torch.save(net.state_dict(), weight_path)

            # print(torch.unique(train_preds))

            # if epoch % 10 == 0:
                # threshold = trainHelper.ROC_to_calculate_thresold(train_preds, train_truths, os.path.join(ROC_path, f'epoch{epoch}.png'), True)
                # threshold = trainHelper.f1score_to_calculate_thresold(train_preds, train_truths, os.path.join(ROC_path, f'epoch{epoch}.png'), True)

            epoch += 1

            f = open(os.path.join(
                model_path, f'{MRI_series_this}_epoch.txt'), "w")
            f.write(f'{epoch}')
            f.close()

            print('--------------------------------------------------\n\n')

            gc.collect()

    except:
        print('Exception!!!')
        if not os.path.isfile(os.path.join(base_path, 'exception_in_trainning', f'{MRI_series_this}_log.txt')):
            f = open(os.path.join(base_path, 'exception_in_trainning',
                     f'{MRI_series_this}_log.txt'), 'x')
            f.close()
        f = open(os.path.join(base_path, 'exception_in_trainning',
                 f'{MRI_series_this}_log.txt'), 'a')
        f.write(f'exception in epoch{epoch}\n')
        f.write(traceback.format_exc())
        f.write('\n')
        f.close()

        torch.save(net.state_dict(), weight_path)

        if iter_out != 1:
            evl_list.single_plot(epoch)
            evl_list.all_plot(epoch)
            evl_list.single_log(epoch)
            evl_list.all_log(epoch)

        trainHelper.sendmail(
            content=r'Your train.py went something wrong' + '\n' traceback.format_exc()+ , subject=r'train.py go wrong')

    else:
        print("Train finishing")

        evl_list.single_plot(epoch)
        evl_list.all_plot(epoch)
        evl_list.single_log(epoch)
        evl_list.all_log(epoch)

        trainHelper.sendmail(
            content=r'Your train.py run success', subject=r'train.py finished')
