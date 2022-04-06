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

    GBM_Dataset = ImageDataset(data_path, 'GBM_MRI_Dataset.csv',
                               MRI_series=MRI_series_this, mode='train', resize=(256, 256))

    training_data_size = 0.8

    train_size = int(training_data_size * len(GBM_Dataset))
    validation_size = len(GBM_Dataset) - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(
        GBM_Dataset, [train_size, validation_size])

    test_dataset = ImageDataset(data_path, 'GBM_MRI_Dataset.csv',
                                MRI_series=MRI_series_this, mode='test', resize=(256, 256))

    batch_size = 2

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    net = UNet().to(device)

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

            train_preds = torch.tensor([])
            train_truths = torch.tensor([])
            for i, (image, mask) in tqdm(enumerate(train_loader), desc=f"train_{epoch}", total=len(train_loader)):
                image, mask = image.to(device), mask.to(device)

                predict_image = net(image)
                train_loss = loss_function(predict_image, mask)
                loss_list = np.append(loss_list, train_loss.item())

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                train_preds = torch.cat((train_preds, predict_image.cpu().detach()), 0)
                train_truths = torch.cat((train_truths, mask.cpu().detach()), 0)


                _image = image[0]
                _mask = mask[0]
                _pred_image = predict_image[0]

                _pred_image[_pred_image >= threshold] = 1
                _pred_image[_pred_image < threshold] = 0

                vaisual_image = torch.stack([_image, _mask, _pred_image], dim=0)
                torchvision.utils.save_image(vaisual_image, os.path.join(monitor_path, 'train', f'{i}.png'))


            average_loss_list = np.append(
                average_loss_list, loss_list.sum() / len(loss_list))
            loss_list = np.array([])

            gc.collect()

            validation_preds = torch.tensor([])
            validation_truths = torch.tensor([])
            for i, (image, mask) in tqdm(enumerate(validation_loader), desc=f"validation_{epoch}", total=len(validation_loader)):
                image, mask = image.to(device), mask.to(device)

                predict_image = net(image)
                train_loss = loss_function(predict_image, mask)
                loss_list = np.append(loss_list, train_loss.item())

                validation_preds = torch.cat((validation_preds, predict_image.cpu().detach()), 0)
                validation_truths = torch.cat((validation_truths, mask.cpu().detach()), 0)

                _image = image[0]
                _mask = mask[0]
                _pred_image = predict_image[0]

                _pred_image[_pred_image >= threshold] = 1
                _pred_image[_pred_image < threshold] = 0

                vaisual_image = torch.stack([_image, _mask, _pred_image], dim=0)
                torchvision.utils.save_image(vaisual_image, os.path.join(monitor_path, 'validation', f'{i}.png'))

            average_loss_list = np.append(
                average_loss_list, loss_list.sum() / len(loss_list))
            loss_list = np.array([])

            gc.collect()


            test_preds = torch.tensor([])
            test_truths = torch.tensor([])
            for i, (image, mask) in tqdm(enumerate(test_loader), desc=f"test_{epoch}", total=len(test_loader)):
                image, mask = image.to(device), mask.to(device)

                predict_image = net(image)
                train_loss = loss_function(predict_image, mask)
                loss_list = np.append(loss_list, train_loss.item())

                test_preds = torch.cat((test_preds, predict_image.cpu().detach()), 0)
                test_truths = torch.cat((test_truths, mask.cpu().detach()), 0)

                _image = image[0]
                _mask = mask[0]
                _pred_image = predict_image[0]

                _pred_image[_pred_image >= threshold] = 1
                _pred_image[_pred_image < threshold] = 0

                vaisual_image = torch.stack([_image, _mask, _pred_image], dim=0)
                torchvision.utils.save_image(vaisual_image, os.path.join(monitor_path, 'test', f'{i}.png'))
            
            average_loss_list = np.append(
                average_loss_list, loss_list.sum() / len(loss_list))
            loss_list = np.array([])

            gc.collect()
            
            
            # print(torch.unique(train_preds))
            evl_dict = trainHelper.evalation_all(train_preds, train_truths, validation_preds, validation_truths, test_preds, test_truths, threshold)
            evl_dict['loss'] = average_loss_list
            evl_list.push(evl_dict)

            average_loss_list = np.array([])

            if epoch % 25 == 0:
                torch.save(net.state_dict(), os.path.join(
                    model_path, f'{MRI_series_this}_epoch_{epoch}.pth'))

            torch.save(net.state_dict(), weight_path)

            # print(torch.unique(train_preds))

            if epoch % 10 == 0:
                threshold = trainHelper.ROC_to_calculate_thresold(train_preds, train_truths, os.path.join(ROC_path, f'epoch{epoch}.png'), True)

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
            content=r'Your train.py went something wrong', subject=r'train.py go wrong')

    else:
        print("Train finishing")

        evl_list.single_plot(epoch)
        evl_list.all_plot(epoch)
        evl_list.single_log(epoch)
        evl_list.all_log(epoch)

        trainHelper.sendmail(
            content=r'Your train.py run success', subject=r'train.py finished')
