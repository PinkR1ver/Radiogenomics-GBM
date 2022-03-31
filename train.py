import os
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

MRI_series_this = sys.argv[1]
epoches = int(sys.argv[2])

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, 'data')
result_path = os.path.join(data_path, 'result', MRI_series_this)
model_path = os.path.join(base_path, 'model', MRI_series_this)
weight_path = os.path.join(model_path, f'{MRI_series_this}_unet.pth')
train_path = os.path.join(result_path, 'tmp', 'train')
test_path = os.path.join(result_path, 'tmp', 'test')
validation_path = os.path.join(result_path, 'tmp', 'validation')
ROC_path = os.path.join(result_path, 'ROC_curve')

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(ROC_path):
    os.mkdir(ROC_path)

if not os.pardir(result_path):
    os.mkdir(result_path)

if not os.path.isdir(train_path):
    os.mkdir(train_path)

if not os.path.isdir(train_path, 'Images'):
    os.mkdir(train_path, 'Images')

if not os.path.isdir(train_path, 'Masks'):
    os.mkdir(train_path, 'Masks')

if not os.path.isdir(train_path, 'Preds'):
    os.mkdir(train_path, 'Preds')

if not os.path.isdir(train_path, 'Monitor'):
    os.mkdir(train_path, 'Monitor')

if not os.path.isdir(validation_path, 'Images'):
    os.mkdir(validation_path, 'Images')

if not os.path.isdir(validation_path, 'Masks'):
    os.mkdir(validation_path, 'Masks')

if not os.path.isdir(validation_path, 'Preds'):
    os.mkdir(validation_path, 'Preds')

if not os.path.isdir(validation_path, 'Monitor'):
    os.mkdir(validation_path, 'Monitor')

if not os.path.isdir(test_path, 'Images'):
    os.mkdir(test_path, 'Images')

if not os.path.isdir(test_path, 'Masks'):
    os.mkdir(test_path, 'Masks')

if not os.path.isdir(test_path, 'Preds'):
    os.mkdir(test_path, 'Preds')

if not os.path.isdir(test_path, 'Monitor'):
    os.mkdir(test_path, 'Monitor')

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

    batch_size = 4

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

    evl_list = trainHelper.evaluation_list()
    loss_list = np.array([])
    average_loss_list = np.array([])

    try:
        for iter_out in range(1, epoches + 1):
            for i, (image, mask) in tqdm(enumerate(train_loader), desc=f"train_{epoch}", total=len(train_loader)):
                image, mask = image.to(device), mask.to(device)

                predict_image = net(image)
                train_loss = loss_function(predict_image, mask)
                loss_list = np.append(loss_list, train_loss)

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                if i % 5 == 0:
                    print(f'{epoch}-{i} train loss=====>>{train_loss.item()}')

                for j in image.shape[0]:
                    _image = image[j]
                    _mask = mask[j]
                    _pred_Image = predict_image[j]

                    torchvision.utils.save_image(_image, os.path.join(
                        train_path, 'Images', f'{i * batch_size + j}.png'))
                    torchvision.utils.save_image(_mask, os.path.join(
                        train_path, 'Masks', f'{i * batch_size + j}.png'))
                    torchvision.utils.save_image(_pred_Image, os.path.join(
                        train_path, 'Preds', f'{i * batch_size + j}.png'))

            average_loss_list = np.append(
                average_loss_list, loss_list.sum() / len(loss_list))
            loss_list = np.array([])

            for i, (image, mask) in tqdm(enumerate(validation_loader), desc=f"validation_{epoch}", total=len(validation_loader)):
                image, mask = image.to(device), mask.to(device)

                predict_image = net(image)
                train_loss = loss_function(predict_image, mask)

                if i % 5 == 0:
                    print(f'{epoch}-{i} validation loss=====>>{train_loss.item()}')

                for j in image.shape[0]:
                    _image = image[j]
                    _mask = mask[j]
                    _pred_Image = predict_image[j]

                    torchvision.utils.save_image(_image, os.path.join(
                        validation_path, 'Images', f'{i * batch_size + j}.png'))
                    torchvision.utils.save_image(_mask, os.path.join(
                        validation_path, 'Masks', f'{i * batch_size + j}.png'))
                    torchvision.utils.save_image(_pred_Image, os.path.join(
                        validation_path, 'Preds', f'{i * batch_size + j}.png'))

            average_loss_list = np.append(
                average_loss_list, loss_list.sum() / len(loss_list))
            loss_list = np.array([])

            for i, (image, mask) in tqdm(enumerate(test_loader), desc=f"test_{epoch}", total=len(test_loader)):
                image, mask = image.to(device), mask.to(device)

                predict_image = net(image)
                train_loss = loss_function(predict_image, mask)

                if i % 5 == 0:
                    print(f'{epoch}-{i} test loss=====>>{train_loss.item()}')

                for j in image.shape[0]:
                    _image = image[j]
                    _mask = mask[j]
                    _pred_Image = predict_image[j]

                    torchvision.utils.save_image(_image, os.path.join(
                        test_path, 'Images', f'{i * batch_size + j}.png'))
                    torchvision.utils.save_image(_mask, os.path.join(
                        test_path, 'Masks', f'{i * batch_size + j}.png'))
                    torchvision.utils.save_image(_pred_Image, os.path.join(
                        test_path, 'Preds', f'{i * batch_size + j}.png'))
            
            average_loss_list = np.append(
                average_loss_list, loss_list.sum() / len(loss_list))
            loss_list = np.array([])
            
            
            evl_dict = trainHelper.evalation_all(train_path=[os.path.join(train_path, 'Preds'), os.path.join(train_path, 'Masks')], validation_path=[os.path.join(validation_path, 'Preds'), os.path.join(validation_path, 'Masks')], test_path=[os.path.join(test_path, 'Preds'), os.path.join(test_path, 'Masks')], ROC_cruve_save_path=ROC_path)
            evl_dict['loss'] = average_loss_list
            evl_list.push(evl_dict)

            average_loss_list = np.array([])

            if epoch % 25 == 0:
                torch.save(net.state_dict(), os.path.join(
                    model_path, f'{MRI_series_this}_epoch_{epoch}.pth'))

            torch.save(net.state_dict(), weight_path)

            if epoch % 20 == 0:
                monitor_train_path = []
                monitor_validation_path = []
                monitor_test_path = []
                for mode in ['Images', 'Masks', 'Preds', 'Monitor']:
                    monitor_train_path.append(os.path.join(train_path, mode))
                    monitor_validation_path.append(os.path.join(validation_path, mode))
                    monitor_test_path.append(os.path.join(test_path, mode))
                trainHelper.updata_monitor(train_path=monitor_train_path, validation_path=monitor_validation_path, test_path=monitor_test_path)

            epoch += 1

            f = open(os.path.join(
                model_path, f'{MRI_series_this}_epoch.txt'), "w")
            f.write(f'{epoch}')
            f.close()

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
