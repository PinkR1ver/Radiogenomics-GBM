import os
import numpy as np
import pandas as pd
import trainHelper
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

basePath = os.path.dirname(__file__)
dataPath = os.path.join(basePath, 'data')
train_or_test = ['train_monitor_image_AX', 'test_image_AX']
MRI_series_choose = ['T1', 'T2', 'FLAIR', 'Stack']
parameter_monitor_choose = ['loss_monitor', 'sensitivity_monitor', 'specificity_monitor']

train_loss_csv = pd.DataFrame({})
train_sensitivity_csv = pd.DataFrame({})
train_specificity_csv = pd.DataFrame({})

test_loss_csv = pd.DataFrame({})
test_sensitivity_csv = pd.DataFrame({})
test_specificity_csv = pd.DataFrame({})

for dir_layer1 in train_or_test:
    for dir_layer2 in MRI_series_choose:
        for dir_layer3 in parameter_monitor_choose:
            file_list = os.listdir(os.path.join(dataPath, dir_layer1, dir_layer2, dir_layer1.split('_')[0] + '_' + dir_layer3))
            file_list = list(filter(lambda x : 'log_' in x, file_list))
            file_list = sorted_alphanumeric(file_list)
            data_list = trainHelper.trainHelper(list=np.array([]), averageList=np.array([]), MRI_series_this=dir_layer2, monitor_para=dir_layer3.replace('_monitor', ''), train_or_test=dir_layer1.split('_')[0], begin=1)
            for file_name in file_list:
                f = open(os.path.join(dataPath, dir_layer1, dir_layer2, dir_layer1.split('_')[0] + '_' + dir_layer3, file_name), 'r')
                while True:
                    line = f.readline()

                    if not line:
                        break

                    if re.match(r'^-?\d+(?:\.\d+)$', line) is not None:
                        data_list.list_pushback(float(line))

                data_list.averageList_pushback()
                data_list.clear_list()

            para_series = pd.Series(data_list.averageList, name=dir_layer1.split('_')[0] + '_' + dir_layer2 + '_' + dir_layer3.replace('_monitor', ''))
            eval(f'{data_list.train_or_test}_{data_list.monitor_para}_csv')[para_series.name] = para_series
            data_list.clear_averageList()

for name_part1 in train_or_test:
    for name_part2 in parameter_monitor_choose:
        t_or_t = name_part1.split('_')[0]
        para_which = name_part2.replace('_monitor', '')
        eval(f'{t_or_t}_{para_which}_csv').to_csv(os.path.join(dataPath ,f'{t_or_t}_{para_which}_togther.csv'))

            



