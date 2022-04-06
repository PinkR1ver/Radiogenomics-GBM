import readline
import trainHelper
import os
import numpy as np
import shutil

replot_list = trainHelper.evaluation_list('T2', 1, os.path.dirname(__file__))

evalution_params = ['loss', 'accuracy', 'precision', 'sensitivity', 'specificity', 'balance_accuracy', 'IoU', 'f1score']

for mode in ['train', 'validation', 'test']:
    for log in os.listdir(f'./{mode}'):
        if 'log' in log:
            for param in evalution_params:
                if param in log:
                    if param == 'accuracy':
                        if 'balance_accuracy' not in log:
                            f = open(os.path.join(f'./{mode}', log), "r")
                            f.readline()
                            f.readline()
                            lines = f.readlines()
                            for line in lines:
                                if 'epoch' in line:
                                    eval(f'replot_list.{mode}_list')[param] = np.append(eval(f'replot_list.{mode}_list')[param], float(line.split(' ')[-1]))
                            f.close()
                    else:
                        f = open(os.path.join(f'./{mode}', log), "r")
                        f.readline()
                        f.readline()
                        lines = f.readlines()
                        for line in lines:
                            if 'epoch' in line:
                                eval(f'replot_list.{mode}_list')[param] = np.append(eval(f'replot_list.{mode}_list')[param], float(line.split(' ')[-1]))
                        f.close()


replot_list.single_plot(101)
replot_list.all_plot(101)
replot_list.single_log(101)
replot_list.all_log(101)