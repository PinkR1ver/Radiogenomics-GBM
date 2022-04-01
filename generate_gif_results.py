import os
import imageio
import sys

MRI_series_this = sys.argv[1]
epoch = sys.argv[2]


basePath = r''
resultPath = os.path.join(basePath, 'data', 'assessment', MRI_series_this, epoch, 'predict_mask')
gifSavePath = os.path.join(basePath, 'data', 'assessment', MRI_series_this, epoch, 'segmentation_gif')

if not os.path.isdir(gifSavePath):
    os.mkdir(gifSavePath)

if __name__ == '__main__':
    f = open(os.path.join(gifSavePath, 'log.txt'), "w")
    f.close()
    images = []
    filenames = os.listdir(resultPath)
    subFilenames = []
    for i in range(len(filenames) - 1):
        tmp_1 = filenames[i].split('_')[-1].replace('.png', '')
        tmp_2 = filenames[i + 1].split('_')[-1].replace('.png', '')
        tmp_1_Series = filenames[i].split('_')[0] + filenames[i].split('_')[1] + filenames[i].split('_')[2]
        tmp_2_Series = filenames[i + 1].split('_')[0] + filenames[i + 1].split('_')[1] + filenames[i + 1].split('_')[2]
        if int(tmp_2) - int(tmp_1) == 1 and tmp_1_Series == tmp_2_Series:
            subFilenames.append(filenames[i])
            #print(filenames[i])
        elif len(subFilenames) > 20:
            for filename in subFilenames:
                images.append(imageio.imread(os.path.join(resultPath, filename)))
            saveName = subFilenames[0].replace('_' + subFilenames[0].split('_')[-1]  ,'.gif')
            imageio.mimsave(os.path.join(gifSavePath, saveName), images)
            f = open(os.path.join(gifSavePath, 'log.txt'), "a")
            f.write(f'{saveName} have:\n')
            for filename in subFilenames:
                f.write(f'{filename}\n')
            f.write('---------------------------------------------------------------\n\n')
            f.close()
            images = []
            subFilenames = []
        else: 
            images = []
            subFilenames = []
    print('Done')
            
        
        



