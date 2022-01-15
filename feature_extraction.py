import os
import pandas as pd
from six import u
from data import *
from radiomics import featureextractor
import SimpleITK as sitk

def ROI_not_one_dim(image_array):
    flag_i = False
    flag_j = False
    for val in (85, 170):
        image_array[image_array == val] = 255
    for i in range(image_array.shape[0] - 1):
        for j in range(image_array.shape[1] - 1):
            if(image_array[i + 1, j] == 255 and image_array[i, j] == 255):
                flag_i = True
                if(flag_i and flag_j):
                    return True
            if(image_array[i, j] == 255 and image_array[i, j + 1] == 255):
                flag_j = True
                if(flag_i and flag_j):
                    return True
    return False

basePath = r''
dataPath = os.path.join(basePath, 'data')
dataFile = 'GBM_MRI_Dataset.csv'

if __name__ == '__main__':
    params = os.path.join(dataPath, "Params.yaml")
    FeatureDataset = FeatureExtractionDataset(dataPath, dataFile)
    i = FeatureDataset[30]
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        imageName = os.path.join(dataPath, (i['ImagePath'].loc[0]).replace('\\', '/'))
        maskName = os.path.join(dataPath, (i['MaskPath'].loc[0]).replace('\\', '/'))
    elif platform.system() == 'Windows':
        imageName = os.path.join(dataPath, i['ImagePath'].loc[0])
        maskName = os.path.join(dataPath, i['MaskPath'].loc[0])
    #print(imageName)
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    mask = sitk.ReadImage(maskName)
    mask_arr = sitk.GetArrayFromImage(mask)
    flag = 0
    if np.any(mask_arr):
        for val in (85, 170):
            mask_arr[mask_arr == val] = 255

        mask_merged = sitk.GetImageFromArray(mask_arr)
        mask_merged.CopyInformation(mask)

        feature = extractor.execute(imageName, mask_merged)
        feature_row = pd.DataFrame({})
        for key, val in feature.items():
            if 'diagnostics' not in key:
                feature_row[key.replace('original_', '')] = pd.Series(val)
        
        result = pd.concat([i, feature_row], axis=1)
        result_all = pd.DataFrame({}, columns=result.columns)

    for i in FeatureDataset:
        imageName = os.path.join(dataPath, (i['ImagePath'].loc[0]))
        maskName = os.path.join(dataPath, (i['MaskPath'].loc[0]))
        #print(imageName)
        extractor = featureextractor.RadiomicsFeatureExtractor(params)

        mask = sitk.ReadImage(maskName)
        mask_arr = sitk.GetArrayFromImage(mask)
        if np.any(mask_arr) and ROI_not_one_dim(mask_arr):
            for val in (85, 170):
                mask_arr[mask_arr == val] = 255

            mask_merged = sitk.GetImageFromArray(mask_arr)
            mask_merged.CopyInformation(mask)

            feature = extractor.execute(imageName, mask_merged)
            feature_row = pd.DataFrame({})
            for key, val in feature.items():
                if 'diagnostics' not in key:
                    feature_row[key.replace('original_', '')] = pd.Series(val)
            
            result = pd.concat([i, feature_row], axis=1)
            result_all = result_all.append(result, ignore_index=True)
            flag += 1
            print(flag)
                
    result_all.to_csv(os.path.join(dataPath, 'feature_extraction.csv'), index=False)

    '''
    image = np.array([[255,255],[0,0],[255,255],[255,0]])
    print(if_ROI_one_dim(image))
    '''