from torch.utils import data
import ShapeBasedFeatures as SBF
import TextureBasedFeatures as TBF
import IntensityBasedFeatures as IBF
import os
from data import *
from utils import *
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

basePath = r'C:\Users\RTX 3090\Desktop\WangYichong\U-net for Ivy Gap'
dataPath = os.path.join(basePath, 'data')

if __name__ == '__main__':
    FeatureTable = pd.DataFrame(
        {
            "Patient": [],
            "Plane": [],
            "MRISeries": [],
            "Slice": [],
            "ImagePath": [],
            "MaskPath": [],
            "Mean_of_image": [],
            "Variance_of_image": [],
            "StandardDeviation_of_image": [],
            "Skewness_of_image": [],
            "Kurtosis_of_image": [],
            "Energy_of_image": [],
            "Entropy_of_image": [],
            "Contrast_of_image": [],
            "Homogeneity_of_image": [],
            "Mean_of_GLCM_i": [],
            "Mean_of_GLCM_j": [],
            "Variance_of_GLCM_i": [],
            "Variance_of_GLCM_j": [],
            "Correlation_of_image": [],
            "Energy_of_GLCM": [],
            "Entropy_of_GLCM": [],
            "Dissimilarity_of_image": [],
            "Area_of_image_by_bitquads_Gray": [],
            "Area_of_image_by_bitquads_Partt": [],
            "Area_of_image_by_chain_code": [],
            "Perimeter_of_image_by_bitquads_Gray": [],
            "Perimeter_of_image_by_bitquads_Partt": [],
            "Perimeter_of_image_by_chain_code": [],
        }
    )
    GBMDataset = ImageDataSet(dataPath, 'GBM_MRI_Dataset.csv')
    for i in range(len(GBMDataset)):
        Patient = (GBMDataset.AxInfo.iloc[i]).Patient
        Plane = (GBMDataset.AxInfo.iloc[i]).Plane
        MRISeries = (GBMDataset.AxInfo.iloc[i]).MRISeries
        Slice = (GBMDataset.AxInfo.iloc[i]).Slice
        ImagePath = (GBMDataset.AxInfo.iloc[i]).ImagePath
        MaskPath = (GBMDataset.AxInfo.iloc[i]).MaskPath
        image = Image.open(os.path.join(dataPath, ImagePath))
        image = ImageOps.grayscale(image)
        image = np.array(image)
        mask = Image.open(os.path.join(dataPath, MaskPath))
        mask = ImageOps.grayscale(mask)
        mask = gray2Binary(mask)
        mask = np.array(mask)
        GLCM_of_image = TBF.gray_level_co_occurence_matrix(
            image=image, image_depth=256, levels=256, offset=[0, 1], symmetric=True, masks=mask)
        Mean_of_image = IBF.mean_of_image(image, mask, 'GRAY')
        Variance_of_image = IBF.variance_of_image(image, mask, 'GRAY')
        StandardDeviation_of_image = IBF.standardDeviation_of_image(
            image, mask, 'GRAY')
        Skewness_of_image = IBF.skewness_of_image(image, mask, 'GRAY')
        Kurtosis_of_image = IBF.kurtosis_of_image(image, mask, 'GRAY')
        Energy_of_image = IBF.energy_of_image(image, mask)
        Entropy_of_image = IBF.entropy_of_image(image, mask)
        Contrast_of_image = TBF.contrast_of_image(GLCM_of_image)
        Homogeneity_of_image = TBF.homogeneity_of_image(GLCM_of_image)
        Mean_of_GLCM_i = TBF.glcm_i_mean_of_image(GLCM_of_image)
        Mean_of_GLCM_j = TBF.glcm_j_mean_of_image(GLCM_of_image)
        Variance_of_GLCM_i = TBF.glcm_i_variance_of_image(GLCM_of_image)
        Variance_of_GLCM_j = TBF.glcm_j_variance_of_image(GLCM_of_image)
        Corrlation_of_image = TBF.correlation_of_image(GLCM_of_image)
        Energy_of_GLCM = TBF.glcm_energy_of_image(GLCM_of_image)
        Entropy_of_GLCM = TBF.glcm_entropy_of_image(GLCM_of_image)
        Dissimilarity_of_image = TBF.dissimilarity_of_image(GLCM_of_image)
        Area_of_image_by_bitquads_Gray = SBF.area_of_image_by_bit_quads_gray(
            mask)
        Area_of_image_by_bitquads_Pratt = SBF.area_of_image_by_bit_quads_pratt(
            mask)
        Area_of_image_by_chain_code = SBF.area_of_image_by_chain_code(mask)
        Perimeter_of_image_by_bitquads_Gray = SBF.perimeter_of_image_by_bit_quads_gray(
            mask)
        Perimeter_of_image_by_bitquads_Pratt = SBF.perimeter_of_image_by_bit_quads_pratt(
            mask)
        Perimeter_of_image_by_chain_code = SBF.perimeter_of_boundary_by_chain_code(
            mask)

        FeatureTableRow = pd.DataFrame([[Patient, Plane, MRISeries, Slice, ImagePath, MaskPath, Mean_of_image, Variance_of_image, StandardDeviation_of_image, Skewness_of_image, Kurtosis_of_image, Entropy_of_image, Entropy_of_image, Contrast_of_image, Homogeneity_of_image, Mean_of_GLCM_i, Mean_of_GLCM_j, Variance_of_GLCM_i, Variance_of_GLCM_j,
                                       Corrlation_of_image, Energy_of_GLCM, Entropy_of_GLCM, Dissimilarity_of_image, Area_of_image_by_bitquads_Gray, Area_of_image_by_bitquads_Pratt, Area_of_image_by_chain_code, Perimeter_of_image_by_bitquads_Gray, Perimeter_of_image_by_bitquads_Pratt, Perimeter_of_image_by_chain_code]], columns=FeatureTable.columns)

        FeatureTable = FeatureTable.append(
            FeatureTableRow, ignore_index=True)

    FeatureTable.to_csv(os.paht.join(dataPath, 'GBM_MRI_Feature.csv'), index=False)