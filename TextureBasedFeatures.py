import os
import numpy as np
import cv2
from skimage import io
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
import math


def ImageCompress(image, image_depth=256, levels=8):
    bins = np.linspace(0, image_depth, levels)
    image_compress = np.digitize(image, bins, True)
    image_compress = np.uint8(image_compress)
    return image_compress


def gray_level_co_occurence_matrix(image, image_depth=256, levels=8, offset=[0, 1], symmetric=False, masks=np.array([])):
    if masks.all():
        image = ImageCompress(image, image_depth, levels)
        GLCM = np.zeros((levels, levels))  # gray levels co-occurence matrix
        if symmetric == False:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if i + offset[0] < image.shape[0] and i + offset[0] >= 0 and j + offset[1] < image.shape[1] and j + offset[1] >= 0:
                        x = image[i, j]
                        y = image[i + offset[0], j + offset[1]]
                        GLCM[x, y] += 1
            return np.uint(GLCM)
        else:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if i + offset[0] < image.shape[0] and i + offset[0] >= 0 and j + offset[1] < image.shape[1] and j + offset[1] >= 0:
                        x = image[i, j]
                        y = image[i + offset[0], j + offset[1]]
                        GLCM[x, y] += 1
                        GLCM[y, x] += 1
            return np.uint(GLCM)
    else:
        image = ImageCompress(image, image_depth, levels)
        GLCM = np.zeros((levels, levels))  # gray levels co-occurence matrix
        if symmetric == False:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if i + offset[0] < image.shape[0] and i + offset[0] >= 0 and j + offset[1] < image.shape[1] and j + offset[1] >= 0:
                        x = image[i, j]
                        y = image[i + offset[0], j + offset[1]]
                        if (masks[i, j] == 1 or masks[i, j] == 255) and (masks[i + offset[0], j + offset[1]] == 1 or masks[i + offset[0], j + offset[1]] == 255):
                            GLCM[x, y] += 1
            return np.uint(GLCM)
        else:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if i + offset[0] < image.shape[0] and i + offset[0] >= 0 and j + offset[1] < image.shape[1] and j + offset[1] >= 0:
                        x = image[i, j]
                        y = image[i + offset[0], j + offset[1]]
                        if (masks[i, j] == 1 or masks[i, j] == 255) and (masks[i + offset[0], j + offset[1]] == 1 or masks[i + offset[0], j + offset[1]] == 255):
                            GLCM[x, y] += 1
                            GLCM[y, x] += 1
            return np.uint(GLCM)


def contrast_of_image(glcm):
    if glcm.sum() != 0:
        norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    contrast = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            contrast += (i - j) ** 2 * norm_glcm[i, j]
    return contrast


def homogeneity_of_image(glcm):
    if glcm.sum() != 0:
        norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    homogeneity = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            homogeneity += norm_glcm[i, j] / (1 + (i - j) ** 2)
    return homogeneity


def glcm_i_mean_of_image(glcm):
    if glcm.sum() != 0:
            norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    mean = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            mean += i * norm_glcm[i, j]
    return mean


def glcm_i_variance_of_image(glcm):
    if glcm.sum() != 0:
            norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    mean = glcm_i_mean_of_image(glcm)
    variance = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            variance += (i - mean) ** 2 * norm_glcm[i, j]
    return variance


def glcm_j_mean_of_image(glcm):
    if glcm.sum() != 0:
            norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    mean = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            mean += j * norm_glcm[i, j]
    return mean


def glcm_j_variance_of_image(glcm):
    if glcm.sum() != 0:
            norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    mean = glcm_j_mean_of_image(glcm)
    variance = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            variance += (j - mean) ** 2 * norm_glcm[i, j]
    return variance


def correlation_of_image(glcm):
    if glcm.sum() != 0:
            norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    correlation = 0
    mean_i = glcm_i_mean_of_image(glcm)
    mean_j = glcm_j_mean_of_image(glcm)
    standardDeviation_i = glcm_i_variance_of_image(glcm) ** (1/2)
    standardDeviation_j = glcm_j_variance_of_image(glcm) ** (1/2)
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            if standardDeviation_i * standardDeviation_j == 0:
                return 0
            else:
                correlation += norm_glcm[i, j] * ((i - mean_i) * (j - mean_j)) / (
                    standardDeviation_i * standardDeviation_j)
    return correlation


def glcm_energy_of_image(glcm):
    if glcm.sum() != 0:
            norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    energy = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            energy += norm_glcm[i, j] ** 2
    return energy


def glcm_entropy_of_image(glcm):
    if glcm.sum() != 0:
            norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    entropy = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            if norm_glcm[i, j] != 0:
                entropy -= norm_glcm[i, j] * math.log(norm_glcm[i, j], 2)
    return entropy


def dissimilarity_of_image(glcm):
    if glcm.sum() != 0:
            norm_glcm = glcm / glcm.sum()
    else:
        norm_glcm = glcm
    dissimilarity = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            dissimilarity += norm_glcm[i, j] * abs(i - j)
    return dissimilarity


def same_or_not(image, other):
    flag = image - other
    is_all_zero = np.all((flag == 0))
    if is_all_zero:
        print("Same Image")
    else:
        print("Not same image")


if __name__ == '__main__':
    levels = 256
    image = cv2.imread('sample/Ivy.jpeg', 0)
    print()
    image_compress = ImageCompress(image, levels=levels)
    same_or_not(image, image_compress)
    stack_image = np.hstack((image, image_compress))
    io.imshow(stack_image)
    plt.show()
    glcm = greycomatrix(image_compress, [1], [0], levels)
    print(glcm.shape)
    print(np.squeeze(np.uint(glcm)))
    for prop in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:

        temp = greycoprops(glcm, prop)

        print(prop, temp)
        print("==============================\n")

    GLCM = gray_level_co_occurence_matrix(image, levels=levels)
    # print(GLCM)
    #print(image.size - np.count_nonzero(image))
    #print(image_compress.size - np.count_nonzero(image_compress))
    print(f'Contrast{contrast_of_image(GLCM)}')
    print(f'Homogeneity:{homogeneity_of_image(GLCM)}')
    print(f'Correlation:{correlation_of_image(GLCM)}')
    print(f'Energy:{glcm_energy_of_image(GLCM)}')
    print(f'Entropy:{glcm_entropy_of_image(GLCM)}')
    print(f'Dissimilarity:{dissimilarity_of_image(GLCM)}')

    test = np.array([[0, 0, 1, 1],
                     [0, 0, 1, 1],
                     [0, 2, 2, 2],
                     [2, 2, 3, 3]], dtype=np.uint8)
    test_glcm = greycomatrix(test, [1], [0], levels=4)
    test_GLCM = gray_level_co_occurence_matrix(test, levels=4, image_depth=4)
    print(np.squeeze(test_glcm))
    print(test_GLCM)
    print(ImageCompress(test, image_depth=4, levels=4))
