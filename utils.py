import os
import cv2
from skimage import io
from matplotlib import pyplot as plt

from PIL import Image, ImageOps
from PIL import GifImagePlugin
import torch
import numpy as np

import nibabel as nib
import imageio  # transfer nii to image
from PIL import Image


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    longestSide = max(img.size)
    _img = Image.new('RGB', (longestSide, longestSide), (0, 0, 0))
    _img.paste(img, (0, 0))
    _img = _img.resize(size)
    return _img


def keep_image_size_open_gray(path, size=(256, 256)):
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    longestSide = max(img.size)
    mask = Image.new('P', (longestSide, longestSide))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def gray2RGB(img):
    out_img = torch.cat((img, img, img), 0)
    return out_img

def gray2Binary(img):
    img = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:
                img[i, j] = 255
    return img
                

if __name__ == '__main__':
    image = keep_image_size_open_gray(r'C:\Users\RTX 3090\Desktop\WangYichong\Data\Masks\Multi-institutional Paired Expert Segmentations MNI images-atlas-annotations\3_Annotations_MNI\CWRU\W1\W1_1996.10.25_CWRU_labels_AX\108.png')
    gimage = ImageOps.grayscale(image)
    ggimage = np.array(gimage)
    io.imshow(ggimage)
    plt.show()
    print(gimage.size)
    ggimage = gray2Binary(gimage)
    ggimage = np.array(ggimage)
    io.imshow(ggimage)
    plt.show()

def read_nii_image(niifile, plane='AX'):
    img = nib.load(niifile)
    img_fdata = img.get_fdata()

    return img_fdata

if __name__ == '__main__':
    pass