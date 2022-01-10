import os
import numpy as np
import cv2
from numpy.core.fromnumeric import shape
from skimage import io
from matplotlib import pyplot as plt
import math
from node import *

def if_point_in_corner(masks, i, j):
    if i ==0 or i == masks.shape[0] - 1 or j == 0 or j == masks.shape[1] - 1:
        return True
    else:
        return False

def odd_or_even(num):
    if (num % 2 == 0):
        return 0
    else:
        return 1

'''
def point_in_which_side(masks, i, j):
    if i == 0 and j == 0:
        return 1
    elif i != 0 and i != masks.shape[0] - 1 and j == 0:
        return 2
    elif i == masks.shape[0] - 1 and j == 0:
        return 3
    elif i == masks.shape[0] - 1 and j != 0 and j != masks.shape[1] - 1:
        return 4
    elif i == masks.shape[0] -1 and j == masks.shape[1] - 1:
        return 5
    elif i != 0 and i!= masks.shape[0] - 1 and j == masks.shape[1] - 1:
        return 6
    elif i == 0 and j == masks.shape[1] - 1:
        return 7
    elif i == 0 and j != masks.shape[1] - 1 and j!= 0:
        return 8
'''

def extract_boundary(masks):
    offset = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
    boundary = np.zeros(masks.shape, dtype=np.uint8)
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            if masks[i, j] == 255:
                if not if_point_in_corner(masks, i, j):
                    flag = 0
                    for k in range(offset.shape[0]):
                        if masks[i + offset[k, 0], j+ offset[k, 1]] != 255:
                            flag = 1
                            break
                    if flag:
                        boundary[i, j] = 255       
                else:
                    boundary[i, j] = 255
                    '''
                    if point_in_which_side(masks, i, j) == 1:
                        flag = 0
                        direction = [0, 1, 2]
                        for k in direction:
                            if masks[i + offset[k, 0], j + offset [k, 1]] != 255:
                                flag = 1
                                break
                        if flag:
                            boundary[i, j] = 255
                    elif  point_in_which_side(masks, i, j) == 2:
                        flag = 0
                        direction = [0, 1, 2, 3, 4]
                        for k in direction:
                            if masks[i + offset[k, 0], j + offset [k, 1]] != 255:
                                flag = 1
                                break
                        if flag:
                            boundary[i, j] = 255
                    elif  point_in_which_side(masks, i, j) == 3:
                        flag = 0
                        direction = [2, 3, 4]
                        for k in direction:
                            if masks[i + offset[k, 0], j + offset [k, 1]] != 255:
                                flag = 1
                                break
                        if flag:
                            boundary[i, j] = 255
                    elif  point_in_which_side(masks, i, j) == 4:
                        flag = 0
                        direction = [2, 3, 4, 5, 6]
                        for k in direction:
                            if masks[i + offset[k, 0], j + offset [k, 1]] != 255:
                                flag = 1
                                break
                        if flag:
                            boundary[i, j] = 255
                    elif  point_in_which_side(masks, i, j) == 5:
                        flag = 0
                        direction = [4, 5, 6]
                        for k in direction:
                            if masks[i + offset[k, 0], j + offset [k, 1]] != 255:
                                flag = 1
                                break
                        if flag:
                            boundary[i, j] = 255
                    elif  point_in_which_side(masks, i, j) == 6:
                        flag = 0
                        direction = [4, 5, 6, 7, 0]
                        for k in direction:
                            if masks[i + offset[k, 0], j + offset [k, 1]] != 255:
                                flag = 1
                                break
                        if flag:
                            boundary[i, j] = 255
                    elif  point_in_which_side(masks, i, j) == 7:
                        flag = 0
                        direction = [6, 7, 0]
                        for k in direction:
                            if masks[i + offset[k, 0], j + offset [k, 1]] != 255:
                                flag = 1
                                break
                        if flag:
                            boundary[i, j] = 255
                    elif  point_in_which_side(masks, i, j) == 8:
                        flag = 0
                        direction = [6, 7, 0, 1, 2]
                        for k in direction:
                            if masks[i + offset[k, 0], j + offset [k, 1]] != 255:
                                flag = 1
                                break
                        if flag:
                            boundary[i, j] = 255
                    '''        
    return boundary


def extract_bit_quads(masks):
    bit_quads = np.array([0, 0, 0, 0, 0]) #Q1,Q2,Q3,Q4,QD
    offset = np.array([[1, 0], [1, 1], [0, 1]])
    masks = np.pad(masks, 2)
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            if i + 1 < masks.shape[0] and j + 1 < masks.shape[1]:
                if (masks[i ,j] == 255 and masks[i + offset[1, 0], j + offset[1, 1]] == 255 and masks[i + offset[0, 0], j+ offset[0, 1]] == 0 and masks[i + offset[2, 0], j+ offset[2, 1]] == 0) or (masks[i ,j] == 0 and masks[i + offset[1, 0], j + offset[1, 1]] == 0 and masks[i + offset[0, 0], j+ offset[0, 1]] == 255 and masks[i + offset[2, 0], j+ offset[2, 1]] == 255):
                    bit_quads[4] +=1
                else:
                    sum_of_one = 0
                    if(masks[i ,j] == 255):
                        sum_of_one += 1
                    for k in range(3):
                        if(masks[i + offset[k, 0], j + offset[k, 1]] == 255):
                            sum_of_one +=1
                    if sum_of_one == 0:
                        pass
                    else:
                        bit_quads[sum_of_one - 1] += 1
    return bit_quads

def area_of_image_by_bit_quads_gray(masks):
    # This method to calculate area of a pixel area invented by Gray and Pratt, called Bit Quads
    bit_quads = extract_bit_quads(masks)
    # area = 1 / 4 * bit_quads[0] + 1 / 2 * bit_quads[1] + 7 / 8 * bit_quads[2] + bit_quads[3] + 3 / 4 * bit_quads[4] # Pratt's method
    area = 1 / 4 * (bit_quads[0] + 2 * bit_quads[1] + 3 * bit_quads[2] + 4 * bit_quads[3] + 2 * bit_quads[4]) # Gray's method 
    return area

def perimeter_of_image_by_bit_quads_gray(masks):
    # This method to calculate perimeter of a pixel area invented by Gray and Pratt, called Bit Quads
    bit_quads = extract_bit_quads(masks)
    # perimeter = bit_quads[1] + 1 / math.sqrt(2) * (bit_quads[0] + bit_quads[2] + 2 * bit_quads[4]) # Pratt's method
    perimeter = bit_quads[0] + bit_quads[1] + bit_quads[2] + 2 * bit_quads[4] # Gray's method 
    return perimeter

def area_of_image_by_bit_quads_pratt(masks):
    # This method to calculate area of a pixel area invented by Gray and Pratt, called Bit Quads
    bit_quads = extract_bit_quads(masks)
    area = 1 / 4 * bit_quads[0] + 1 / 2 * bit_quads[1] + 7 / 8 * bit_quads[2] + bit_quads[3] + 3 / 4 * bit_quads[4] # Pratt's method
    # area = 1 / 4 * (bit_quads[0] + 2 * bit_quads[1] + 3 * bit_quads[2] + 4 * bit_quads[3] + 2 * bit_quads[4]) # Gray's method 
    return area

def perimeter_of_image_by_bit_quads_pratt(masks):
    # This method to calculate perimeter of a pixel area invented by Gray and Pratt, called Bit Quads
    bit_quads = extract_bit_quads(masks)
    perimeter = bit_quads[1] + 1 / math.sqrt(2) * (bit_quads[0] + bit_quads[2] + 2 * bit_quads[4]) # Pratt's method
    # perimeter = bit_quads[0] + bit_quads[1] + bit_quads[2] + 2 * bit_quads[4] # Gray's method 
    return perimeter

def extract_chain_code_Freeman(masks):
    # Freeman Chain code
    # Considering the numpy i is column direction and j is row direction, the chain_code direction is different from essay, you need to find it.
    chain_code = SingleLinkList()
    #chain_code = []
    offset_xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    offset_diagonal = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) 
    boundary = extract_boundary(masks)
    #flag = np.zeros([boundary.shape[0], boundary.shape[1]])
    for i in range(boundary.shape[0]):
        for j in range(boundary.shape[1]):
            if boundary[i, j] == 255:
                temp_chain_code = np.array([], dtype=np.uint8)
                end_point = np.array([i, j])
                boundary[i, j] = 0 #flag[i, j] = 1
                while True:
                    mark = 0
                    for k in range(4):
                        if i + offset_xy[k, 0] >= 0  and i + offset_xy[k, 0] < boundary.shape[0] and j + offset_xy[k, 1] >=0 and j + offset_xy[k, 1] < boundary.shape[1]:
                            if boundary[i + offset_xy[k, 0], j + offset_xy[k, 1]] == 255:
                                temp_chain_code = np.append(temp_chain_code, k * 2)
                                boundary[i + offset_xy[k, 0], j + offset_xy[k, 1]] = 0 #flag[i + offset_xy[k, 0], j + offset_xy[k, 1]] = 1
                                i = i + offset_xy[k, 0]
                                j = j + offset_xy[k, 1]
                                mark = 1
                                break
                    if not mark:
                        for k in range(4):
                            if i + offset_diagonal[k, 0] >= 0  and i + offset_diagonal[k, 0] < boundary.shape[0] and j + offset_diagonal[k, 1] >=0 and j + offset_diagonal[k, 1] < boundary.shape[1]:
                                if boundary[i + offset_diagonal[k, 0], j + offset_diagonal[k, 1]] == 255:
                                    boundary[i + offset_diagonal[k, 0], j + offset_diagonal[k, 1]] = 0
                                    temp_chain_code = np.append(temp_chain_code, k * 2 + 1)
                                    i = i + offset_diagonal[k, 0]
                                    j = j + offset_diagonal[k, 1]
                                    mark = 1
                                    break
                    if not mark:
                        offset = end_point - np.array([i, j])
                        for k in range(4):
                            if offset_xy[k, 0] == offset[0] and offset_xy[k, 1] == offset[1]:
                                temp_chain_code = np.append(temp_chain_code, k * 2)
                        for k in range(4):
                            if offset_diagonal[k, 0] == offset[0] and offset_diagonal[k, 1] == offset[1]:
                                temp_chain_code = np.append(temp_chain_code, k * 2 + 1)
                        chain_code.append(temp_chain_code)
                        break
    return chain_code

def perimeter_of_boundary_by_chain_code(masks):
    chain_code = extract_chain_code_Freeman(masks)
    sum_even = 0
    sum_odd = 0 
    perimeter = 0
    for i in chain_code.items():
        for j in i:
            if(odd_or_even(j)):
                sum_odd += 1
            else:
                sum_even +=1
        perimeter += sum_even
        perimeter += math.sqrt(2) * sum_odd
        sum_odd = 0
        sum_even = 0
    return perimeter

def area_of_image_by_chain_code(masks):
    chain_code = extract_chain_code_Freeman(masks)
    area = 0
    for i in chain_code.items():
        y_i = 0
        for j in i:
            if j == 1 or j == 2 or j == 3:
                c_i_x = 1
            elif j == 0 or j== 4:
                c_i_x = 0
            else:
                c_i_x = -1
            
            if j == 1 or j == 0 or j == 7:
                c_i_y = 1
            elif j == 2 or j == 6:
                c_i_y = 0
            else:
                c_i_y = -1
            
            area += c_i_x * (y_i + c_i_y / 2)
            y_i += c_i_y
    
    return area
            
def diameter_of_boundary(masks):
    diameter = 0
    location = SingleLinkList()
    offset_xy = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    offset_diagonal = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) 
    boundary = extract_boundary(masks)
    for i in range(boundary.shape[0]):
        for j in range(boundary.shape[1]):
            if boundary[i, j] == 255:
                temp_location = np.array([[i, j]], dtype=np.uint8)
                boundary[i, j] = 0
                while True:
                    mark = 0
                    for k in range(4):
                        if i + offset_xy[k, 0] >= 0  and i + offset_xy[k, 0] < boundary.shape[0] and j + offset_xy[k, 1] >=0 and j + offset_xy[k, 1] < boundary.shape[1]:
                            if boundary[i + offset_xy[k, 0], j + offset_xy[k, 1]] == 255:
                                boundary[i + offset_xy[k, 0], j + offset_xy[k, 1]] = 0
                                i = i + offset_xy[k, 0]
                                j = j + offset_xy[k, 1]
                                temp_location = np.append(temp_location, [[i, j]], axis=0)
                                mark = 1
                                break
                    if not mark:
                        for k in range(4):
                            if i + offset_diagonal[k, 0] >= 0  and i + offset_diagonal[k, 0] < boundary.shape[0] and j + offset_diagonal[k, 1] >=0 and j + offset_diagonal[k, 1] < boundary.shape[1]:
                                if boundary[i + offset_diagonal[k, 0], j + offset_diagonal[k, 1]] == 255:
                                    boundary[i + offset_diagonal[k, 0], j + offset_diagonal[k, 1]] = 0
                                    i = i + offset_diagonal[k, 0]
                                    j = j + offset_diagonal[k, 1]
                                    temp_location = np.append(temp_location, [[i, j]], axis=0)
                                    mark = 1
                                    break
                    if not mark:
                        location.append(temp_location)
                        break
    for i in location.items():
        for j in i:
            for k in i:
                distance = math.sqrt(sum((j - k) ** 2))
                diameter = max(diameter, distance)
    return diameter



            
           

if __name__ == '__main__':
    image = cv2.imread(r'sample/TCGA_CS_6666_20011109_16_mask.tif', cv2.IMREAD_GRAYSCALE)
    image = np.array(image)
    io.imshow(image)
    plt.show()

    white = np.zeros(image.shape, dtype=np.uint8)
    for i in range(4, white.shape[0] - 4):
        for j in range(4, white.shape[1] - 4):
            white[i, j] = 255

    boundary = extract_boundary(image)
    io.imshow(boundary)
    plt.show()

    io.imshow(white)
    plt.show()

    white_boundary = extract_boundary(white)

    io.imshow(white_boundary)
    plt.show()

    print(area_of_image_by_bit_quads_gray(image))
    print(perimeter_of_image_by_bit_quads_gray(image))
    print(area_of_image_by_bit_quads_pratt(image))
    print(perimeter_of_image_by_bit_quads_pratt(image))

    experiment = np.array([[0, 255], [255, 255]], dtype=np.uint8)
    io.imshow(experiment)
    plt.show()
    bit = extract_bit_quads(experiment)
    print(bit)
    print(area_of_image_by_bit_quads_gray(experiment))
    print(perimeter_of_image_by_bit_quads_gray(experiment))

    print(area_of_image_by_bit_quads_pratt(experiment))
    print(perimeter_of_image_by_bit_quads_pratt(experiment))

    chain_code = extract_chain_code_Freeman(experiment)
    for i in chain_code.items():
        print(i)
    exp = np.array([[255]])
    chain_code_exp = extract_chain_code_Freeman(exp)
    for i in chain_code_exp.items():
        print(i)

    chain_code_exp2 = extract_chain_code_Freeman(image)
    for i in chain_code_exp2.items():
        print(i)

    exp2 = np.array([[0, 255, 255, 0, 0, 0, 0], [255, 0, 0, 255, 0, 0, 0], [255, 0, 0, 0, 255, 255, 255], [255, 0, 0, 0, 0, 0, 255], [0, 255, 0, 255, 255, 255, 0], [0, 255, 255, 0, 0, 0, 0]])
    io.imshow(exp2)
    plt.show()
    chain_code_exp3 = extract_chain_code_Freeman(exp2)
    for i in chain_code_exp3.items():
        print(i)
    
    io.imshow(exp2)
    plt.show()

    print(perimeter_of_boundary_by_chain_code(experiment))
    print(perimeter_of_boundary_by_chain_code(image))
    
    print(perimeter_of_boundary_by_chain_code(exp2))
    print(perimeter_of_image_by_bit_quads_gray(exp2))
    print(perimeter_of_image_by_bit_quads_pratt(exp2))

    print(perimeter_of_boundary_by_chain_code(image))
    print(perimeter_of_image_by_bit_quads_gray(image))
    print(perimeter_of_image_by_bit_quads_pratt(image))

    print(area_of_image_by_bit_quads_gray(image))
    print(area_of_image_by_bit_quads_pratt(image))
    print(area_of_image_by_chain_code(image))

    print(diameter_of_boundary(image))
    print(diameter_of_boundary(exp2))
    print(diameter_of_boundary(experiment))