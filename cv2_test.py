#!/usr/bin/python3

# -- implement RRT algorithm on a image

import cv2
import numpy as np
import networkx as nx
import random as rd
import math
from scipy.optimize import fsolve
import time

def erode_image(map_array, filter_size = 1):
    """ Erodes the image to reduce the chances of robot colliding with the wall
    each pixel is 0.05 meter. The robot is 30 cm wide, that is 0.3 m. Half is
    0.15 m. If we increase the walls by 20 cm on either side, the kernel should
    be 40 cm wide. 0.4 / 0.05 = 8
    """
    kernel = np.ones((filter_size,filter_size), np.uint8)
    eroded_img = cv2.erode(map_array, kernel, iterations = 1)
    return eroded_img



img = []

'''
for i in range(8):
    temp_img = cv2.imread('map_sequences/map' + str(1+i) + '.png')
    binary= cv2.threshold(temp_img, 206, 255, cv2.THRESH_BINARY)[1]
    binary.astype(np.uint8)
    img.append(binary)

for i in range(8):

    if i > 0:
        imgLast = img[i-1]
        subtracted = cv2.subtract(img[i], imgLast)
        subtracted = erode_image(subtracted)
        cv2.imwrite("map_sequences/delta_pgm_example_" + str(i + 1) + ".png", subtracted)

    else:
        map_data = erode_image(img[i])
        cv2.imwrite("map_sequences/delta_pgm_example_" + str(i + 1) + ".png", map_data)

    cv2.imwrite("map_sequences/pgm_to_binary_" + str(1+i) + ".png", img[i])
'''



img.append(cv2.imread('map_sequences/graphed_sequences/PRM_lifetime_delta_example_1.png'))
img.append(cv2.imread('map_sequences/graphed_sequences/PRM_lifetime_example_frontier6.png'))
img.append(cv2.imread('map_sequences/graphed_sequences/PRM_lifetime_delta_example_6.png'))
#img.append(cv2.imread('map_sequences/pgm_to_binary_6.png'))
#img.append(cv2.imread('map_sequences/delta_pgm_example_6.png'))





for i in range(3):
    img[i] = img[i][100:410, 150:425]

cv2.imwrite('map_sequences/cropped_images/cropped_PRM_lifetime_delta_example_1.png', img[0])
cv2.imwrite('map_sequences/cropped_images/cropped_PRM_lifetime_example_frontier6.png', img[1])
cv2.imwrite('map_sequences/cropped_images/cropped_PRM_lifetime_delta_example_6.png', img[2])


imgHeight, imgWidth, channels = img[0].shape
print("Image Height: " + str(imgHeight))
print("Image Width: " + str(imgWidth))


#cv2.imshow('My Image', img[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()



'''
img = []

for i in range(8):
    img.append(cv2.imread('resources/map' + str(i+1) + '.pgm'))

for i in range(8):
    cv2.imwrite('map_sequences/map' + str(i+1) + '.png', img[i])
'''























'''
for i in range(8):
    img.append(cv2.imread('resources/ma
imgLast = None
for i in range(5):

    if i > 0:
        imgLast = img[i-1]
        subtracted = cv2.subtract(img[i], imgLast)
        subtracted = erode_image(subtracted)
        cv2.imwrite("map_sequences/delta_example_" + str(i + 1) + ".png", subtracted)

    else:
        map_data = erode_image(img[i])
        cv2.imwrite("map_sequences/delta_example_" + str(i + 1) + ".png", map_data)
'''




































