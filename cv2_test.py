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

def mySubtract(imgLast, imgNext, Height, Width):
    for i in range(Height):
        for j in range(Width):
            if np.all(imgLast[i, j] == 254):
                imgNext[i, j] = 0



    return imgNext





img1 = cv2.imread("resources/map1.pgm",0)
img2 = cv2.imread("resources/map2.pgm",0)

imgHeight, imgWidth = img1.shape


subtracted_img = mySubtract(img1, img2, imgHeight, imgWidth)
subtracted_img = erode_image(subtracted_img)

subtracted_img = cv2.cvtColor(subtracted_img, cv2.COLOR_GRAY2BGR)

cv2.imshow('My Image',subtracted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
























'''
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




































