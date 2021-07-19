#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: Jakub Ciemiega
MatrNr: 12005481
"""

import cv2
import numpy as np


def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """
    ######################################################
    # Write your own code here

    # create kernel for x axis
    s_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_x = cv2.filter2D(img.copy(), -1, s_kernel, cv2.BORDER_TRANSPARENT)

    # y is increasing up
    gradient_y = cv2.filter2D(img.copy(), -1, -s_kernel.T, cv2.BORDER_TRANSPARENT)

    # calculate gradient and its direction
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    orientation = np.arctan2(gradient_y, gradient_x)

    ######################################################
    return gradient, orientation
