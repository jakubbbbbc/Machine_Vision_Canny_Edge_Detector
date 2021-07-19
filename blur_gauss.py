#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Jakub Ciemiega
MatrNr: 12005481
"""

import cv2
import numpy as np


def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################
    # Write your own code here

    # calculate kernel size
    k_size = 2 * round(3 * sigma) + 1
    # k_size = 39

    # create the gaussian kernel according to the slides
    g_kernel = np.zeros((k_size, k_size))
    for x in range(k_size):
        for y in range(k_size):
            g_kernel[x][y] = 1 / (2 * np.pi * sigma ** 2) \
                             * np.exp(-((x - (k_size - 1) / 2) ** 2 + (y - (k_size - 1) / 2) ** 2) / (2 * sigma ** 2))

    # sum of the kernel must be 1
    g_kernel /= g_kernel.sum()

    # use transparent border type so that it doesn't affect the image
    img_blur = cv2.filter2D(img, -1, g_kernel, cv2.BORDER_TRANSPARENT)

    # helper_functions.plot_row_intensities(img_blur, 100, "1.2_30")

    ######################################################
    return img_blur
