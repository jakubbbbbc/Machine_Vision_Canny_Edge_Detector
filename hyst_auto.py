#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: Jakub Ciemiega
MatrNr: 12005481
"""

import cv2
import numpy as np

from hyst_thresh import hyst_thresh


def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high and low thresholds of standard
    hysteresis threshold. low_prop is the proportion of edge pixels which are above the low threshold and high_prop is
    the proportion of pixels above the high threshold.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################

    # calculate thresholds based on the proportion values
    l_thres = np.percentile(edges_in, (1.-low_prop)*100)
    h_thres = np.percentile(edges_in, (1.-high_prop)*100)

    # call the hysteresis threshold function
    hyst_out = hyst_thresh(edges_in, l_thres, h_thres)

    ######################################################
    return hyst_out
