#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Jakub Ciemiega
MatrNr: 12005481
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################

    # define thresholds as int values
    l_thres = int(low * 255)
    h_thres = int(high * 255)
    # print(l_thres, h_thres)

    edges = (edges_in.copy() * 255).astype(np.uint8)
    # np.set_printoptions(threshold=np.inf)
    # print(np.amax(edges))
    # get positions of pixels with value above h_thres
    high_positions = np.where(edges > h_thres)

    # get labels of connected edges that have at least one pixel above h_thres
    retval, labels = cv2.connectedComponents(edges, connectivity=8)
    valid_labels = np.unique(labels[high_positions[0], high_positions[1]])
    # print(len(valid_labels))

    # discard invalid edges (those which don't have any pixel above h_thres)
    edges = np.where(np.isin(labels, valid_labels), edges, 0)

    # low threshold, create bit image
    edges = np.where(edges > l_thres, 1, 0)

    bitwise_img = edges.astype(np.float32)

    ######################################################
    return bitwise_img
