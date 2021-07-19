#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Jakub Ciemiega
MatrNr: 12005481
"""

import cv2
import numpy as np


def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    # Write your own code here

    # create an array with direction information
    #     horizontal: 1
    #     diag_right: 2
    #     vertical: 3
    #     diag_lef: 4
    directions = np.zeros(gradients.shape)
    pi = np.pi
    directions[(-pi <= orientations) & (orientations < -7 / 8 * pi) |
               (-1 / 8 * pi <= orientations) & (orientations < 1 / 8 * pi) |
               (7 / 8 * pi <= orientations) & (orientations < pi)] = 1
    directions[(-7 / 8 * pi <= orientations) & (orientations < -5 / 8 * pi) |
               (1 / 8 * pi <= orientations) & (orientations < 3 / 8 * pi)] = 2
    directions[(-5 / 8 * pi <= orientations) & (orientations < -3 / 8 * pi) |
               (3 / 8 * pi * pi <= orientations) & (orientations < 5 / 8 * pi)] = 3
    directions[directions == 0] = 4

    # x is vertical and y is horizontal
    x_size = gradients.shape[0]
    y_size = gradients.shape[1]

    # employ the non-maximum suppression algorithm

    # add border to be able to compare
    edges = cv2.copyMakeBorder(gradients, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    edges1 = edges.copy()
    for x in range(1, x_size):
        for y in range(1, y_size):
            # for horizontal
            if directions[x - 1, y - 1] == 1:
                # if (-pi <= orientations[x, y] < -7 / 8 * pi) or (-1 / 8 * pi <= orientations[x, y] < 1 / 8 * pi) \
                #         or (7 / 8 * pi <= orientations[x, y] <= pi):
                if edges[x, y] < edges[x, y - 1] or edges[x, y] < edges[x, y + 1]:
                    edges1[x, y] = 0

            # for diagonal right .->'
            if directions[x - 1, y - 1] == 2:
                # elif (-7 / 8 * pi <= orientations[x, y] < -5 / 8 * pi) or (1 / 8 * pi <= orientations[x, y] < 3 / 8 * pi):
                if edges[x, y] < edges[x + 1, y - 1] or edges[x, y] < edges[x - 1, y + 1]:
                    edges1[x, y] = 0

            # for vertical
            if directions[x - 1, y - 1] == 3:
                # elif (-5 / 8 * pi <= orientations[x, y] < -3 / 8 * pi) or (3 / 8 * pi <= orientations[x, y] < 5 / 8 * pi):
                if edges[x, y] < edges[x - 1, y] or edges[x, y] < edges[x + 1, y]:
                    edges1[x, y] = 0

            # for diagonal left '->.
            else:
                if edges[x, y] < edges[x - 1, y - 1] or edges[x, y] < edges[x + 1, y + 1]:
                    edges1[x, y] = 0

    # remove border
    edges1 = edges1[1:-1, 1:-1]

    # edges1 must contain values from 0.0 to 1.0
    edges1 /= np.amax(edges1)

    ######################################################

    return edges1
