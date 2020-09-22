#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    contains all the standalone utility functions for this project.
"""

__author__ = 'Rishab Pal'

import cv2


def draw_face_box(img, pt1, pt2, color, thickness, radius):
    """Draws and Displays Face border"""

    x1, y1 = pt1
    x2, y2 = pt2
    # Difference
    d = (int((x2 - x1) / 4))

    # Top left
    cv2.line(img, (x1 + radius, y1), (x1 + radius + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y1 + radius + d), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - radius, y1), (x2 - radius - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y1 + radius + d), color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + radius, y2), (x1 + radius + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - radius), (x1, y2 - radius - d), color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - radius, y2), (x2 - radius - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - radius), (x2, y2 - radius - d), color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)