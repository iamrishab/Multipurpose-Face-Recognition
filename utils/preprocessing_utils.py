#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'


import os
import cv2
import numpy as np


# adaptive histogram equlaization
def apply_clahe(frame, gridsize=8):
	# converting BRG image to LAB
	lab = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2LAB)
	# splitting the LAB
	lab_planes = cv2.split(lab)
	# crEate the CLAHE image
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
	# apply clahe to L channel (Luminescense)
	lab_planes[0] = clahe.apply(lab_planes[0])
	# merge all the planes
	lab = cv2.merge(lab_planes)
	# converting LAB back to BGR
	bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	# return the bgr image
	return bgr


# histogram equalization for colored image
def apply_histogram_equalization(frame):
	frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
	# equalize the histogram of the Y channel
	frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
	# convert the YUV image back to BGR format
	frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)
	# returning the BGR frame
	return frame


# this function applies 2D rotation
def align_face_eyes(img, eyes):
	# angle between eyes
	dY = eyes[1][1] - eyes[0][1]
	dX = eyes[1][0] - eyes[0][0]
	angle = -np.degrees(np.arctan2(dY, dX))
	face_aligned_img = rotate_bound(img.copy(), angle)


# brightness and contrast adjystment to frames
def adjustment(frame, alpha=1.0, beta=0):
	new_image = np.zeros(frame.shape, frame.dtype)
	alpha = alpha # Simple contrast control
	beta = beta    # Simple brightness control
	# Do the operation new_image(i,j) = alpha*image(i,j) + beta
	new_image = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
	return new_image


# blurr detection
def blurr_detection(frame, threshold):
	# convert image to gray
	gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
	# compute the Laplacian of the image amf then return th focus
	# measure, which is simply the variance of the Laplacian
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	# print(f'blurr value: {fm}')
	if fm <= threshold:
		return True
	# returns True if the frame is not blurr
	return False


# def yen_histogram_equalization(img):
# 	from skimage.filters import threshold_yen
# 	from skimage.exposure import rescale_intensity
# 	yen_threshold = threshold_yen(img)
# 	bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
# 	return bright


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)