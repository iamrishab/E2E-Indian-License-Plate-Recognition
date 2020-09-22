#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'


import os
import cv2
import base64
import numpy as np


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
def automatic_brightness_and_contrast(image, clip_hist_percent=5):
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

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def blurr_detection(img):
    # convert image to gray
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # compute the Laplacian of the image amf then return th focus
    # measure, which is simply the variance of the Laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm
    

def sharpen_image(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    # Creating our sharpening filter
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # Applying cv2.filter2D function on our Cybertruck image
    sharpen_img = cv2.filter2D(img, -1, kernel)
    return img


def im2str(im):
    _, imdata = cv2.imencode('.JPG', im)
    return base64.b64encode(imdata).decode('ascii')
