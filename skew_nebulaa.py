# import the necessary packages
import numpy as np
import argparse
import cv2
import math
import time
import imutils
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
from skimage.filters import threshold_local

from page_dewarp_nebulaa import *


def correct_skew(clean_thresh_orig, image):
    RESCALED_HEIGHT = 600
    clean_thresh = imutils.resize(clean_thresh_orig, height = int(RESCALED_HEIGHT))

    img = im.fromarray(clean_thresh)

    wd, ht = img.size

    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)


    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score


    delta = 0.5
    limit = 15
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # print('Best angle: {}'.format(best_angle))

    # correct skew
    (h, w) = clean_thresh_orig.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center,best_angle, 1.0)

    unskew_image = cv2.warpAffine(image, M, (w, h),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

   
    return unskew_image

def find_text(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    dilate1Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)

    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    thresh = cv2.threshold(gradX, 100, 255, cv2.THRESH_BINARY,8)[1] # | cv2.THRESH_OTSU,20)[1]

    thresh = cv2.medianBlur(thresh,13)  #remove noise


    thresh = cv2.dilate(thresh, dilate1Kernel,iterations = 1)
    
    return thresh


if __name__ == '__main__':

    imagepath = '17.jpg'
    image = cv2.imread(imagepath)

    image = imutils.resize(image, width = 1200)

    # text_mask threshold image that highlights only the text
    text_mask = find_text(image)
    cv2.imwrite('text.jpg',text_mask)

    # skew correction
    unskew_image = correct_skew(text_mask, image)
    cv2.imwrite('unskew_image.jpg',unskew_image)
 
    # page dewarp
    # HAS TO BE TURNED OFF FOR MRCs and VIDs
    dewarped = page_dewarp(unskew_image, text_mask)

    cv2.imwrite('output.jpg', dewarped)
