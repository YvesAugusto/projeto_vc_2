import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imutils
from growing_regions import bfs_factory
from common import WatershedLimiar

def distance_transform_info(img: np.ndarray):
    _, bin = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    bin = 255 - bin
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(bin, cv.MORPH_OPEN, kernel, iterations = 2)
    dt = cv.distanceTransform(opening, cv.DIST_L2, 3)
    dt = cv.normalize(dt, None, 0, 255, cv.NORM_MINMAX)
    dt = 255 - dt
    return dt

def local_minima_points(dt):
    _, sure_fg = cv.threshold(dt, int(0.7*dt.max()), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    sure_fg = cv.dilate(sure_fg, kernel, iterations=2)
    cnts, _ = cv.findContours(sure_fg, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    local_minima = []
    for c in cnts:
        if cv.contourArea(c) > 3000:
            continue
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        local_minima.append([cX, cY])
        # cv.circle(dt, (cX, cY), 5, 255)
    return sure_fg, local_minima

if __name__ == '__main__':
    img = cv.imread("images/coins.jpg", 0)
    img = imutils.resize(img, width=400)
    dt = distance_transform_info(img)
    foreground, local_minima = local_minima_points(dt)

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(dt, cmap='gray')
    ax[2].imshow(foreground, cmap='gray')
    plt.show()