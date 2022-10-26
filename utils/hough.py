import cv2 as cv
import numpy as np
import imutils
img = cv.imread(cv.samples.findFile('images/xadrez.jpg', 0))
img = imutils.resize(img, 400)
cv.imshow('Original', img)
edges = cv.Canny(img, 50, 150,apertureSize = 3)
lines = cv.HoughLines(edges, 1, np.pi/180, 200)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv.imshow('Canny', edges)
cv.imshow('Hough', img)
cv.waitKey(0)