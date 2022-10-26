import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("images/triangle.jpg", 0)
image = 255 - image
kernel = np.ones((5,5), np.uint8)
# kernel = np.array([
#     [0,0,1,0,0],
#     [0,1,1,1,0],
#     [0,1,1,1,0],
#     [1,1,1,1,1],
#     [1,1,1,1,1]
# ], np.uint8)

eroded = cv.erode(image, kernel, iterations=1)
dilated = cv.dilate(image, kernel, iterations=1)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(image, cmap='gray')
ax[1].imshow(eroded, cmap='gray')
ax[2].imshow(dilated, cmap='gray')
plt.show()