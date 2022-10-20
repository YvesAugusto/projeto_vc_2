import numpy as np
import matplotlib.pyplot as plt
import cv2

def movingThreshold(img, n, b):
    # img = img_.copy()
    img[1:-1:2, :] = np.fliplr(img[1:-1:2, :])  #  Vector flip 
    f = img.flatten()  #  Flatten to one dimension 
    ret = np.cumsum(f)
    ret[n:] = ret[n:] - ret[:-n]
    m = ret / n  #  Moving average 
    g = np.array(f>=b*m).astype(int)  #  Threshold judgment ,g=1 if f>=b*m
    g = g.reshape(img.shape)  #  Restore to 2D 
    g[1:-1:2, :] = np.fliplr(g[1:-1:2, :])  #  Flip alternately 
    return g*255

img1 = cv2.imread("images/nome_3.jpeg", 0)
img2 = cv2.imread("images/nome_4.jpeg", 0)

ret1, imgOtsu1 = cv2.threshold(img1, 127, 255, cv2.THRESH_OTSU)
ret2, imgOtsu2 = cv2.threshold(img2, 127, 255, cv2.THRESH_OTSU)
# cv2.imwrite("images/otsu_1.png", imgOtsu1)
# cv2.imwrite("images/otsu_2.png", imgOtsu1)
imgMoveThres1 = movingThreshold(img1, 30, 0.8)
imgMoveThres2 = movingThreshold(img2, 30, 0.8)
# cv2.imwrite("images/moving_mean_1.png", imgMoveThres1)
# cv2.imwrite("images/moving_mean_2.png", imgMoveThres2)
plt.figure(figsize=(9, 6))
plt.subplot(231), plt.axis('off'), plt.title("Origin")
plt.imshow(img1, 'gray')
plt.subplot(232), plt.axis('off'), plt.title("OTSU(T={})".format(ret1))
plt.imshow(imgOtsu1, 'gray')
plt.subplot(233), plt.axis('off'), plt.title("Moving threshold")
plt.imshow(imgMoveThres1, 'gray')
plt.subplot(234), plt.axis('off'), plt.title("Origin")
plt.imshow(img2, 'gray')
plt.subplot(235), plt.axis('off'), plt.title("OTSU(T={})".format(ret2))
plt.imshow(imgOtsu2, 'gray')
plt.subplot(236), plt.axis('off'), plt.title("Moving threshold")
plt.imshow(imgMoveThres2, 'gray')
plt.tight_layout()
plt.show()