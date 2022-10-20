import numpy as np
import matplotlib.pyplot as plt
import imutils
import pywt
import pywt.data

import cv2 as cv

def apply_fourier(orig):
    orig = imutils.resize(orig, width=200)
    orig_spectrum = np.fft.fftshift(np.fft.fft2(orig))
    print(orig.shape)
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(orig, cmap='gray')
    ax[1].imshow(np.log(abs(orig_spectrum)))
    plt.show()

# Load image
original = cv.imread('images/brad.jpg', 0)

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
_, (_, _, HH2) = pywt.dwt2(HH, 'bior1.3')
_, (_, _, HH3) = pywt.dwt2(HH2, 'bior1.3')
_, (_, _, HH4) = pywt.dwt2(HH3, 'bior1.3')
_, (_, _, HH5) = pywt.dwt2(HH4, 'bior1.3')

for i, a in enumerate([LL, abs(HH), abs(HH2), abs(HH3), abs(HH4), abs(HH5)]):
    plt.imshow(a, cmap="gray")
    plt.show()

for H in [LL, abs(HH), abs(HH2), abs(HH3), abs(HH4), abs(HH5)]:
    apply_fourier(H.astype(np.uint8))