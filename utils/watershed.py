'''
Author: Manohar Mukku
Date: 06.12.2018
Desc: Watershed Segmentation algorithm
GitHub: https://github.com/manoharmukku/watershed-segmentation
'''

import sys
import imutils
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

def movingThreshold(img_, n, b):
    img = img_.copy()
    img[1:-1:2, :] = np.fliplr(img[1:-1:2, :])  #  Vector flip 
    f = img.flatten()  #  Flatten to one dimension 
    ret = np.cumsum(f)
    ret[n:] = ret[n:] - ret[:-n]
    m = ret / n  #  Moving average 
    g = np.array(f>=b*m).astype(int)  #  Threshold judgment ,g=1 if f>=b*m
    g = g.reshape(img.shape)  #  Restore to 2D 
    g[1:-1:2, :] = np.fliplr(g[1:-1:2, :])  #  Flip alternately 
    return g*255

def distance_transform_info(img: np.ndarray):
    # _, bin = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    _, bin = cv.threshold(img, 160, 255, cv.THRESH_BINARY)
    # img_ = clahe.apply(img.copy())
    # img_ = cv.GaussianBlur(img_, (13, 13), 0)
    # bin = movingThreshold(img_, 90, 0.95).astype(np.uint8)
    bin = cv.GaussianBlur(bin, (3, 3), 0)
    plt.imshow(bin)
    plt.show()
    bin = 255 - bin
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(bin, cv.MORPH_OPEN, kernel, iterations = 2)
    dt = cv.distanceTransform(opening, cv.DIST_L2, 3)
    plt.imshow(dt)
    plt.show()
    dt = cv.normalize(dt, None, 0, 255, cv.NORM_MINMAX)
    dt = 255 - dt
    dt[dt==255] = 0
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

def neighbourhood(image, x, y):
    # Save the neighbourhood pixel's values in a dictionary
    neighbour_region_numbers = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0):
                continue
            if (x+i < 0 or y+j < 0): # If coordinates out of image range, skip
                continue
            if (x+i >= image.shape[0] or y+j >= image.shape[1]): # If coordinates out of image range, skip
                continue
            if (neighbour_region_numbers.get(image[x+i][y+j]) == None):
                neighbour_region_numbers[image[x+i][y+j]] = 1 # Create entry in dictionary if not already present
            else:
                neighbour_region_numbers[image[x+i][y+j]] += 1 # Increase count in dictionary if already present

    # Remove the key - 0 if exists
    if (neighbour_region_numbers.get(0) != None):
        del neighbour_region_numbers[0]

    # Get the keys of the dictionary
    keys = list(neighbour_region_numbers)

    # Sort the keys for ease of checking
    keys.sort()

    if (keys[0] == -1):
        if (len(keys) == 1): # Separate region
            return -1
        elif (len(keys) == 2): # Part of another region
            return keys[1]
        else: # Watershed
            return 0
    else:
        if (len(keys) == 1): # Part of another region
            return keys[0]
        else: # Watershed
            return 0

def watershed_segmentation(image):
    # Create a list of pixel intensities along with their coordinates
    intensity_list = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Append the tuple (pixel_intensity, xy-coord) to the end of the list
            intensity_list.append((image[x][y], (x, y)))

    # Sort the list with respect to their pixel intensities, in ascending order
    intensity_list.sort()

    # Create an empty segmented_image numpy ndarray initialized to -1's
    segmented_image = np.full(image.shape, -1, dtype=int)

    # Iterate the intensity_list in ascending order and update the segmented image
    region_number = 0
    for i in range(len(intensity_list)):
        # Print iteration number in terminal for clarity
        sys.stdout.write("\rPixel {} of {}...".format(i, len(intensity_list)))
        sys.stdout.flush()

        # Get the pixel intensity and the x,y coordinates
        intensity = intensity_list[i][0]
        x = intensity_list[i][1][0]
        y = intensity_list[i][1][1]

        # Get the region number of the current pixel's region by checking its neighbouring pixels
        region_status = neighbourhood(segmented_image, x, y)

        # Assign region number (or) watershed accordingly, at pixel (x, y) of the segmented image
        if (region_status == -1): # Separate region
            region_number += 1
            segmented_image[x][y] = region_number
        elif (region_status == 0): # Watershed
            segmented_image[x][y] = 0
        else: # Part of another region
            segmented_image[x][y] = region_status

    # Return the segmented image
    return segmented_image

def set_of_pixel_intensities(image):
    intensities = cv.calcHist([image], [0], None, [256], [0, 256])
    intensities = [arg for arg in range(0, 256) if intensities[arg] > 0]
    del intensities[intensities == 0]
    return intensities

def random_colorize(img, intensities):
    image = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
    for i in intensities:
        random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        image[img == i] = random_color
    return image

def main(argv):
    # Read the input image
    img = cv.imread(argv[0], 0)
    img = imutils.resize(img, width = 200)
    dt = distance_transform_info(img)
    # img, local_minima = local_minima_points(dt)

    # Check if image exists or not
    if (img is None):
        print ("Cannot open {} image".format(argv[0]))
        print ("Make sure you provide the correct image path")
        sys.exit(2)

    # Perform segmentation using watershed_segmentation on the input image
    segmented_image = watershed_segmentation(dt)
    segmented_image[dt == 0] = 0
    segmented_image = cv.normalize(segmented_image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    segmented_image = cv.equalizeHist(segmented_image)
    ints_set = set_of_pixel_intensities(segmented_image)
    segmented_image_ = random_colorize(segmented_image, ints_set)
    # Save the segmented image as png to disk
    # cv2.imwrite("images/target.png", segmented_image)

    # Show the segmented image and original image side by side
    # input_image = cv.resize(img, (0,0), None, 0.5, 0.5)
    # seg_image = cv.resize(cv.imread("images/target.png", 0), (0,0), None, 0.5, 0.5)
    # numpy_horiz = np.hstack((input_image, seg_image))
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(dt)
    ax[2].imshow(segmented_image_)
    plt.show()

    # cv2.imshow('Input image ------------------------ Segmented image', numpy_horiz)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv[1:])