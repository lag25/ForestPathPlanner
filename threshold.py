import numpy as np
import cv2

def RGNull(img):
    arr = img.copy()

    # Zero out red and blue channels (keep green)
    arr[:, :, 0] = 0  # Red
    arr[:, :, 2] = 0  # Blue

    # Efficient mean of green channel
    green_channel = arr[:, :, 1]
    green_mean = np.mean(green_channel)

    # Threshold
    threshold = green_mean / 1.5
    return arr, threshold

def IsoGray(img):
    RGNull_img, thresh = RGNull(img)

    # Convert to grayscale (only green channel matters now)
    gray_img = cv2.cvtColor(RGNull_img, cv2.COLOR_RGB2GRAY)
    return gray_img, thresh

def IsoGrayThresh(img):
    gray_img, threshold = IsoGray(img)

    # Apply threshold using vectorized operation
    binary_img = np.where(gray_img > threshold, 255, 0).astype(np.uint8)

    return binary_img
