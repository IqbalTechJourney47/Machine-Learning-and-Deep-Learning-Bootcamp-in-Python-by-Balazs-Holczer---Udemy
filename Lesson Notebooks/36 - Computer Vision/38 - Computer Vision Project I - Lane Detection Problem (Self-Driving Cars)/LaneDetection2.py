import numpy as np
import cv2

original_image = cv2.imread('black-and-white-bird-C.jpg', cv2.IMREAD_COLOR)

# we have to transform the image into grayscale
# OpenCV handles BGR insteal of RGB
gray_image = cv2.Laplacian(gray_image, -1)

cv2.imshow('Original Image', gray_image)
cv2.imshow('Result', result_image)

cv2.waitKey(0)
