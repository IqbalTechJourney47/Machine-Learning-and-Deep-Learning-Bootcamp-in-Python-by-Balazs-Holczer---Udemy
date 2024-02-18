#################################################################################################

# 002 Handling pixel intensities I

import numpy as np
import cv2

# print(cv2.__version__)

image = cv2.imread('700x365-C.jpg', cv2.IMREAD_COLOR)

print(image.shape)   # shape of 2D array - 545 rows and 800 columns
print(np.amax(image))
print(np.amin(image))

# values close to 0: darker pixels
# values closer to 255: brighter pixels
print(image)   # 2D array representation

cv2.imshow('Computer Vision', image)

# Add x and y axes
cv2.line(image, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2), (0, 0, 255), 2)
cv2.line(image, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]), (0, 0, 255), 2)

# Display RGB values at a specific pixel
pixel_value = image[100, 100]
print("RGB Values at pixel (100, 100):", pixel_value)

# Add text to the image (showing RGB values at a specific pixel)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, f'RGB: {pixel_value}', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Display the modified image with text
cv2.imshow('Computer Vision with Axes and Text', image)

# Wait for a key event and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# when the image is a grayscale, the pixel intensity values are within the range [0,255]

#################################################################################################
