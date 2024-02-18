#################################################################################################

# 002 Handling pixel intensities I

import cv2

# print(cv2.__version__)

image = cv2.imread('800x545-B&W(1).jpg', cv2.IMREAD_GRAYSCALE)

print(image.shape)   # shape of 2D array - 545 rows and 800 columns

# values close to 0: darker pixels
# values closer to 255: brighter pixels
print(image)   # 2D array representation

cv2.imshow('Computer Vision', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# when the image is a grayscale, the pixel intensity values are within the range [0,255]

#################################################################################################

image2 = cv2.imread('700x365-C.jpg', cv2.IMREAD_COLOR)

cv2.imshow('Computer Vision', image2)
