import cv2
import numpy as np

def get_detected_lanes(image):

    (height, width) = (image.shape[0], image.shape[1])

    # we have to turn the image intograyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection kernel (Canny's algorithm)
    canny_image = cv2.Canny(gray_image, 100, 120)

    return image

# video = several frames (images shown after each other)
video = cv.VideoCapture('lane_detection_video.mp4')

while video.isOpened():
    is_grabbed, frame = video.read()

    # because the end of the video
    if not is_grabbed:
        break

    frame = get_detected_lanes(frames)

    cv2.imshow('Lane Detection Video', frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()
