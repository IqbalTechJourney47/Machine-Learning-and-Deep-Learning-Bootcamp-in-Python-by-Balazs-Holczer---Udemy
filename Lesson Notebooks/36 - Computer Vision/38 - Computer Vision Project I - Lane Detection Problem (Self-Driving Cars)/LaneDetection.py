import cv2
import numpy as np

# video = several frames (images shown after each other)
video = cv.VideoCapture('lane_detection_video.mp4')

while video.isOpened():
    is_grabbed, frame = video.read()

    # because the end of the video
    if not is_grabbed:
        break

    cv2.imshow('Lane Detection Video', frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()
