import time
import cv2

for camera_index in range(5, -1, -1):
    camera = cv2.VideoCapture(camera_index)
    print(f"checking for camera input at index: {camera_index}")
    test, frame = camera.read()
    if test and frame is not None:
        if camera_index > 1:
            print(f"external camera found at index: {camera_index}")
        else:
            print(f"built-in camera found at index: {camera_index}")
        print(f"camera resolution: {frame.shape[0]} {frame.shape[1]}")
    camera.release()
