import numpy as np
from numpy.lib.shape_base import expand_dims
import tensorflow as tf
import cv2 
import mediapipe as mp 

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../mediapipe_tflite_models/pose_landmark_full.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\ninput_details: {input_details}")
print(f"\noutput details: {output_details}")
print(f"\n")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, image = cap.read()
    image = image.astype(np.float32)
    image_width, image_height, _ = image.shape
    print(f"image width: {image_width} and image height: {image_height}")

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    
    image = cv2.resize(image, (255,255))
    image = np.expand_dims(image, axis=0)
    print(f"image shape: {image.shape}")
    interpreter.set_tensor(input_details[0]['index'], 1)
    interpreter.invoke()
    results = interpreter.get_tensor(output_details[0]['index'])
    print(results)

cap.release()

