import cv2
import mediapipe as mp
import numpy as np
import time
import json
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras.models import load_model


class WebcamTest:

    def __init__(self, model):
        self.hands = mp.solutions.hands
        self.model = model
        self.internal_camera_index = None
        self.external_camera_index = None
        self.video = None
        self.select_camera_source()

        with open('encoded_labels.json', 'r') as f:
            self.labels = json.load(f, encoding='unicode_escape')
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    def select_camera_source(self):
        """
        Tries to find the camera source.
        Priority is given to external webcams over built-in cameras
        because of better quality input.
        Assuming,
        1. The external camera cam be located with an index < 5.
        2. In-built camera index is
        :return: selected camera source
        """
        for camera_index in range(5, -1, -1):
            camera = cv2.VideoCapture(camera_index)
            print(f"checking for camera input at index: {camera_index}")
            success, frame = camera.read()
            if success and frame is not None:
                if camera_index > 1:
                    print(f"external camera found at index: {camera_index}")
                    self.external_camera_index = camera_index
                else:
                    print(f"built-in camera found at index: {camera_index}")
                    self.internal_camera_index = camera_index
                print(f"camera resolution: {frame.shape[0]} {frame.shape[1]}")
            camera.release()

        if self.external_camera_index:
            self.video = cv2.VideoCapture(self.external_camera_index)
        else:
            self.video = cv2.VideoCapture(self.internal_camera_index)

    def detect_hands(self):

        with self.hands.Hands(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5,
                              max_num_hands=1) as hands:

            while self.video.isOpened():

                success, frames = self.video.read()

                if not success:
                    print("No camera connected.")
                    break

                # hand_frame = frames.copy()
                start_time = time.time()
                frames = cv2.cvtColor(cv2.flip(frames, 1), cv2.COLOR_BGR2RGB)
                frames.flags.writeable = False

                results = hands.process(frames)
                image = cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:

                    tmp1 = str(results.multi_hand_landmarks[0]).split()
                    coordinates_list = [elem for elem in tmp1 if elem != 'landmark' if elem != '{'
                                   if elem != '}' if elem != 'x:' if elem != 'y:' if elem != 'z:']

                    coordinates_array = np.array(coordinates_list, dtype='float32')
                    input_array = np.expand_dims(coordinates_array, axis=0)
                    prediction = self.model.predict(input_array)
                    predicted_label = self.labels[str(np.argmax(prediction))]

                else:
                    predicted_label = "No hand"

                cv2.putText(image, predicted_label, (35, 35), self.font, 1, (0, 255, 255), 1, cv2.LINE_4)
                cv2.putText(image, str("{0:.3f}".format((time.time() - start_time) * 100)) + 'ms',
                            (35, 65), self.font, 1, (0, 0, 255), 2,
                            cv2.LINE_4)
                cv2.imshow('Live Test', image)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    self.on_destroy()

    def on_destroy(self):
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('--model_dir', required=False, type=str,
                   default='models/model_27-6-6_58/trained_model.h5',
                   help='path to trained_model.h5')

    print(p.format_usage())
    args = p.parse_args()
    model = load_model(args.model_dir)
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        print("GPU is available")
    else:
        print("GPU is not available")

    test = WebcamTest(model)
    test.detect_hands()
