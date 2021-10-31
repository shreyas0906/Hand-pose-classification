import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video_folder = "Video/"
# cap = cv2.VideoCapture(video_folder + 'video_3_30fps.mp4')
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

fig = plt.figure()
ax = plt.axes(projection='3d')

with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        start = time.time()
        success, image = cap.read()

        image_width, image_height, _ = image.shape

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filler_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(filler_image)  # filler_image

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for point in mp_hands.HandLandmark:
                    index_finger_mcp = np.array((int(hand_landmarks.landmark[point.INDEX_FINGER_MCP].x * image_height),
                                                 int(hand_landmarks.landmark[point.INDEX_FINGER_MCP].y * image_width)))

                    pinky_finger_mcp = np.array((int(hand_landmarks.landmark[point.PINKY_MCP].x * image_height),
                                                 int(hand_landmarks.landmark[point.PINKY_MCP].y * image_width)))

                    middle_finger_mcp = np.array(
                        (int(hand_landmarks.landmark[point.MIDDLE_FINGER_MCP].x * image_height),
                         int(hand_landmarks.landmark[point.MIDDLE_FINGER_MCP].y * image_width)))

                    ring_finger_mcp = np.array((int(hand_landmarks.landmark[point.RING_FINGER_MCP].x * image_height),
                                                int(hand_landmarks.landmark[point.RING_FINGER_MCP].y * image_width)))

                    thumb_cmc = np.array((int(hand_landmarks.landmark[point.THUMB_CMC].x * image_height),
                                          int(hand_landmarks.landmark[point.THUMB_CMC].y * image_width)))

                    wrist = np.array((int(hand_landmarks.landmark[point.WRIST].x * image_height),
                                      int(hand_landmarks.landmark[point.WRIST].y * image_width)))

                    normalizedLandmark = hand_landmarks.landmark[point]
                    pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,
                                                                                           normalizedLandmark.y,
                                                                                           image_width, image_height)

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    vertices = np.array([[middle_finger_mcp[0], middle_finger_mcp[1]],
                                         [index_finger_mcp[0], index_finger_mcp[1]],
                                         [ring_finger_mcp[0], ring_finger_mcp[1]],
                                         [pinky_finger_mcp[0], pinky_finger_mcp[1]],
                                         [wrist[0], wrist[1]],
                                         [thumb_cmc[0], thumb_cmc[1]]], np.int32)
                    pts = vertices.reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=1)

                    # fill it
                    cv2.fillPoly(image, [pts], color=(0, 255, 255))

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()


cap.release()
