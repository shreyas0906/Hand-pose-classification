import cv2
import mediapipe as mp
import numpy as np
import time
import collections
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from tensorflow.keras.models import load_model

ls_fps = collections.deque(maxlen=10)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model = load_model("src/models/model_27-6-6:58/trained_model.h5")
print(model.summary())

font = cv2.FONT_HERSHEY_SIMPLEX
t = 0.90

a = [[0, 1, 0]]
i = 1
y_pred = []
y_true = []
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
    while cap.isOpened():

        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        start = time.time()
        hand_image = image.copy()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        score = 0
        if results.multi_hand_landmarks:

            S = str(results.multi_hand_landmarks[0]).split()
            ls = [elem for elem in S if elem != 'landmark' if elem != '{'
                  if elem != '}' if elem != 'x:' if elem != 'y:' if elem != 'z:']
            ls = np.float32(ls)
            B = np.array(ls)
            B = np.expand_dims(B, axis=0)
            a = model.predict(B)
            i = np.argmax(a)

            if i == 0:
                gt = "C"
            elif i == 1:
                gt = "L"
            elif i == 2:
                gt = "five"
            elif i == 3:
                gt = "four"
            elif i == 4:
                gt = "hang"
            elif i == 5:
                gt = "heavy"
            elif i == 6:
                gt = "ok"
            elif i == 7:
                gt = "palm"
            elif i == 8:
                gt = "three"
            elif i == 9:
                gt = "two"

        else:
            gt = "no hand"
        cv2.putText(image,
                    gt,
                    (35, 35),
                    font, 1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)
        cv2.putText(image,
                    str(start - time.time() * 100) + 'ms',
                    (35, 65),
                    font, 1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)

        cv2.imshow('Annotated hand', image)
        cv2.imshow("just hand", hand_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
