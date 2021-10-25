import numpy as np 

def is_valid_ok(mp_hands, hand_landmarks):

      index_finger = np.array((float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x ),
            float(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y )))

      thumb_finger = np.array((float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x ),
            float(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y)))

      #calculating the distance between index finger tip and thumb finger tip
      distance = np.linalg.norm(index_finger - thumb_finger) * 100

      if 0.5 < distance < 3.0:
            return True
      
      return False

def L_gesture(mp_hands, hand_landmarks):
      pass
