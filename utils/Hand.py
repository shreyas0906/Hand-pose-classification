import numpy as np
from numpy.core.numeric import cross 

class Hand():

    def __init__(self, mp_hands, hand_landmarks, hand):
        self.IMAGE_WIDTH = 480
        self.IMAGE_HEIGHT = 640
        self.orientation = hand.multi_handedness[-1].classification.pop().label
        self.bbox = None
        self.landmarks = None

        self.wrist = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * self.IMAGE_HEIGHT ),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * self.IMAGE_WIDTH)))

        self.thumb = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * self.IMAGE_WIDTH)))

        self.index_finger = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.IMAGE_WIDTH)))

        self.middle_finger = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * self.IMAGE_HEIGHT ),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * self.IMAGE_WIDTH)))

        self.ring_finger = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * self.IMAGE_WIDTH)))

        self.pinky_finger = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * self.IMAGE_WIDTH)))

        self.thumb_cmc = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * self.IMAGE_WIDTH)))

        self.thumb_ip = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * self.IMAGE_WIDTH)))

        self.index_finger_mcp = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * self.IMAGE_WIDTH)))

        self.middle_finger_mcp = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * self.IMAGE_WIDTH)))

        self.ring_finger_mcp = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * self.IMAGE_WIDTH)))

        self.pinky_mcp = np.array((int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * self.IMAGE_HEIGHT),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * self.IMAGE_WIDTH)))

        self.palm_line = [self.index_finger_mcp, self.pinky_mcp]


    def is_point_above_palm_line(self, point):
        
        if self.cross_product(point) > 0:
            return True

        return False

    def cross_product(self, point):

        vector_1 = [self.palm_line[1][0] - self.palm_line[0][0], self.palm_line[1][1] - self.palm_line[0][1]]

        vector_2 = [self.palm_line[1][0] - point[0], self.palm_line[1][1] - point[1]]

        result = (vector_1[0] * vector_2[1]) - (vector_1[1] * vector_2[0])

        return result

    def print_all_finger_tips(self):
        print(f"wrist point: {self.wrist}")
        print(f"thumb: {self.thumb}")
        print(f"index finger: {self.index_finger}")
        print(f"middle finger: {self.middle_finger}")
        print(f"ring finger: {self.ring_finger}")
        print(f"pinky finger: {self.pinky_finger}")

