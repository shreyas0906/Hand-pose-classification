from utils import Hand
import numpy as np 

class Gesture(Hand):

    def __init__(self):
        super().__init__()
        # self.type = None # return the type of gesture given the landmarks.

    def is_ok(self):

        distance = np.linalg.norm(self.index_finger - self.thumb) * 100

        if 0.5 < distance < 3.0:
            return True
        return False

    def is_five(self):
        pass

    def is_four(self):
        
        index_finger_above_palm = self.is_point_above_palm_line(self.index_finger)
        middle_finger_above_palm = self.is_point_above_palm_line(self.middle_finger)
        ring_finger_above_palm = self.is_point_above_palm_line(self.ring_finger)
        pinky_finger_above_palm = self.is_point_above_palm_line(self.pinky_finger)
        thumb_finger_below_palm = self.is_point_above_plam_line(self.thumb)

        if index_finger_above_palm and middle_finger_above_palm and ring_finger_above_palm and pinky_finger_above_palm and not thumb_finger_below_palm:
            return True

        return False


    def is_three(self):
        index_finger_above_palm = self.is_point_above_palm_line(self.index_finger)
        middle_finger_above_palm = self.is_point_above_palm_line(self.middle_finger)
        ring_finger_above_palm = self.is_point_above_palm_line(self.ring_finger)
        thumb_ip = self.is_point_above_palm_line(self.thumb_ip)
        pinky_finger = self.is_point_above_palm_line(self.pinky_finger)

        if (index_finger_above_palm and middle_finger_above_palm and ring_finger_above_palm) and not (thumb_ip and pinky_finger):
            return True
        
        return False


    def is_two(self):
        index_finger_above_palm = self.is_point_above_palm_line(self.index_finger)
        middle_finger_above_palm = self.is_point_above_palm_line(self.middle_finger)
        ring_finger_above_palm = self.is_point_above_palm_line(self.ring_finger)
        thumb_ip = self.is_point_above_palm_line(self.thumb_ip)
        pinky_finger = self.is_point_above_palm_line(self.pinky_finger)

        if (index_finger_above_palm and middle_finger_above_palm) and not (ring_finger_above_palm and thumb_ip and pinky_finger):
            return True
        
        return False

    def is_hang(self):
        index_finger_above_palm = self.is_point_above_palm_line(self.index_finger)
        middle_finger_above_palm = self.is_point_above_palm_line(self.middle_finger)
        ring_finger_above_palm = self.is_point_above_palm_line(self.ring_finger)
        thumb_ip = self.is_point_above_palm_line(self.thumb_ip)
        pinky_finger = self.is_point_above_palm_line(self.pinky_finger)
        
        if not (index_finger_above_palm and middle_finger_above_palm and ring_finger_above_palm and thumb_ip) and pinky_finger:
            return True
        
        return False

    def is_heavy(self):
        pass

    def is_L(self):
        pass

    def is_C(self):
        pass

    def is_palm(self):
        pass
