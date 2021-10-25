import cv2
import os, glob
import mediapipe as mp
import pandas as pd
from argparse import ArgumentParser

def b_box(args):
    src_path = args.image_dir
    user = os.listdir(src_path)
    mphands = mp.solutions.hands
    # hands = mphands.Hands()

    src_path = args.image_dir
    csv_columns = ['file_name', 'x_min', 'x_max', 'y_min','y_max']

    user = os.listdir(src_path)

    with mphands.Hands(
      static_image_mode=True,
      max_num_hands=1,
      min_detection_confidence=0.5) as hands:
        for usr in user:
            files = ['five','four','three','two','C','heavy','hang','L','ok','palm','palm_u']
            for fol in files:
                imgs = glob.glob(src_path+usr+"/train_pose/"+fol+"/aug/*.png")
                # print(imgs)
                ls = []
                for img in imgs:
                    print(img)
                    image = cv2.imread(img)
                    h, w, _ = image.shape
                    framergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = hands.process(framergb)
                    hand_landmarks = result.multi_hand_landmarks
                    if hand_landmarks:
                        list = []
                        for handLMs in hand_landmarks:
                            x_max = 0
                            y_max = 0
                            x_min = w 
                            y_min = h 
                            for lm in handLMs.landmark:
                                x, y = int(lm.x * w), int(lm.y * h)
                                if x > x_max:
                                    x_max = x
                                if x < x_min:
                                    x_min = x
                                if y > y_max:
                                    y_max = y
                                if y < y_min:
                                    y_min = y
                            # cv2.rectangle(frame, (x_min - 30, y_min - 30), (x_max + 30 , y_max + 30), (0, 255, 0), 2)
                            
                            list = [str(img).split("/")[-1],x_min-30,x_max+30,y_min-30,y_max+30]
                        ls.append(list)
                        # break
                data = ls.copy()
                data = pd.DataFrame(data,columns=csv_columns)
                print(data.head())
                data.to_csv(src_path+usr+"/train_pose/"+fol+".csv")
                print(src_path+usr+"/train_pose/"+fol+"/aug/"+fol+".csv")
                                

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--image_dir', type=str, default='./', help='Directory containing the images')
    args = p.parse_args()
    b_box(args)
