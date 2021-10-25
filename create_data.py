import cv2
import mediapipe as mp
import os
import tqdm
from argparse import ArgumentParser
import pandas as pd
import time


def detect_hands(args):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    landmarks = []

    # For static images:
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:

        gesture_folder = os.listdir(args.gesture_folder)

        if not os.path.exists(os.getcwd() + '/' + args.save_dir):  # if save dir is not created, create a directory to save images.
            os.mkdir(args.save_dir)

        print(f"Gestures: {gesture_folder}\n")

        for dir in gesture_folder:

            image_files = [img for img in os.listdir(os.path.join(args.gesture_folder, dir)) if
                           img.endswith('.jpeg') or img.endswith('.png') or img.endswith('.jpg')]

            for file in tqdm.tqdm(image_files, desc=f'Processing mediapipe on {dir} gesture images'):

                file_path = os.path.join(args.gesture_folder, dir, file)

                image = cv2.flip(cv2.imread(file_path), 1)                        # Read an image, flip it around y-axis for correct handedness output
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))   # Convert the BGR image to RGB before processing.

                try:
                    hand = results.multi_handedness[-1].classification.pop().label  # To get which hand (left, right)
                except TypeError as e:
                    hand = None

                if not results.multi_hand_landmarks:
                    continue

                annotated_image = image.copy()

                for hand_landmarks in results.multi_hand_landmarks:

                    if args.save_images == 'True':
                        cv2.imwrite(args.save_dir + '/' + file, cv2.flip(annotated_image, 1))

                    points = [file, dir, hand]

                    for point in mp_hands.HandLandmark:
                        normalized_landmark = hand_landmarks.landmark[point]
                        points.append(normalized_landmark.x)
                        points.append(normalized_landmark.y)
                        points.append(normalized_landmark.z)

                    landmarks.append(points)

    return landmarks


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--save_dir', type=str, default='annotated_images', help='Directory containing the saved images')
    p.add_argument('--gesture_folder', type=str, required=False, default='All_images', help="Gesture Training images")
    p.add_argument('--save_images', type=str, required=False, default='False', help='Flag to save annotated images')
    p.add_argument('--name_csv', type=str, required=False, default='landmarks',
                   help='Name of the csv to save the landmarks')

    print(p.format_usage())
    start = time.time()
    args = p.parse_args()

    landmarks = detect_hands(args)

    csv_columns = ['file_name', 'label', 'hand',
                   '0-wrist.x', '0-wrist.y', '0-wrist.z',
                   '1-thumb_cmc.x', '1-thumb_cmc.y', '1-thumb_cmc.z',
                   '2-thumb_mcp.x', '2-thumb_mcp.y', '2-thumb_mcp.z',
                   '3-thumb_ip.x', '3-thumb_ip.y', '3-thumb_ip.z',
                   '4-thumb_tip.x', '4-thumb_tip.y', '4-thumb_tip.z',
                   '5-index_finger_mcp.x', '5-index_finger_mcp.y', '5-index_finger_mcp.z',
                   '6-index_finger_pip.x', '6-index_finger_pip.y', '6-index_finger_pip.z',
                   '7-index_finger_dip.x', '7-index_finger_dip.y', '7-index_finger_dip.z',
                   '8-index_finger_tip.x', '8-index_finger_tip.y', '8-index_finger_tip.z',
                   '9-middle_finger_mcp.x', '9-middle_finger_mcp.y', '9-middle_finger_mcp.z',
                   '10-middle_finger_pip.x', '10-middle_finger_pip.y', '10-middle_finger_pip.z',
                   '11-middle_finger_dip.x', '11-middle_finger_dip.y', '11-middle_finger_dip.z',
                   '12-middle_finger_tip.x', '12-middle_finger_tip.y', '12-middle_finger_tip.z',
                   '13-ring_finger_mcp.x', '13-ring_finger_mcp.y', '13-ring_finger_mcp.z',
                   '14-ring_finger_pip.x', '14-ring_finger_pip.y', '14-ring_finger_pip.z',
                   '15-ring_finger_dip.x', '15-ring_finger_dip.y', '15-ring_finger_dip.z',
                   '16-ring_finger_tip.x', '16-ring_finger_tip.y', '16-ring_finger_tip.z',
                   '17-pinky_mcp.x', '17-pinky_mcp.y', '17-pinky_mcp.z',
                   '18-pinky_pip.x', '18-pinky_pip.y', '18-pinky_pip.z',
                   '19-pinky_dip.x', '19-pinky_dip.y', '19-pinky_dip.z',
                   '20-pinky_tip.x', '20-pinky_tip.y', '20-pinky_tip.z',
                   ]

    csv_landmarks = pd.DataFrame(landmarks, columns=csv_columns)
    print("Saving landmarks to landmarks.csv...")
    csv_landmarks.to_csv(f'data/{args.name_csv}.csv', index=False)
    print(f"total time taken: {(time.time() - start) / 60} mins")


