"""
This code is for MultiMoodHandGestRecog dataset
link: https://www.gti.ssr.upm.es/images/Data/Downloads/MultiModalHandGestureDataset/MultiModHandGestRecog.rar

"""

import numpy as np
import cv2
import os
import glob
import tqdm 
from argparse import ArgumentParser
from rotate import rotate_image, random_rotation

# "/media/zeki/Data/Dataset/MultiModHandGestRecog/near-infrared/"


def img_aug(args):
    src_path = args.image_dir
    ang_1 = -25
    ang_2 = 25
    # user = os.listdir(src_path)
    # for usr in user:
    folder = ['five','four','three','two','C','heavy','hang','L','ok','palm','palm_u']
    for dir in tqdm.tqdm(folder):
        print(f"Processing: {dir}")
        imgs = glob.glob(src_path + dir + "/*.png") # + usr + "/train_pose/"
        dest = "/aug-" + dir + "/"  
        if not os.path.exists(src_path + dir + dest):
            os.mkdir(src_path + dir + dest) # usr + "/train_pose/" 
        for img in imgs:
            print(f"Processing image: {img}")

            image = cv2.imread(img)
            image = cv2.resize(image, (640,480))
            
            #flipping
            """
            save image in same folder as F_image_name
            """
            
            f_img = image.copy()
            f_img = cv2.flip(f_img, 1)
            f_img = cv2.resize(f_img, (640, 480))
            cv2.imwrite(src_path + dir + dest + f"{dir}_F_%s"%str(img).split("/")[-1], f_img) #usr + "/train_pose/"
            # cv2.imwrite(src_path + dir + img, image) # usr + "/train_pose/" +
            #random_rotation
            """
            save image in same folder as R_image_name
            """
            r_img = image.copy()
            r_img = random_rotation(r_img, ang_1, ang_2)
            r_img = cv2.resize(r_img, (640, 480))
            cv2.imwrite(src_path + dir + dest + f"{dir}_R_%s"%str(img).split("/")[-1], r_img) # usr + "/train_pose/" +
            
            #ramdom_rotation_and_flipping
            """
            save image in same folder as R_F_image_name
            """
            r_f_img = image.copy()
            r_f_img = cv2.flip(r_f_img, 1)
            r_f_img = cv2.resize(r_f_img, (640, 480))
            cv2.imwrite(src_path + dir + dest + f"{dir}_R_F_%s"%str(img).split("/")[-1], r_f_img) # usr + "/train_pose/" +

        

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--image_dir', type=str, default='./', help='Directory containing the images')
    print(p.format_usage())
    args = p.parse_args()
    img_aug(args)
