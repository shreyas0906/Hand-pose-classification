import numpy as np
import cv2
import os
import glob
import tqdm 
from argparse import ArgumentParser

def img_aug(args):
    
    src_path = args.image_dir
    folder = os.listdir(src_path)
    
    for dir in tqdm.tqdm(folder):
        print(f"Processing: {dir}")
        imgs = glob.glob(src_path + dir + "/*.png") # + usr + "/train_pose/"
        dest = "/resize-" + dir + "/"  
       
        if not os.path.exists(src_path + dir + dest):
            os.mkdir(src_path + dir + dest) # usr + "/train_pose/" 
        
        for img in imgs:
            print(f"Processing image: {img}")

            image = cv2.imread(img)
            image = cv2.resize(image, (640,480))
            cv2.imwrite(src_path + dir + dest + f"{dir}_%s"%str(img).split("/")[-1], image)
            

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--image_dir', type=str, default='./', help='Directory containing the images')
    print(p.format_usage())
    args = p.parse_args()
    img_aug(args)