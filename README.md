# Hand pose classification

In this project, we'll be doing hand pose classification based on the landmarks provided by [Mediapipe](https://google.github.io/mediapipe/solutions/hands#python-solution-api)
The idea is to classify the hand poses on the extracted landmarks (3D) rather than on the image itself.

![](hand_landmarks.png)

Video demo can be found on [youtube](https://youtu.be/3V5tQBCl8wQ)
## Dataset creation and input pipeline.

In this section,
- Downloading the data. 
- Creating dataset with hand landmarks.
- Using tf.data.Dataset to create input pipeline to the model.

### Downloading the dataset

1. Download the dataset from [link](https://www.gti.ssr.upm.es/data/MultiModalHandGesture_dataset)
2. The dataset contains 16 different gestures from 25 unique users.
3. Please note that the images are captured from a near-infrared camera.
4. The gesture is a mirror image when viewed from the back of the hand and the nail side.
5. For simplicity sake, I have selected gestures for C, five, four, hang, heavy, L, ok, palm, three, two 
6. Before running the script to extract the landmarks to a csv file, copy all images from different users.
- [ ] write a script gather train images into a single folder.

### Creating dataset with hand landmarks.

loop:
   - for image in the gesture_dir:
      - run mediapipe on the image.
      - get the landmarks.
      - save the landmarks, gesture label, file_name to a csv file.
     
usage:
   
`python3 create_data.py --gesture_folder --save_dir --save_images --name_csv`

- gesture_folder --> name of directory containing folders of gestures.
- save_dir --> name of directory to save annotated images if --save_images is True
- save_images --> flag to save annotated images or not.
- name_csv --> name of the csv file containing the hand landmarks.


![](examples/frame_17653_l.png) ![](examples/frame_17653_l_annotated.png)

### creating input pipeline

1. The file at src/data.py is pretty self-explanatory for this step. 

## Training structure

1. src/train.py contains all the configuration requirements to train the model.
2. src/models.py contains the model arcitectures. Add your models here.
3. src/train.py can test the model on live webcam feed or the test set if the flags are set appropriately.

The code is pretty self-explanatory. If you need any explanations, please feel free to contact me at shreyas0906@gmail.com

