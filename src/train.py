import cv2
import datetime
import glob
import os
import tqdm
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model, save_model

import models
from live_test import WebcamTest
from checkpoints import model_checkpoint, tensorboard, reduce_lr_on_plateau
from data import Dataloader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_pose_labels():

    with open('encoded_labels.json', 'r') as f:
        labels = json.load(f, encoding='unicode_escape')

    return labels


def genModelPath():
    now = datetime.datetime.now()

    if not os.path.exists(os.getcwd() + '/models'):
        os.makedirs(os.getcwd() + '/models')
    else:
        name = 'model_{}-{}-{}:{}'.format(now.day, now.month, now.hour, now.minute)
        if not os.path.exists('models/' + name):
            os.makedirs('models/' + name)

        return 'models/' + name + '/trained_model.h5'


def get_latest_model_dir():
    return max(glob.glob(os.path.join('models/', '*/')), key=os.path.getmtime)


def check_gpu_status():
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        for device in physical_devices:
            print(f" GPU is available: {device}")
    else:
        print("GPU not available")


def train(args):
    check_gpu_status()
    save_path = genModelPath()

    data = Dataloader(args)
    train_ds = data.get_train_data()
    test_ds = data.get_test_data()
    number_of_gestures = 10

    model = models.conv_model((63, 1), number_of_gestures)

    print(model.summary())

    history = model.fit(train_ds,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        callbacks=[tensorboard(), reduce_lr_on_plateau(), model_checkpoint(save_path)],
                        validation_data=test_ds,
                        verbose=1
                        )

    model_dir = 'models/'
    save_model(model, model_dir, overwrite=True, include_optimizer=False)
    os.environ['MODEL_DIR'] = model_dir
    convert_to_tflite(save_path)
    print('-' * 50 + '\n')
    print("\nModel training has ended\n")
    print('-' * 50 + '\n')
    plot_loss(history)


def plot_loss(history):
    print("Plotting graphs")
    plt.plot(history.history['loss'], color='red')
    plt.plot(history.history['accuracy'])
    plt.xlabel('epoch')
    plt.ylabel('loss and accuracy')
    plt.title('Accuracy vs loss')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig(f'model_diagnostics/model_losses_performance.jpg')
    print("done plotting losses at model_diagnostics/model_losses_performance.jpg")


def get_recent_model():
    print('-' * 50 + '\n')
    print(f'\n Fetching the recent model from: {get_latest_model_dir()}')
    model = load_model(os.path.join(get_latest_model_dir(), 'trained_model.h5'))
    return model


def convert_to_tflite(model_dir):
    print('-' * 50 + '\n')
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir) 
    tflite_model = converter.convert()

    with open(model_dir + '/gesture_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"converted to tflite version")


def test(args):

    model = get_recent_model()

    if args.test_live == 'True':
        test_on_camera = WebcamTest(model)
        test_on_camera.detectHands()
    elif args.test_csv:
        test_folder(model, args.test_csv)


def test_folder(model, csv):
    
    test_csv = pd.read_csv(csv)
    test_images = test_csv['file_name']
    test_label = list(test_csv['label']) 
    predicted_label = []
    labels = get_pose_labels()

    for img_file_name in tqdm.tqdm(test_images, desc='Testing on images.'):     
        file_path = img_file_name.split("../../")[1]
        img = cv2.imread(file_path)

        img = np.expand_dims(img, axis=0)
        prediction = np.argmax(model.predict(img))
        predicted_label.append(labels[prediction])

    confusion_mat = confusion_matrix(test_label, predicted_label)
    accuracy = accuracy_score(test_label, predicted_label)
    print(f"confusion matrix: {confusion_mat} accuracy: {accuracy}")
    

if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('--csv', type=str, required=False, default='../data/landmarks.csv', help='Location of the csv file '
                                                                                            'containing landmarks '
                                                                                            'data')
    p.add_argument('--split_size', type=float, required=False, default=0.05, help='Splitting the data to train and '
                                                                                  'test size')
    p.add_argument('--test_csv', required=False, type=str, default='../Test_images/test_data.csv', help='Folder '
                                                                                                        'containing '
                                                                                                        'test images')
    p.add_argument('--test_live', required=False, type=str, default='False', help='Testing on live video')
    p.add_argument('--batch_size', required=False, default=64, type=int, help='Batch size for training')
    p.add_argument('--epochs', required=False, type=int, default=10, help='Number of training epochs')
    p.add_argument('--train', required=True, type=str, default='True', help='Train mode')
    p.add_argument('--test', required=True, type=str, default='False', help='Test mode')

    print(p.format_usage())
    args = p.parse_args()

    if args.train == 'True':
        train(args)
    elif args.test == 'True':
        test(args)

