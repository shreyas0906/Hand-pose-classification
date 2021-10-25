# from re import S
import numpy as np
from numpy import array
import pandas as pd
import os, pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Dataloader:

    def __init__(self, args):
        self.csv_data = pd.read_csv(args.csv)
        self.csv_label = self.csv_data.pop('label')
        self.clean_csv()

        self.split_size = args.split_size
        self.batch_size = args.batch_size
        self.train_x = self.test_x = self.train_y = self.test_y = None
        self.encode_labels()
        self.split_data_to_train_test()

    def clean_csv(self):
        self.csv_data = self.csv_data.drop('file_name', axis=1)
        self.csv_data = self.csv_data.drop('hand', axis=1)
        self.csv_data = shuffle(self.csv_data)

        self.csv_data = np.array(self.csv_data)
        self.csv_label = array(self.csv_label)
        print(f"csv_data shape: {self.csv_data.shape}")

    def encode_labels(self):
        label_mappings = {}
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.csv_label)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        self.csv_label = onehot_encoder.fit_transform(integer_encoded)

        for i in range(0, self.csv_label.shape[1]):
            inverted = label_encoder.inverse_transform([i])
            label_mappings[str(i)] = inverted[0]
            print(f"{i} --> {inverted}")

        with open('encoded_labels.json', 'wb') as fp:
            pickle.dump(label_mappings, fp)

    def split_data_to_train_test(self):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.csv_data, self.csv_label,
                                                                                test_size=self.split_size)
        print(f"train_x shape: {self.train_x.shape}")
        print(f"train_y shape: {self.train_y.shape}")
        print(f"test_x shape: {self.test_x.shape}")
        print(f"test_y shape: {self.test_y.shape}")

    def get_train_data(self):
        self.train_x = self.train_x.reshape(self.train_x.shape[0], self.train_x.shape[1], 1)
        train_ds = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds

    def get_test_data(self):
        self.test_x = self.test_x.reshape(self.test_x.shape[0], self.test_x.shape[1], 1)
        test_ds = tf.data.Dataset.from_tensor_slices((self.test_x, self.test_y))
        test_ds = test_ds.batch(self.batch_size)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        return test_ds
