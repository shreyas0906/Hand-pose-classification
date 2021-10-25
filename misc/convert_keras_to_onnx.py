import tensorflow as tf 
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import onnx
import keras2onnx, os
from argparse import ArgumentParser
from datetime import datetime
import tf2onnx
import onnxruntime as rt


def convert_from_keras_to_onnx():

    model = load_model('cloud_models/model_24-6-3:48/') #os.path.join(args.keras_model, 'trained_model.h5'
    print(model.summary())

    onnx_model = keras2onnx.convert_keras(model, name="example", target_opset=9, channel_first_inputs=None)
    onnx.save_model(onnx_model, 'cloud_converted_model.onnx')

def convert_from_tf_to_onxx():
    
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    output_path = model.name + ".onnx"



if __name__ == '__main__':

    # p = ArgumentParser()
    # p.add_argument('--keras_model', type=str, required=True, default='models/model_24-6-3:48/')

    # args = p.parse_args()
    
    convert_from_keras_to_onnx()
