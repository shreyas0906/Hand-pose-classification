from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.core import Flatten


def test_model():
    model = Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(10)
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


def conv_model(input_size, number_of_gestures):

    inputs = Input(input_size)
    conv1d = Conv1D(filters=512, kernel_size=3, activation='linear')(inputs)
    conv2d = Conv1D(filters=256, kernel_size=3, activation='elu')(conv1d)
    conv3d = Conv1D(filters=128, kernel_size=3, activation='relu')(conv2d)
    conv4d = Conv1D(filters=64, kernel_size=2, activation='elu')(conv3d)
    drop1 = Dropout(0.2)(conv4d)
    max_pool = MaxPooling1D(pool_size=2)(drop1) 
    flatten = Flatten()(max_pool)
    dense1 = Dense(256, activation='relu')(flatten)
    dense2 = Dense(128, activation='elu')(dense1)
    dense3 = Dense(32, activation='relu')(dense2)
    outputs = Dense(number_of_gestures, activation='softmax')(dense3)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='model_diagnostics/model_architecture.jpg', show_shapes=True)

    return model

