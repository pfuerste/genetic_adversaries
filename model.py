import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from utils import wav2mfcc, get_labels
import numpy as np

num_classes = len(get_labels()[0])


def get_model(simple=False):
    if simple:
        input_shape = (20, 11, 1)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
        model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
        model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(num_classes, activation='softmax'))

    else:
        input_shape = (98, 40, 1)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(60, 8), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(10, 4), input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    for layer in model.layers:
        print(layer, layer.output_shape)
    return model


model = get_model()


# Predicts one sample
def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]