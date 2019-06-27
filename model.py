import keras
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from fileutils import get_labels
from utils import wav2mfcc


class Model:
    def __init__(self, input_shape, type):
        self.num_classes = len(get_labels()[0])
        self.input_shape = input_shape
        self.model = self.get_model(input_shape, type)

    def get_model(self, input_shape, type):
        if type == 1:
            print('Simple Model chosen.')
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
            model.add(Dense(self.num_classes, activation='softmax'))

        elif type == 2:
            print('Complex Model chosen.')
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(60, 20), activation='relu', input_shape=input_shape))
            model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=(10, 4), activation='relu'))
            model.add(Flatten())
            model.add(Dense(32))
            model.add((Dropout(0.25)))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return model

    # Predicts one sample
    def predict(self, filepath):
        sample = wav2mfcc(filepath, max_len=self.input_shape[1], n_mfcc=self.input_shape[0])
        sample_reshaped = sample.reshape(1, self.input_shape[0],
                                         self.input_shape[1], self.input_shape[2])
        return get_labels()[0][
                np.argmax(self.model.predict(sample_reshaped))
        ]