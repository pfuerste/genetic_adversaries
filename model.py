import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from utils import wav2mfcc, get_labels
import numpy as np


class Model:
    def _init__(self, input_shape):
        try:
            self.num_classes = len(get_labels()[0])
        except FileNotFoundError:
            self.num_classes = len(get_labels(path='../content/drive/My Drive/data')[0])
        self.model = self.get_model(input_shape)

    def get_model(self, input_shape):
        if input_shape == (20, 11, 1):
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

        elif input_shape == (98, 40, 1):
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(60, 20), input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(1, 3)))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=(10, 4)))
            model.add(Flatten())
            model.add(Dense(32))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(self.num_classes, activation='softmax'))


        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model


    # Predicts one sample
    def predict(self, filepath):
        sample = wav2mfcc(filepath)
        sample_reshaped = sample.reshape(1, self.input_shape[0],
                                         self.input_shape[1], self.input_shape[2])
        return get_labels()[0][
                np.argmax(self.model.predict(sample_reshaped))
        ]