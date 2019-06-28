import keras
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from fileutils import get_labels
from utils import wav2mfcc


class Model:
    def __init__(self, input_shape, type=1, optimizer='Adadelta'):
        self.num_classes = len(get_labels()[0])
        self.input_shape = input_shape
        self.model = self.get_model(input_shape, type, optimizer)

    def get_model(self, input_shape, type, optimizer):
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
            model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
            model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
            model.add(Flatten())
            model.add(Dense(32))
            model.add((Dropout(0.25)))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(self.num_classes, activation='softmax'))

        elif type == 3:
            print('Experimental Model chosen.')
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(8, 20), activation='relu', input_shape=input_shape))
            model.add(Dropout(0.25))
            model.add(MaxPooling2D(pool_size=(3, 1)))
            model.add(Conv2D(64, kernel_size=(4, 10), activation='relu'))
            model.add(Flatten())
            model.add(Dense(32, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return model

    def get_confidence_scores(self, filepath):
        sample = wav2mfcc(filepath, input_shape=self.input_shape)
        sample_reshaped = sample.reshape(1, self.input_shape[0],
                                         self.input_shape[1], self.input_shape[2])
        return self.model.predict(sample_reshaped, 1)

    # Predicts one sample
    def predict(self, filepath):
        sample = wav2mfcc(filepath, input_shape=self.input_shape)
        sample_reshaped = sample.reshape(1, self.input_shape[0],
                                         self.input_shape[1], self.input_shape[2])
        return get_labels()[0][
                np.argmax(self.model.predict(sample_reshaped))
        ]