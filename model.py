import os
import keras
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model

from paths import get_labels, get_data_path, get_small_path
from utils import wav2mfcc, array2mfcc


class Model:

    def __init__(self, input_shape, version=1, path=get_small_path()):
        self.num_classes = len(get_labels(path)[0])
        self.input_shape = input_shape
        self.path = path
        self.model = self.get_model(input_shape, version)

    def get_model(self, input_shape, version):
        if type(version) is str:
            try:
                model = load_model(version)
            except OSError:
                model = load_model(os.path.join('/net/projects/scratch/winter/valid_until_31_July_2020/pfuerste', version))
            except FileNotFoundError:
                'The chosen model does not exist yet.'

        if version == 1:
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

        elif version == 2:
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

        elif version == 3:
            print('Experimental Model (3) chosen.')
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(4, 4), activation='relu', input_shape=input_shape))
            model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
            model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
            model.add(Dropout(0.25))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(4, 10), activation='relu'))
            model.add(Flatten())
            model.add(Dense(32, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(self.num_classes, activation='softmax'))

        elif version == 4:
            print('Experimental Model (4) chosen.')
            model = Sequential()
            model.add(Conv2D(64, kernel_size=(8, 20), activation='relu', input_shape=input_shape))
            model.add(Dropout(0.25))
            model.add(MaxPooling2D(pool_size=(3, 1)))
            model.add(Conv2D(64, kernel_size=(4, 10), activation='relu'))
            model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(self.num_classes, activation='softmax'))

        return model

    # Predicts one filepath
    def predict(self, filepath):
        sample = wav2mfcc(filepath, input_shape=self.input_shape)
        sample_reshaped = sample.reshape(1, self.input_shape[0],
                                         self.input_shape[1], self.input_shape[2])

        label = get_labels(self.path)[0][
                    np.argmax(self.model.predict(sample_reshaped))
            ]
        index = get_labels(self.path)[1][
                np.argmax(self.model.predict(sample_reshaped))
            ]
        return label, index

    def get_confidence_scores(self, array):
        sample = array2mfcc(array, input_shape=self.input_shape)
        sample_reshaped = sample.reshape(1, self.input_shape[0],
                                         self.input_shape[1], self.input_shape[2])
        return self.model.predict(sample_reshaped)[0]

    # In: wav_array
    # Out: label index
    def predict_array(self, array):
        sample = array2mfcc(array, input_shape=self.input_shape)
        sample_reshaped = sample.reshape(1, self.input_shape[0],
                                         self.input_shape[1], self.input_shape[2])

        label = get_labels(self.path)[0][np.argmax(self.model.predict(sample_reshaped))]
        index = get_labels(self.path)[1][np.argmax(self.model.predict(sample_reshaped))]
        return label, index
