import os
import random

import numpy as np
from keras.utils import to_categorical


def get_data_path():
    if os.path.isdir(os.path.join('..', 'genetic_adversaries')) and not os.name == 'nt':
        path = os.path.join('..', '..', 'content', 'drive', 'My Drive', 'data')
        return path
    else:
        path = os.path.join('.', 'data')
        return path


def get_small_path():
    if os.path.isdir(os.path.join('..', 'genetic_adversaries')) and not os.name == 'nt':
        path = os.path.join('..', '..', 'content', 'drive', 'My Drive', 'small_data')
        return path
    else:
        path = os.path.join('.', 'small_data')
        return path


def get_dummy_path():
    if os.path.isdir(os.path.join('..', 'genetic_adversaries')) and not os.name == 'nt':
        path = os.path.join('..', '..', 'content', 'drive', 'My Drive', 'dummydata')
        return path
    else:
        path = os.path.join('.', 'dummydata')
        return path


# Input: Folder Path
def get_labels(path=get_data_path()):

    labels = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    non_labels = ['mfcc_vectors_40x98', 'mfcc_vectors_98x40', 'mfcc_vectors', 'mfcc_vectors_big',
                  'mfcc_vectors_13x100', 'mfcc_vectors_big2', 'spec_vectors', '_background_noise_' ]
    for non_label in non_labels:
        try:
            labels.remove(non_label)
        except ValueError:
            pass

    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


def pick_random_sample(path=get_data_path(), input_size=32044):
    labels = get_labels(path)[0]
    label = random.choice(labels)
    rnd_label_path = os.path.join(path, label)
    rnd_sample = random.choice([x for x in os.listdir(rnd_label_path)
                                if os.path.isfile(os.path.join(rnd_label_path, x))
                                and os.path.getsize(os.path.join(rnd_label_path, x)) == input_size])
    rnd_sample_path = os.path.join(path, label, rnd_sample)
    return rnd_sample_path
