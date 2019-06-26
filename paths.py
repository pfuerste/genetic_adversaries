import os
import random

import numpy as np
from keras.utils import to_categorical


def get_data_path():
    path = './data/'
    if not os.path.isdir(path):
        path = '../content/drive/My Drive/data'
    return path


def get_dummy_path():
    try:
        path = './dummydata/'
    except FileNotFoundError:
        path = '../content/drive/My Drive/dummydata'
    return path


# Input: Folder Path
def get_labels(path=get_data_path()):
    # labels = os.listdir(path)
    labels = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if 'mfcc_vectors' in labels:
        labels.remove('mfcc_vectors')
    if 'mfcc_vectors_big' in labels:
        labels.remove('mfcc_vectors_big')
    if 'spec_vectors' in labels:
        labels.remove('spec_vectors')
    if '_background_noise_' in labels:
        labels.remove('_background_noise_')
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