import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from paths import get_data_path, get_labels


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=98):
    wave, sr = librosa.load(file_path, mono=True, sr=None)

    # wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, n_mfcc=40, hop_length = int(sr*0.01), n_fft = int(sr*0.03))

    # mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=40)
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if max_len > mfcc.shape[1]:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def wav2spec(file_path, downsample=True):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    if downsample:
        wave = wave[::3]
    # Why abs?
    spec = np.abs(librosa.stft(wave))
    return spec


def get_train_test(path=get_data_path(), input_shape=(40, 98, 1), split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(path)

    # Getting first arrays
    if input_shape == (40, 98, 1): vec_dir = os.path.join(path, 'mfcc_vectors_big')
    else: vec_dir = os.path.join(path, 'mfcc_vectors')

    X = np.load(os.path.join(vec_dir, labels[0]+'.npy'))
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(os.path.join(vec_dir, label+'.npy'))
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    for elem in X:
        elem = elem.T
    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


# returns a dictionary of format dic['label'][['path']['mfcc']]
def prepare_dataset(path=get_data_path()):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            # wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


# returns the first 100 mfccs per label in a list[[label][mfcc]]
def load_dataset(path=get_data_path()):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]


# display an array containing the mfcc
def visualize_mfcc(array):
    mfcc = array
    plt.figure()
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

