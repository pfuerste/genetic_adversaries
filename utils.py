import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import random
import soundfile as sf

from paths import get_data_path, get_labels


def wav(file_path):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    return wave


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, input_shape):
    n_mfcc = input_shape[0]
    max_len = input_shape[1]
    wave, sr = librosa.load(file_path, mono=True, sr=None)

    mfcc = librosa.feature.mfcc(wave, n_mfcc=n_mfcc, hop_length=int(sr*0.01), n_fft=int(sr*0.03))

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if max_len > mfcc.shape[1]:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def array2mfcc(array, input_shape=(40, 98, 1)):
    n_mfcc = input_shape[0]
    max_len = input_shape[1]
    wave = array
    sr = 16000
    mfcc = librosa.feature.mfcc(wave, n_mfcc=n_mfcc, hop_length=int(sr*0.01), n_fft=int(sr*0.03))

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if max_len > mfcc.shape[1]:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def pad_fourier(array):
    n = len(array)
    y_pad = np.pad(array, [0, np.mod(n, 512)], mode='constant')
    padded_fourier = librosa.stft(y_pad, hop_length=512)
    #assert np.shape(padded_fourier) == (16128,)
    return padded_fourier


def get_train_test(path=get_data_path(), input_shape=(40, 98, 1), split_ratio=0.7, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(path)

    # Getting first arrays
    if input_shape == (40, 98, 1): vec_dir = os.path.join(path, 'mfcc_vectors_40x98')
    elif input_shape == (98, 40 , 1): vec_dir = os.path.join(path, 'mfcc_vectors_98x40')
    else: vec_dir = os.path.join(path, 'mfcc_vectors')
    print('Loading .npy data from {}'.format(vec_dir))

    # Load data of first label
    X = np.load(os.path.join(vec_dir, labels[0]+'.npy'))
    y = np.zeros(X.shape[0])
    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(os.path.join(vec_dir, label+'.npy'))
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


def reshape_data(input_shape, x_train, x_test, y_train, y_test):
    x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
    x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2])

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    return x_train, x_test, y_train_hot, y_test_hot


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


def visualize_stf(array):
    librosa.display.specshow(librosa.amplitude_to_db(array,
                                                  ref=np.max),
                                                    y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


# Compare arrays of original and attacker
def compare_wavs(original, attacker, range=[-0.7, 0.7]):
    sr = 16000
    comp = plt.subplot(211)
    librosa.display.waveplot(original, sr=sr, alpha=1)
    librosa.display.waveplot(attacker, sr=sr, color='r', alpha=1)
    plt.title('Original vs Attacker ')
    comp.set_ylim(range)
    noise = plt.subplot(212)
    librosa.display.waveplot(original - attacker, sr=sr)
    plt.title('Original minus Attacker ')
    noise.set_ylim(-0.7, 0.7)
    plt.tight_layout()
    plt.show()


def compare_mfccs(original, attacker, range=[-0.7, 0.7]):
    original = array2mfcc(original)
    attacker = array2mfcc(attacker)
    ogplt = plt.subplot(311)
    librosa.display.specshow(original, x_axis='original')
    atplt = plt.subplot(312)
    librosa.display.specshow(attacker, x_axis='attacker')
    noiseplt = plt.subplot(313)
    librosa.display.specshow(original-attacker, x_axis='noise')
    plt.tight_layout()
    plt.show()


def compare_stft(original, attacker, range=[-0.7, 0.7]):
    original = np.abs(librosa.stft(original))
    attacker = np.abs(librosa.stft(attacker))
    ogplt = plt.subplot(311)
    librosa.display.specshow(librosa.amplitude_to_db(original, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Original Spectogram')
    atplt = plt.subplot(312)
    librosa.display.specshow(librosa.amplitude_to_db(attacker, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Attackers Spectogram')
    noiseplt = plt.subplot(313)
    librosa.display.specshow(librosa.amplitude_to_db(original, ref=np.max) + librosa.amplitude_to_db(attacker, ref=np.max)
                             , y_axis='log', x_axis='time')
    plt.colorbar()
    plt.title('Pertubation Spectogram')
    plt.tight_layout()
    plt.show()


def random_pairs(number_list):
    return [number_list[i] for i in random.sample(range(len(number_list)), 2)]


def save_array_to_wav(out_dir, filename, array, sr):
    path = os.path.join(out_dir, filename)
    if type(array) is not np.ndarray:
        array = wav(array)
    sf.write(path, array, sr)