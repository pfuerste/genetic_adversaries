import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import random
from shutil import copyfile
import matplotlib.pyplot as plt
import librosa.display

DATA_PATH = "./data/"
DUMMY_PATH = "./dummydata/"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    # labels = os.listdir(path)
    labels = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


# saves wavs in a folder in numpy-array, one for each label sub-folder
def save_data_to_array(path=DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [os.path.join(path, label, wavfile) for wavfile in os.listdir(os.path.join(path, label))]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        try:
            os.mkdir(os.path.join(path,'mfcc_vectors'))
        except FileExistsError:
            pass
        np.save(os.path.join(path, 'mfcc_vectors', label, '.npy'), mfcc_vectors)
        print('Converted and saved all wavs in {}'.format(label))


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(os.path.join(labels[0], '.npy'))
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(os.path.join(label, '.npy'))
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


# returns a dictionary of format dic['label'][['path']['mfcc']]
def prepare_dataset(path=DATA_PATH):
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
def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]


# Copies dir_size wva-files per class to a dummy-data-folder for testing purposes
def make_dummy_dir(path=DATA_PATH, dummypath=DUMMY_PATH, dir_size=1):
    labels = get_labels(path)[0]
    for label in labels:
        old_dir = os.path.join(path, label)
        new_dir = os.path.join(dummypath, label)
        try:
            os.mkdir(new_dir)
        except FileExistsError:
            pass
        for dummies in range(dir_size):
            file = random.choice([x for x in os.listdir(old_dir) if os.path.isfile(os.path.join(old_dir, x))])
            copyfile(os.path.join(old_dir, file), os.path.join(new_dir, file))


# display an array containing the mfcc
def visualize_mfcc(array):
    mfcc = array
    plt.figure()
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()



