import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


DATA_PATH = "./data/"
DUMMY_PATH = "./dummydata/"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    # labels = os.listdir(path)
    labels = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if 'mfcc_vectors' in labels:
        labels.remove('mfcc_vectors')
    if 'spec_vectors' in labels:
        labels.remove('spec_vectors')
    if '_background_noise_' in labels:
        labels.remove('_background_noise_')
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=98):
    wave, sr = librosa.load(file_path, mono=True, sr=None)

    #wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, n_mfcc=40, hop_length = int(sr*0.01), n_fft = int(sr*0.03))


    #mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=40)
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


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(os.path.join('mfcc_vectors', labels[0], '.npy'))
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(os.path.join('mfcc_vectors', label, '.npy'))
        X = np.vstack(X, x)
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


# display an array containing the mfcc
def visualize_mfcc(array):
    mfcc = array
    plt.figure()
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


'''
file = random.choice([x for x in os.listdir(os.path.join(DATA_PATH, 'no'))
                      if os.path.isfile(os.path.join(DATA_PATH, 'no', x)) and
                      os.path.getsize(os.path.join(DATA_PATH, 'no', x)) == 32044])
file = os.path.join(DATA_PATH, 'no', file)
wav, sr = librosa.load(file, mono=True)
print(np.shape(wav))
#D = librosa.amplitude_to_db(np.abs(wav2spec(file)))
D = np.abs(librosa.stft(wav))
print(np.shape(D))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()




file = random.choice([x for x in os.listdir(os.path.join(DATA_PATH, 'no'))
                      if os.path.isfile(os.path.join(os.path.join(DATA_PATH, 'no'), x))])
path = os.path.join(DATA_PATH, 'no', file)
wave, sr = librosa.load(path, mono=True, sr=None)
#wave = wave[::3]
#import matplotlib.pyplot as plt
#import librosa.display
plt.figure(figsize=(15, 10))
#D = librosa.amplitude_to_db(librosa.stft(wave), ref=np.max)
#plt.subplot(4, 2, 1)
#librosa.display.specshow(D, y_axis='linear')
plt.subplot(3, 1, 1)
librosa.display.waveplot(wave, sr)
#plt.colorbar(format='%+2.0f dB')
plt.show()
'''

