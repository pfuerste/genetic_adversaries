import os
from utils import get_labels
import random
from shutil import copyfile
import numpy as np
from utils import wav2mfcc, wav2spec
from tqdm import tqdm

DATA_PATH = "./data/"
DUMMY_PATH = "./dummydata/"


# Copies dir_size wva-files per class to a dummy-data-folder for testing purposes
def make_dummy_dir(path=DATA_PATH, dummypath=DUMMY_PATH, input_size=32044, dir_size=1):
    labels = get_labels(path)[0]
    for label in labels:
        old_dir = os.path.join(path, label)
        new_dir = os.path.join(dummypath, label)
        try:
            os.mkdir(new_dir)
        except FileExistsError:
            pass
        for dummies in range(dir_size):
            file = random.choice([x for x in os.listdir(old_dir) if (os.path.isfile(os.path.join(old_dir, x))
                                  and os.path.getsize(os.path.join(old_dir, x)) == input_size)])
            copyfile(os.path.join(old_dir, file), os.path.join(new_dir, file))


def check_file_sizes(path=DATA_PATH, check_size='32044', recursive=True):

    dirs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))
            and 'background_noise' not in x]
    for folder in dirs:
        sizes = {}
        files = [x for x in os.listdir(os.path.join(path, folder)) if
                 os.path.isfile(os.path.join(path, folder, x)) and '.wav' or '.npy' in x]
        for file in files:
            size = os.path.getsize(os.path.join(path, folder, file))
            sizestr = str(size)
            try:
                sizes[sizestr] += 1
            except KeyError:
                sizes[sizestr] = 1

        all = sum(sizes.values())
        try:
            frac = sizes[check_size]/all
        except KeyError:
            print('No files of this size!')
            pass

        print('{:>5} Sizes in dir {:>20}: {}'.format(sum(sizes.values()), folder, sizes))
        print('{:>10} percent are 32044 byte long'.format(frac))


# saves wavs in a folder in numpy-array, one for each label sub-folder
def save_data_to_array(output_format, path=DATA_PATH, input_size=32044, max_len=98, Transpose = True):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        out_vectors = []

        wavfiles = [os.path.join(path, label, wavfile) for wavfile in os.listdir(os.path.join(path, label))
                    if os.path.getsize(os.path.join(path, label, wavfile)) == input_size]
        if output_format is 'spec':
            for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
                spec = wav2spec(wavfile)
                out_vectors.append(spec)
        elif output_format is 'mfcc':
            for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
                mfcc = wav2mfcc(wavfile, max_len=max_len)
                if Transpose:
                    out_vectors.append(mfcc.T)
                else:
                    out_vectors.append(mfcc)
        try:
            os.mkdir(os.path.join(path, output_format+'_vectors_big'))
        except FileExistsError:
            pass
        np.save(os.path.join(path, output_format+'_vectors_big', label+'.npy'), out_vectors)


def pick_random_sample(path=DATA_PATH, input_size=32044):
    labels = get_labels(path)[0]
    label = random.choice(labels)
    rnd_label_path = os.path.join(path, label)
    rnd_sample = random.choice([x for x in os.listdir(rnd_label_path)
                                if os.path.isfile(os.path.join(rnd_label_path, x))
                                and os.path.getsize(os.path.join(rnd_label_path, x)) == input_size])
    rnd_sample_path = os.path.join(path, label, rnd_sample)
    return rnd_sample_path

'''
from utils import visualize_mfcc
sample = pick_random_sample()
mfcc = wav2mfcc(sample)
print(np.shape(mfcc))
visualize_mfcc(mfcc)

#make_dummy_dir(dir_size=3)
#check_file_sizes(DUMMY_PATH)
#save_data_to_array(path=DUMMY_PATH)
'''