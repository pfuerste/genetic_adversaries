import os
import random

import numpy as np
from keras.utils import to_categorical


def get_out_dir(rate, mode):
    if os.path.isdir(os.path.join('..', 'genetic_adversaries')) and not os.name == 'nt':
        path = os.path.join('..', '..', 'content', 'drive', 'My Drive', 'test_out', mode, str(rate))
    else:
        path = os.path.join('test_out', mode, str(rate))
    return path


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


def rename_fails(dir):
    labels = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    all_files = os.listdir(dir)
    ids = set([file.split('id')[0] for file in all_files])
    int_ids = [int(i) for i in ids]
    for id in range(np.max(int_ids)+1):
        curr_list = list()
        for file in all_files:
            if file.split('id')[0] == str(id):
                curr_list.append(file)
        for elem in curr_list:
            if 'ORIGINAL' in elem:
                curr_original = elem
                original_label = curr_original.split('ORIGINAL_')[1].split('_label')[0]

        for elem in curr_list:
            if 'FAIL' in elem:
                elem_label = elem.split('FAIL_')[1].split('_label')[0]
                if elem_label != original_label:
                    target_label = labels[int(elem.split('target')[0].split('_')[1])]
                    if target_label == elem_label:
                        print('Wrong label: {} and {} from {}'.format(elem, target_label,
                                                                      curr_original))
                        new_file = str(elem.split('FAIL_')[0]+elem.split('FAIL_')[1])
                        print('renaming {} to {}'.format(os.path.join(dir, elem), os.path.join(dir, new_file)))
                        os.rename(os.path.join(dir, elem), os.path.join(dir, new_file))
            if 'FAIL' not in elem:
                if 'target' in elem:
                    elem_label = elem.split('target_')[1].split('_label')[0]
                    target_label = labels[int(elem.split('target')[0].split('_')[1])]
                    if target_label == original_label:
                        print(os.path.join(dir, elem))
                        os.remove(os.path.join(dir, elem))
                    if target_label != elem_label:
                        print('Wrong label: {} and {} from {}'.format(elem, target_label,
                                                                      curr_original))
                        new_file = str(elem.split('id_')[0] + 'id_FAIL_' + elem.split('id_')[1])
                        print('renaming {} to {}'.format(os.path.join(dir, elem),
                                                         os.path.join(dir, new_file)))
                        os.rename(os.path.join(dir, elem), os.path.join(dir, new_file))

        curr_list = list()
        curr_original = None
        original_label = None