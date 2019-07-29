import utils, fileutils, paths
import numpy as np
import librosa
import geneticsearch
from model import Model
import keras
import tensorflow as tf
import os
from utils import get_train_test, reshape_data
import scipy

tf.set_random_seed(0)

input_shape = (40, 98, 1)
model_file = 'model3.h5'
model_path = os.path.join('models', 'small', model_file)

'''
X_train, X_test, y_train, y_test = get_train_test(path=paths.get_small_path(), input_shape=input_shape)
X_train, X_test, y_train_hot, y_test_hot = reshape_data(input_shape,X_train, X_test, y_train, y_test)
print(model.model.evaluate(X_test, y_test_hot))
print(model.model.metrics_names)
np.random.seed(42)
#path = paths.pick_random_sample(path=paths.get_small_path())
#print(path)

path = r'.\small_data\stop\f92e49f3_nohash_2.wav'
y = utils.wav(path)
n = len(y)
hop_length = 512
y_pad = np.pad(y, [0, np.mod(n, hop_length)], mode='constant')
print(np.shape(y_pad))
ft = librosa.stft(y_pad, hop_length=hop_length)

utils.save_array_to_wav('test_out', 'ft.wav', y, 16000)
#ft = (librosa.stft(y))
#utils.visualize_stf(ft)
#ft[-350:][:] = 1.0
gensearch = geneticsearch.GeneticSearch(model=model, filepath=path,
                                                epochs=200, nb_parents=8, mutation_rate=0.001,
                                                popsize=70)

for _ in range(5):
    y = gensearch.mutate_fourier(y)
utils.visualize_stf(librosa.stft(y))

y_out = librosa.util.fix_length(librosa.istft(ft, hop_length=hop_length), n)
new = librosa.istft(ft)
#print(scipy.spatial.distance.euclidean(y, y_out))
utils.save_array_to_wav('test_out', 'padded_tft.wav', y, 16000)

files = os.listdir('test_out')
utils.compare_wavs( utils.wav(os.path.join('test_out', files[0])), utils.wav(os.path.join('test_out', files[1])))



#utils.compare_stft(utils.wav(os.path.join('test_out', 'epoch_0_up.wav')),utils.wav(os.path.join('test_out', 'epoch_228_down.wav')))

#files = os.listdir('test_out')
#for file in files:
#wav = utils.wav(os.path.join('test_out', 'epoch_0_up.wav'))

sum = np.empty(16000)
    for _ in range(1000):
        sum = gensearch.mutate(sum)
    utils.compare_stft(utils.wav(os.path.join('test_out', 'epoch_0_up.wav')), sum)


utils.compare_wavs(utils.wav(os.path.join('test_out', 'epoch_0_on.wav')), utils.wav(os.path.join('test_out', 'epoch_181_down.wav')))
utils.compare_stft(utils.wav(os.path.join('test_out', 'epoch_0_on.wav')), utils.wav(os.path.join('test_out', 'epoch_181_down.wav')))
'''
model = Model(input_shape=input_shape, version=model_path, path=paths.get_small_path())
path = paths.pick_random_sample(path=paths.get_small_path())

for target in range(0, 9):
    gensearch = geneticsearch.GeneticSearch(model=model, filepath=path,
                                            epochs=1000, nb_parents=12,
                                            popsize=50)

    gensearch.targeted_search(target, 'test_out')
    #geneticsearch.search('test_out')
