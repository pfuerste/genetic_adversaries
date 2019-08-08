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
import simba
tf.set_random_seed(0)


'''
input_shape = (13, 100, 1)
epochs = 50
batch_size = 64
verbose = 1
num_classes = 10
X_train, X_test, y_train, y_test = get_train_test(path=paths.get_small_path(), input_shape=input_shape)
X_train, X_test, y_train_hot, y_test_hot = reshape_data(input_shape,X_train, X_test, y_train, y_test)

model = Model(input_shape, 'model3_13x100')
model.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
#cb_w = keras.callbacks.ModelCheckpoint('../drive/My Drive/cputestw.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
#cb_m = keras.callbacks.ModelCheckpoint('model3_13x100', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#
#model.model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot), callbacks=[cb_m])
#model.model.save('model3_13x100.h5')
# model.model.load_weights('../drive/My Drive/weights4_e70_bs64_c30.h5')
print(model.model.evaluate(X_test, y_test_hot))
print(model.model.metrics_names)
#print(model.model.evaluate(X_test, y_test_hot))
#print(model.model.metrics_names)
#np.random.seed(42)
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
for file in files:
    if not os.path.isdir(os.path.join('.', 'test_dir', file)):
        print(file)
        utils.compare_wavs(utils.wav(os.path.join('test_out', files[2])), utils.wav(os.path.join('test_out', file)))

        #utils.compare_stft(utils.wav(os.path.join('test_out', 'epoch_0_up.wav')),utils.wav(os.path.join('test_out', 'epoch_228_down.wav')))

for target in range(0,1):
    gensearch = geneticsearch.GeneticSearch(model=model, filepath=path,
                                            epochs=3000, nb_parents=12,
                                            popsize=50)

    #gensearch.targeted_search(target, os.path.join('test_out', '3000epochs_noGE'))
    gensearch.search(os.path.join('test_out', 'untargeted0'))


sum = np.empty(16000)
    for _ in range(1000):
        sum = gensearch.mutate(sum)
    utils.compare_stft(utils.wav(os.path.join('test_out', 'epoch_0_up.wav')), sum)
    sf = librosa.feature.spectral_flatness(y=utils.wav(os.path.join('test_out', '0.5000000378365392_fail_stop.wav'))

_, files = os.walk(os.path.join('test_out', 'SimBA'))

utils.compare_wavs(utils.wav(os.path.join('test_out', files[0])), utils.wav(os.path.join('test_out', files[1])))
utils.compare_stft(utils.wav(os.path.join('test_out', files[0])), utils.wav(os.path.join('test_out', files[1])))
'''
input_shape = (13, 100, 1)
model_file = 'model3_13x100'
model_path = os.path.join('models', 'small', model_file)
model = Model(input_shape=input_shape, version=model_path, path=paths.get_small_path())
path = paths.pick_random_sample(path=paths.get_small_path())
for label in range(0, 9):
    sim = simba.SimBA(model=model, path=path, id=0)
    sim.targeted_attack(label)
    sim.attack()
'''

dir = r'test_out\SimBA\run5'
files = os.listdir(dir)
og = files[0]
files.remove(og)
for file in files:
    utils.compare_wavs(utils.wav(os.path.join(dir, og)),
                       utils.wav(os.path.join(dir, file)), range=[-1.3, 1.3])
    utils.compare_stft(utils.wav(os.path.join(dir, og)),
                       utils.wav(os.path.join(dir, file)), range=[-1.3, 1.3])
    utils.compare_mfccs(utils.wav(os.path.join(dir, og)),
                       utils.wav(os.path.join(dir, file)), range=[-1.3, 1.3], input_shape=(13, 100))

'''
#fileutils.save_data_to_array([13, 100], path=paths.get_small_path())