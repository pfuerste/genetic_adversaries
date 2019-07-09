import utils, fileutils, paths
import numpy as np
import librosa
import geneticsearch, model

'''
#print(np.shape(utils.wav2mfcc(paths.pick_random_sample())))
fileutils.save_data_to_array(path=paths.get_dummy_path(), input_shape=(40, 98, 1))
X_train, X_test, y_train, y_test = utils.get_train_test(path=paths.get_dummy_path(), input_shape=(40, 98, 1))
sample = X_train[3]
print(np.shape(sample))
utils.visualize_mfcc(sample)
'''
max = 0.0
min = 0.0


path = paths.pick_random_sample(paths.get_dummy_path())
model = model.Model((40, 98, 1), 1)
geneticsearch = geneticsearch.GeneticSearch(model=model, filepath=path,
                                            epochs=2, nb_parents=2, mutation_rate=0.015,
                                            popsize=10)
