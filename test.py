import utils, fileutils, paths
import numpy as np
import librosa
import geneticsearch
from model import Model
import keras
import tensorflow as tf
import os

tf.set_random_seed(0)

input_shape = (40, 98, 1)
model_file = 'model2.h5'
model_path = os.path.join(model_file)

model = Model(input_shape, model_path)
path = paths.pick_random_sample()
geneticsearch = geneticsearch.GeneticSearch(model=model, filepath=path,
                                            epochs=100, nb_parents=8, mutation_rate=0.005,
                                            popsize=48)
#geneticsearch.search('test_out')
geneticsearch.targeted_search(np.random.randint(0, 29), 'test_out')

