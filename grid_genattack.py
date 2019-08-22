import os
from model import Model
import paths
import geneticsearch
import numpy as np


input_shape = (13, 100, 1)
model_file = 'model3_13x100.h5'
model_path = os.path.join('models', 'small', model_file)
model = Model(input_shape=input_shape, version=model_path, path=paths.get_small_path())

for _ in range(100):
    for eps in [0.005, 0.01, 0.015, 0.02]:
        path = paths.pick_random_sample(path=paths.get_small_path())
        gensearch = geneticsearch.GeneticSearch(model=model, filepath=path,
                                                epochs=3000, popsize=40,
                                                nb_parents=8, noise_std=eps, softmax_parenting=True)
        gensearch.search()
        rand = np.random.randint(0, 9, 2)
        gensearch.targeted_search(rand[1])
        gensearch.targeted_search(rand[0])
