import os
from model import Model
import paths
import simba
import numpy as np

input_shape = (13, 100, 1)
model_file = 'model3_13x100.h5'
model_path = os.path.join('models', 'small', model_file)
model = Model(input_shape=input_shape, version=model_path, path=paths.get_small_path())

for _ in range(1000):
    eps = [0.001, 0.005, 0.01, 0.015, 0.02]
    for ep in eps:
        print('starting ep {}.'.format(ep))
        for random in range(5):
            path = paths.pick_random_sample(path=paths.get_small_path())
            sim = simba.SimBA(model=model, path=path, eps=ep)
            sim.attack()
            rand = np.random.randint(0, 9, 2)
            sim.targeted_attack(rand[0])
            sim.targeted_attack(rand[1])
        print('ep {} done.'.format(ep))


