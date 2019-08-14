import numpy as np
import matplotlib
import os
from utils import wav


class Evaluation:
    def __init__(self, eps=[0.01, 0.005, 0.0075], mode='SimBA'):
        self.mode = mode
        self.eps = eps
        self.sim_dir = get_dir('SimBA')
        self.gen_dir = get_dir('gen')
        self.sim_data = self.get_data('SimBA')
        self.gen_data = self.get_data('gen')

    def get_data(self, mode):
        data = dict()
        dir = self.gen_dir if mode == 'gen' else self.sim_dir
        for val in self.eps:
            data[val] = dict()
            ep_path = os.path.join(dir, str(val))
            try:
                files = [x for x in os.listdir(ep_path)]
            except FileNotFoundError:
                return None
            for index, file in enumerate(files):
                if file.endswith('.wav'):
                    success = False if 'FAIL' in file else True
                    try:
                        deltas = np.load(os.path.join(ep_path, file[:-4]+'_deltas.npy'))
                        status = 'tar' if 'target' in file else 'ntar'
                    except FileNotFoundError:
                        status = 'original'
                        success = None
                        deltas = None
                    data[val][index] = {'wav': wav(os.path.join(ep_path, file)),
                                        'deltas': deltas,
                                        'queries': file.split('q_')[0].split('label_')[1],
                                        'status': status, 'success': success}
                else:
                    pass
        return data


def get_dir(mode):
    top_dir = 'test_out'
    if mode == 'SimBA':
        return os.path.join(top_dir, 'SimBA')
    elif mode == 'gen':
        return os.path.join(top_dir, 'gen')
    else:
        return top_dir


#take eps_dir
def get_mean_corr(dir):
    wavs = [x['wav'] for x in dir]
    corr_coffs = np.corrcoef(wavs)
    sum = 0
    n = len(corr_coffs)
    for x in range(n):
        for y in range(n):
            if x < y:
                sum += (corr_coffs[x, y])
    return sum/((n ^ 2 - n)/2)


def sort_status(dic):
    og = dict()
    tar = dict()
    ntar = dict()
    for key, val in dic.items():
        if val['status'] == 'original':
            og[key] = val
        elif val['status'] == 'tar':
            tar[key] = val
        elif val['status'] == 'ntar':
            ntar[key] = val
    return og, tar, ntar


def get_suc_rate(dic):
    sucs = len([x for x in dic if dic[x]['success']])
    fails = len([x for x in dic if dic[x]['success']==False])
    try:
        return sucs/(sucs+fails)
    except ZeroDivisionError:
        return 0

#TODO analyze deltas


eval = Evaluation()
og, tar, ntar = sort_status(eval.sim_data[0.005])
print(get_suc_rate(tar), get_suc_rate(ntar))
