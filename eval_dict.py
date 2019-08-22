import numpy as np
import matplotlib.pyplot as plt
import os
from utils import wav
import glob


class Evaluation:
    def __init__(self, mode='SimBA'):
        self.mode = mode
        self.sim_dir = get_dir('SimBA')
        self.gen_dir = get_dir('gen')
        self.sim_data = self.get_data('SimBA')
        self.gen_data = self.get_data('gen')

    def get_data(self, mode):
        data = dict()
        dir = self.gen_dir if mode == 'gen' else self.sim_dir
        eps = [float(i) for i in os.listdir(dir)]
        for val in eps:
            data[val] = dict()
            ep_path = os.path.join(dir, str(val))
            try:
                files = [x for x in os.listdir(ep_path)]
            except FileNotFoundError:
                return None
            for index, file in enumerate(files):
                if file.endswith('.wav'):
                    success = False if 'FAIL' in file else True
                    if 'ORIGINAL' in file:
                        status = 'original'
                        success = None
                        cc = None
                    else:
                        id = file.split('id')[0]
                        original_file = glob.glob(os.path.join(ep_path, id + 'id_ORIGINAL*'))
                        cc = np.corrcoef(wav(os.path.join(ep_path, file)), wav(original_file[0]))[0][1]
                    if 'target' in file:
                        status = 'tar'
                    else:
                        status = 'ntar'
                    try:
                        deltas = np.load(os.path.join(ep_path, file[:-4]+'_deltas.npy'))
                    except FileNotFoundError:
                        deltas = None
                    data[val][index] = {'wav': wav(os.path.join(ep_path, file)),
                                        'deltas': deltas,
                                        'queries': file.split('q_')[0].split('label_')[1],
                                        'status': status, 'success': success,
                                        'cc': cc}
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
def get_mean_corr(dic):
    wavs = [x['wav'] for x in dic]
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
    return og, ntar, tar


def sort_by(dic, param):
    new_dic = dict()
    for key, val in dic.items():
        if [val[param]][0] != None:
            new_dic[val[param]] = val
    return new_dic


def get_suc_rate(dic):
    sucs = len([x for x in dic if dic[x]['success']])
    fails = len([x for x in dic if dic[x]['success']==False])
    try:
        return sucs/(sucs+fails)
    except ZeroDivisionError:
        return 0


def simple_plot(x, y):
    plt.plot(x, y[0], label='non-targeted sr', marker='o', color='blue')
    plt.plot(x, y[1], label='targeted sr', marker='o', color='red')
    plt.xlabel('Epsilon')
    plt.legend()
    plt.show()


#def heatmap():
#TODO analyze deltas

def sr_to_eps():
    eval = Evaluation()
    eps = list()
    nt_sr = list()
    t_sr = list()
    for ep in eval.sim_data:
        og, ntar, tar = sort_status(eval.sim_data[ep])
        eps.append(ep)
        nt_sr.append(get_suc_rate(ntar))
        t_sr.append(get_suc_rate(tar))
        print('success rate for ep {}: \nnon-targeted: {}\ntargeted: {}'.format(ep, get_suc_rate(ntar), get_suc_rate(tar)))
    simple_plot(eps, [nt_sr, t_sr])

# Something here went terribly wrong & SPAGHETTI AF
def sr_to_cc():
    eval = Evaluation()
    cc = list()
    suc = list()
    for ep in eval.sim_data:
        new = sort_by(eval.sim_data[ep], 'cc')
        for key in sorted(new.keys()):
            cc.append(key)
            suc.append(1 if new[key]['success'] else 0)
    cc_sorted = list()
    suc_sorted = list()
    i = 10
    min = 0.70
    max = np.max(cc)
    ccrange = max-min
    for index in range(i):
        low = min+(ccrange/i)*index
        high = low+(ccrange/i)
        lower = False
        lower_index = 0
        for index2, elem in enumerate(cc):
            if elem > low:
                if not lower:
                    lower_index = index2
                    lower = True
            if elem > high:
                cc_sorted.append(elem)
                sucs = [1 for x in suc[lower_index:index2] if x == 1]
                fails = [1 for x in suc[lower_index:index2] if x == 0]
                try:
                    suc_sorted.append(len(sucs)/(len(sucs)+len(fails)))
                except ZeroDivisionError:
                    suc_sorted.append(0)
                lower = False
    plt.scatter(cc_sorted, suc_sorted)
    plt.show()

#eval = Evaluation()
#sort_by(eval.sim_data[0.005], 'cc')
sr_to_cc()

