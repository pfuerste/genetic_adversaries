import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import wav
import glob


class Evaluation:
    def __init__(self):
        self.sim_data = get_data('SimBA')
        self.gen_data = get_data('gen')

    def get_mean_wavcc(self, mode, eps):
        data = self.sim_data[eps] if mode == 'SimBA' else self.gen_data[eps]
        mean_cc = data['cc'].mean()
        return mean_cc

    def get_suc_rate(self, mode, eps, small_dic=None):
        if small_dic is None:
            data = self.sim_data[eps] if mode == 'SimBA' else self.gen_data[eps]
        else:
            data = small_dic
        rate = data['success'].value_counts()
        if rate.size == 0:
            return 0
        try:
            true = rate[True]
        except KeyError:
            return 0
        try:
            false = rate[False]
        except KeyError:
            return 1
        sr = true/(true+false)
        return sr

    def sort_by_param(self, mode, eps, param):
        data = self.sim_data[eps] if mode == 'SimBA' else self.gen_data[eps]
        values = data[param].unique()
        new_dict = dict()
        for value in values:
            new_dict[value] = data[data[param] == value]
        return new_dict

    def sr_by_param(self, mode, eps, param):
        param_sr = dict()
        data = self.sort_by_param(mode, eps, param)
        for key, df in data.items():
            sr = self.get_suc_rate(mode, eps, small_dic=df)
            param_sr[key] = sr
        return param_sr


def get_data(mode):
    data = dict()
    if mode == 'SimBA':
        columns = ['run_id', 'wav_array', 'status', 'queries', 'success',
                   'original_file', 'cc', 'deltas', 'replace']
    else:
        columns = ['run_id', 'wav_array', 'status', 'queries', 'success',
                   'original_file', 'cc', 'deltas', 'filter', 'softmax_parenting']
    folder = get_dir(mode)
    #TODO change to get inner folders
    eps = [float(i) for i in os.listdir(folder)]
    for val in eps:
        data[val] = pd.DataFrame(columns=columns)
        ep_path = os.path.join(folder, str(val))
        run_folders = [x for x in os.listdir(ep_path) if os.path.isdir(os.path.join(ep_path, x))]
        for run_folder in run_folders:
            files = [x for x in os.listdir(os.path.join(ep_path, run_folder))]
            for file in files:
                if file.endswith('.wav'):
                    run_id = run_folder
                    wav_array = wav(os.path.join(ep_path, run_id, file))
                    if 'ORIGINAL' in file:
                        status = 'original'
                        queries = 0
                        success = None
                        original_file = None
                        cc = None
                        deltas = None
                    else:
                        status = 'tar' if 'target' in file else 'ntar'
                        queries = file.split('q_')[0].split('label_')[1]
                        success = False if 'FAIL' in file else True
                        original_file = glob.glob(os.path.join(ep_path,run_id, run_id+'id_ORIGINAL*'))
                        cc = np.corrcoef(wav(os.path.join(ep_path, run_id, file)), wav(original_file[0]))[0][1]
                        deltas = np.load(os.path.join(ep_path, run_id, file[:-4]+'_deltas.npy'))
                        #TODO new params
                    file_data = [run_id, wav_array, status, queries, success, original_file, cc, deltas] if mode == 'SimBA' \
                        else [run_id, wav_array, status, queries, success, original_file, cc, deltas]
                    file_dict = dict(zip(columns, file_data))
                    data[val] = data[val].append(file_dict, ignore_index=True)
                else:
                    pass
    return data


def get_eps(path):
    eps = list()
    for root, dirs, files in os.walk(path, topdown=False):
        if dirs:
            eps = [float(i) for i in dirs]
            print(eps)
            break
    return eps

def get_dir(mode):
    top_dir = 'test_out'
    if mode == 'SimBA':
        return os.path.join(top_dir, 'SimBA')
    elif mode == 'gen':
        return os.path.join(top_dir, 'gen')
    else:
        return top_dir


def bin_sr(dic):
    #keys, items = dic.items()
    key_list = list(dic.keys())

    sorted_keys = sorted((key_list))
    print(sorted_keys)

#eva = Evaluation()
#print(eva.get_mean_wavcc('SimBA', 0.025))
#print(eva.get_suc_rate('SimBA', 0.025))
#(eva.sort_by_param('SimBA', 0.025, 'queries'))
#print((eva.sr_by_param('SimBA', 0.025, 'queries')))
#print(pq)
print(get_eps(get_dir('SimBA')))