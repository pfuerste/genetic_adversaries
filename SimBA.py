import paths, utils
import librosa
import numpy as np
import os


class SimBA:
    def __init__(self, model, path, eps=0.0075):
        self.model = model
        self.path = path
        self.ini_wav, self.sr = librosa.load(self.path, mono=True, sr=None)
        self.wav = np.copy(self.ini_wav)
        self.budget = 16000
        self.queries = 0
        self.eps = eps
        self.out_dir = paths.get_out_dir(eps, mode='SimBA')
        self.ini_class = None
        self.runid = self.get_runid()
        self.targeted = False
        self.ini_class, self.ini_index = self.model.predict(self.path)
        self.ini_prob = self.get_probs(self.ini_wav)[self.ini_index]
        self.init = True
        self.save_attack(None, 'ORIGINAL_'+self.ini_class, None)
        self.init = False

    def reset_instance(self):
        self.targeted = False
        self.queries = 0
        self.wav = self.ini_wav

    def print_log(self):
        ini_class = self.model.predict_array(self.ini_wav)[0]
        new_class = self.model.predict_array(self.wav)[0]
        if ini_class == new_class:
            print('Attack was not successfull.')
        else:
            print('Changed prediction from {} to {}'.format(ini_class, new_class))
        print('Correlation between Original and pertubed wav: ', np.corrcoef(self.wav, self.ini_wav)[0][1])
        print('Finding this pertubation took {} queries to the model and the step size was {}'
              .format(self.queries, self.eps))
        print('|--------------------------------------------------------------------------------------------|')

    def get_runid(self):
        try:
            files = os.listdir(self.out_dir)
            ids = [int(elem.split('id')[0]) for elem in files]
            return np.max(ids)+1
        except (ValueError, FileNotFoundError) as error:
            return 0

    def get_saveid(self, target, label):
        if self.targeted or target == 'Original':
            return '{}id_{}target_{}_label_{}q_{}ip.wav'.format(self.runid, target, label, self.queries, self.ini_prob)
        else:
            return '{}id_{}_label_{}q_{}ip.wav'.format(self.runid, label, self.queries, self.ini_prob)

    def save_attack(self, target, label, deltas):
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        utils.save_array_to_wav(self.out_dir, self.get_saveid(target, label), self.wav, self.sr)
        if not self.init:
            print('saving probs')
            self.save_probs(target, label, deltas)

    def save_probs(self, target, label, deltas):
        np.save(os.path.join(self.out_dir, '{}_deltas.npy'.format(self.get_saveid(target, label)[:-4])), deltas)

    def get_probs(self, wav):
        return self.model.get_confidence_scores(wav)

    def get_qprobs(self, wav, q):
        wav[q] += self.eps
        return self.model.get_confidence_scores(wav)

    def get_neg_qprobs(self, wav, q):
        wav[q] -= self.eps
        return self.model.get_confidence_scores(wav)

    def attack(self):
        Q = np.arange(self.sr)
        deltas = np.empty(16000)
        probs = self.get_probs(self.wav)
        for epoch in range(self.budget):
            #if epoch % 1000 == 0:
                #print('Epoch {}, initial class probability: {}'.format(epoch, probs[self.ini_index]))
                #print('Ini label {}, label now {}'.format(self.ini_class, (self.model.predict_array(self.wav)[0])))
            if probs[self.ini_index] < np.max(probs):
                self.save_attack(None, self.model.predict_array(self.wav)[0], deltas)
                self.print_log()
                self.reset_instance()
                return None
            q = np.random.choice(Q)
            Q = np.delete(Q, np.argwhere(Q == q))
            pos_probs = self.get_qprobs(self.wav, q)
            if pos_probs[self.ini_index] < probs[self.ini_index]:
                deltas[epoch] = pos_probs[self.ini_index] - probs[self.ini_index]
                self.wav[q] += self.eps
                probs = pos_probs
                self.queries += 1
            else:
                neg_probs = self.get_neg_qprobs(self.wav, q)
                if neg_probs[self.ini_index] < probs[self.ini_index]:
                    deltas[epoch] = neg_probs[self.ini_index] - probs[self.ini_index]
                    self.wav[q] -= self.eps
                    probs = neg_probs
                else:
                    deltas[epoch] = 0
                self.queries += 2
        self.save_attack(None, 'FAIL_'+self.model.predict_array(self.wav)[0], deltas)
        self.print_log()
        self.reset_instance()

    def targeted_attack(self, target_label):
        if target_label == self.ini_index:
            return None
        deltas = np.empty([16000, 2])
        Q = np.arange(self.sr)
        self.targeted = True
        probs = self.get_probs(self.wav)
        for epoch in range(self.budget):
            #if epoch % 1000 == 0:
            #    print('Epoch {}, initial class probability: {}, target class probability: {}'
            #          .format(epoch, probs[self.ini_index], probs[target_label]))
            if np.float(probs[self.ini_index]) < np.float(probs[target_label]):
                self.save_attack(target_label, 'FAIL_' + self.model.predict_array(self.wav)[0], deltas)
                self.print_log()
                self.reset_instance()
                return None
            q = np.random.choice(Q)
            Q = np.delete(Q, np.argwhere(Q == q))
            pos_probs = self.get_qprobs(self.wav, q)
            if pos_probs[target_label] > probs[target_label]:
                deltas[epoch][0] = pos_probs[self.ini_index] - probs[self.ini_index]
                deltas[epoch][1] = pos_probs[target_label] - probs[target_label]
                self.wav[q] += self.eps
                probs = pos_probs
                self.queries += 1
            else:
                neg_probs = self.get_neg_qprobs(self.wav, q)
                if neg_probs[target_label] > probs[target_label]:
                    deltas[epoch][0] = pos_probs[self.ini_index] - probs[self.ini_index]
                    deltas[epoch][1] = pos_probs[target_label] - probs[target_label]
                    self.wav[q] -= self.eps
                    probs = neg_probs
                else:
                    deltas[epoch][:] = 0
                self.queries += 2
        self.save_attack(target_label, 'FAIL_'+self.model.predict_array(self.wav)[0], deltas)
        self.print_log()
        self.reset_instance()
