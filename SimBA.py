import paths, utils
import librosa
import numpy as np
import os


out_dir = os.path.join('test_out', 'SimBA', 'run1')


class SimBA:
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.ini_wav, self.sr = librosa.load(self.path, mono=True, sr=None)
        self.wav = np.copy(self.ini_wav)
        self.budget = 16000
        self.queries = 0
        self.eps = 0.001
        self.ini_class = None

    def print_log(self):
        ini_class = self.model.predict_array(self.ini_wav)[0]
        new_class = self.model.predict_array(self.wav)[0]
        if ini_class == new_class:
            print('Attack was not successfull.')
        else:
            print('Changed prediction from {} to {}'.format(ini_class, new_class))
        print('Absolute difference between Original and pertubed wav: ', np.sum(self.wav-self.ini_wav))
        print('Finding this pertubation took {} queries to the model and the step size was {}'
              .format(self.queries, self.eps))

    def get_id(self, label):
        return '{}_{}q_{}eps.wav'.format(label, self.queries, self.eps)

    def save_attack(self, label):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        utils.save_array_to_wav(out_dir, self.get_id(label), self.wav, self.sr)

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
        self.ini_class, ini_index = self.model.predict(self.path)
        self.save_attack(self.ini_class)
        probs = self.get_probs(self.wav)
        #probs = np.float(probs[ini_index])
        #probs = np.delete(probs, ini_index)
        for epoch in range(self.budget):
            if epoch % 1000 == 0:
                print('Epoch {}, initial class probability: {}'.format(epoch, probs[ini_index]))
                print('Ini label {}, label now {}'.format(self.ini_class, (self.model.predict_array(self.wav)[0])))
                print(probs[ini_index], np.max(probs))
            if probs[ini_index] < np.max(probs):
                self.save_attack(self.model.predict_array(self.wav)[0])
                self.print_log()
                return None
            q = np.random.choice(Q)
            Q = np.delete(Q, np.argwhere(Q == q))
            pos_probs = self.get_qprobs(self.wav, q)
            if pos_probs[ini_index] < probs[ini_index]:
                self.wav[q] += self.eps
                probs = pos_probs
                self.queries += 1
            else:
                neg_probs = self.get_neg_qprobs(self.wav, q)
                if neg_probs[ini_index] < probs[ini_index]:
                    self.wav[q] -= self.eps
                    probs = neg_probs
                self.queries += 2
        self.save_attack('FAILURE'+self.model.predict_array(self.wav)[0])
        self.print_log()

    def targeted_attack(self, target_label):
        Q = np.arange(self.sr)
        self.ini_class, ini_index = self.model.predict(self.path)
        self.save_attack(self.ini_class)
        probs = self.get_probs(self.wav)
        for epoch in range(self.budget):
            if epoch % 1000 == 0:
                print('Epoch {}, initial class probability: {}, target class probability: {}'
                      .format(epoch, probs[ini_index], probs[target_label]))
            if np.float(probs[ini_index]) < np.float(probs[target_label]):
                self.save_attack(self.model.predict_array(self.wav)[0])
                self.print_log()
                return None
            q = np.random.choice(Q)
            Q = np.delete(Q, np.argwhere(Q == q))
            pos_probs = self.get_qprobs(self.wav, q)
            if pos_probs[target_label] > probs[target_label]:
                self.wav[q] += self.eps
                probs = pos_probs
                self.queries += 1
            else:
                neg_probs = self.get_neg_qprobs(self.wav, q)
                if neg_probs[target_label] > probs[target_label]:
                    self.wav[q] -= self.eps
                    probs = neg_probs
                self.queries += 2
        self.save_attack('FAILURE' + self.model.predict_array(self.wav)[0])
        self.print_log()

