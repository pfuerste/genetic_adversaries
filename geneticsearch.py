import numpy as np
import model, utils
import librosa
import soundfile as sf
import os
import paths
import scipy.special
import sklearn.preprocessing as skp
from scipy.signal import butter, lfilter


def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)


class GeneticSearch:
    def __init__(self, model, filepath, epochs,  popsize, nb_parents=8,
                 mutation_rate=0.005, noise_std=0.0075):
        self.out_dir = paths.get_out_dir(mutation_rate, mode='gen')
        self.runid = self.get_runid()
        self.queries = 0
        self.model = model
        self.filepath = filepath
        self.wav = utils.wav(filepath)
        self.ini_wav = self.wav
        self.fourier = None
        self.ini_class, self.ini_index = model.predict(filepath)
        self.ini_prob = self.model.get_confidence_scores(self.ini_wav)[self.ini_index]
        self.epochs = epochs
        self.nb_parents = nb_parents
        self.mutation_rate = mutation_rate
        self.init_rate = 0.01
        self.noise_std = noise_std
        self.popsize = popsize
        self.nb_genes = 16000
        self.population = self.init_population()
        self.save_attack(None, 'ORIGINAL_' + self.ini_class)

        # TODO: Momentum for temperature (>1: softer softmax, more parents)
        self.temp = 0.01
        self.alpha = 0.9
        self.beta = 0.000001

    def reset_instance(self):
        self.targeted = False
        self.queries = 0
        self.wav = self.ini_wav
        self.init_population()

    def get_runid(self):
        try:
            files = os.listdir(self.out_dir)
            ids = [int(elem.split('id')[0]) for elem in files]
            return np.max(ids)+1
        except (ValueError, FileNotFoundError) as error:
            return 0

    def get_saveid(self, target, label):
        if target == None:
            return '{}id_{}_label_{}q_{}std_{}ip.wav'.format(self.runid, label, self.queries, self.noise_std, self.ini_prob)
        else:
            return '{}id_{}target_{}_label_{}q_{}std_{}ip.wav'.format(self.runid, target, label, self.queries,
                                                                      self.noise_std, self.ini_prob)

    def save_attack(self, target, label):
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        utils.save_array_to_wav(self.out_dir, self.get_saveid(target, label), self.wav, self.nb_genes)

    # Set up an initial population of slightly mutated wav-arrays
    def init_population(self):
        population = np.empty([self.popsize, self.nb_genes])
        for index in range(self.popsize):
            chromosome = self.mutate(self.ini_wav, init=True)
            population[index] = chromosome
        return population

    #TODO this method + save_id

    # Mutates by randomly changing popsize*mutation_rate genes of a given chromosome by a small random value
    def mutate(self, chromosome, init=False):
        rate = self.init_rate if init else self.mutation_rate
        noise = np.random.randn(self.nb_genes)*self.noise_std
        mask = np.random.rand(self.nb_genes) < self.mutation_rate
        noisemask = highpass_filter(mask*noise)
        #noisemask = mask*noise
        new_chromosome = chromosome + noisemask
        return new_chromosome

    # Mutates by randomly changing popsize*mutation_rate genes within a certain frequency range
    # of a given chromosome by a small random value
    def mutate_fourier(self, array, init=False):
        rate = self.init_rate if init else self.mutation_rate
        if self.fourier is None:
            self.fourier = utils.pad_fourier(array)
        for gene in range(int(self.nb_genes * self.mutation_rate)):
            xloc = np.random.randint(0, 200)
            yloc = np.random.randint(0, 31)
            self.fourier[-xloc][yloc] = self.fourier[-xloc][yloc] + np.random.normal(0.0, rate, 1)
        ifourier = librosa.util.fix_length(librosa.istft(self.fourier, hop_length=512), 16000)
        #self.fourier = librosa.istft(self.fourier)
        return ifourier

    # Generates a child for a pair of chromosomes
    def crossover(self, par1, par2):
        crossover_point = np.random.randint(0, self.nb_genes-1)
        ch1 = np.concatenate((par1[:crossover_point], par2[crossover_point:]))
        ch2 = np.concatenate((par2[:crossover_point], par1[crossover_point:]))
        return ch1, ch2

    # Mates pairs of parents drawn from the pool based on a weighted probability
    def mate_pool(self, pool, scores):
        probs = scipy.special.softmax(scores/self.temp)
        par1 = np.random.choice(self.popsize, int(self.popsize/2), replace=True, p=probs)
        par2 = np.random.choice(self.popsize, int(self.popsize/2), replace=True, p=probs)
        parents = np.stack([par1, par2], axis=1)
        parents = [[self.population[pair[0]], self.population[pair[1]]] for pair in parents]
        offspring = [self.crossover(*pair) for pair in parents]
        offspring = np.concatenate(np.array(offspring))
        return offspring

    # Randomly selects pairs of the nb_parents fittest individuals and generates the next population
    def strongest_mate(self, pool):
        parents = [utils.random_pairs(pool[:self.nb_parents]) for i in range(int(self.popsize/2))]
        offspring = [self.crossover(*pair) for pair in parents]
        offspring = np.concatenate(np.array(offspring))
        return offspring

    # Sort by lowest confidence score of initial label / highest  score for target label
    def fit_sort(self, target_label=None):
        scores = np.empty(self.popsize)
        for index, elem in enumerate(self.population):
            if target_label is None:
                scores[index] = self.model.get_confidence_scores(elem)[self.ini_index]
            else:
                scores[index] = self.model.get_confidence_scores(elem)[target_label]
        self.queries += len(self.population)
        if target_label is not None:
            sort_scores = np.flip(scores.argsort(), 0)
            scores = np.flip(np.sort(scores), 0)
        else:
            sort_scores = scores.argsort()
        sorted_pop = self.population[sort_scores]
        #scores = scipy.special.expit(scores)
        return sorted_pop, scores


    # TODO change saving scheme: '{}id_{}target_{}_label_{}q_{}ip.wav'
    # Tries to decrease confidence in ini label
    # by applying crossover between the fittest members and mutating the offspring
    def search(self):
        self.population, old_scores = self.fit_sort()
        for epoch in range(self.epochs):
            if epoch % 100 == 0:
                print('Best Score: ', old_scores[0])
            # print('Mutation Rate: ', self.mutation_rate)
            # if epoch < 50:
            offspring = self.strongest_mate(self.population)
            # offspring = self.mate_pool(self.population, old_scores)
            # if epoch < 5: self.population = self.noise_std = 0.05
            # else: self.noise_std = 0.01
            self.population = np.array([self.mutate(chromosome) for chromosome in offspring])
            # self.population = np.array([self.mutate_fourier(chromosome) for chromosome in offspring])
            self.population, new_scores = self.fit_sort()
            self.mutation_rate = self.get_mutation_rate(old_scores[0], new_scores[0])
            # print('Score diff: ', np.abs(old_scores-new_scores))
            old_scores = new_scores
            winner = self.population[0]
            winner_label, winner_index = self.model.predict_array(winner)
            if winner_index != self.ini_index:
                print('Changed prediction from {} to {} in {} epochs.'.format(self.ini_class, winner_label, epoch))
                self.wav = winner
                self.save_attack(None, winner_label)
                #utils.save_array_to_wav(out_dir, 'epoch_{}_{}.wav'.format(epoch, winner_label), winner, self.nb_genes)
                print('Aborting.')
                self.reset_instance()
                return None
        self.wav = winner
        self.save_attack(None, 'FAIL_'+winner_label)
        self.reset_instance()
        #utils.save_array_to_wav(out_dir, '{}_fail_{}.wav'.format(old_scores[0], winner_label), winner, self.nb_genes)
        print('Failed to produce adversarial example with the given parameters.')


    # Maximizes the confidence in a chosen label
    def targeted_search(self, target_label):
        if target_label == self.ini_index:
            print('Skipping target == initial label.')
            self.reset_instance()
            return None
        self.population, old_scores = self.fit_sort(target_label)
        for epoch in range(self.epochs):
            if epoch%100==0:
                print('Best Score: ', old_scores[0])
            #print('Mutation Rate: ', self.mutation_rate)
            #if epoch < 50:
            offspring = self.strongest_mate(self.population)
            #offspring = self.mate_pool(self.population, old_scores)
            #if epoch < 5: self.population = self.noise_std = 0.05
            #else: self.noise_std = 0.01
            self.population = np.array([self.mutate(chromosome) for chromosome in offspring])
            #self.population = np.array([self.mutate_fourier(chromosome) for chromosome in offspring])
            self.population, new_scores = self.fit_sort(target_label)
            self.mutation_rate = self.get_mutation_rate(old_scores[0], new_scores[0])
            old_scores = new_scores
            winner = self.population[0]
            winner_label, winner_index = self.model.predict_array(winner)
            if winner_index == target_label:
                print('Changed prediction from {} to {} in {} epochs.'.format(self.ini_class, winner_label, epoch))
                self.wav = winner
                self.save_attack(target_label, winner_label)
                #utils.save_array_to_wav(out_dir, 'epoch_{}_{}.wav'.format(epoch, winner_label), winner, 16000)
                print('Aborting.')
                self.reset_instance()
                return None
        self.wav = winner
        self.save_attack(target_label, 'FAIL_'+winner_label)
        self.reset_instance()
        print('Failed to produce adversarial example with the given parameters.')


    def get_mutation_rate(self, old, new):
        if old == new:
            return 0
        else:
            p_new = self.alpha*self.mutation_rate+(self.beta/np.abs(old-new))
            if p_new > self.mutation_rate*2: p_new = self.mutation_rate*2
            if p_new > 0.01: p_new = 0.01
            #print(p_new)
            return p_new














    # Utility functions for testng purposes
    def just_add(self, chromosome):
        genes = np.random.choice(self.nb_genes, int(self.nb_genes*self.mutation_rate), replace=False)
        for index in genes:
            chromosome[index] = chromosome[index]+0.1
            # chromosome[index] = np.random.uniform(-1.0, 1.0)
        return chromosome

    def max_val(self, chromosome):
        chromosome[np.random.choice(self.nb_genes)] = 1.0
        return chromosome

    def just_random(self, chromosome):
        genes = np.random.choice(self.nb_genes, int(self.nb_genes*self.mutation_rate), replace=False)
        for index in genes:
            chromosome[index] = np.random.uniform(-1.0, 1.0)
            # chromosome[index] = np.random.uniform(-1.0, 1.0)
        return chromosome

    def all_random(self):
        chromosome = np.random.uniform(-1.0, 1.0, 16000)
        return chromosome