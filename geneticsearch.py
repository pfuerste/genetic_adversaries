import numpy as np
import model, utils
import librosa
import soundfile as sf
import os
import paths
import scipy.special

class GeneticSearch:
    def __init__(self, model, filepath, epochs, nb_parents, mutation_rate,
                 popsize, nb_genes=16000, temp=1.0):
        self.model = model
        self.filepath = filepath
        self.wav = utils.wav(filepath)
        self.fourier = None
        self.og_label = model.predict(filepath)
        self.og_index = model.predict(filepath, index=True)
        self.epochs = epochs
        self.nb_parents = nb_parents
        self.mutation_rate = mutation_rate
        self.init_rate = 0.005
        self.crossover_method = 'spc'
        self.popsize = popsize
        self.nb_genes = nb_genes
        self.population = self.init_population()
        self.conf_scores = None
        # TODO: Momentum for temperature (>1: softer softmax, more parents)
        self.temp = temp
        self.alpha = 0.9
        self.beta = 0.000001

    # Set up an initial population of slightly mutated wav-arrays
    def init_population(self):
        population = np.empty([self.popsize, self.nb_genes])
        for index in range(self.popsize):
            chromosome = self.mutate(self.wav, init=True)
            population[index] = chromosome
        return population

    # TODO momentum / scaling
    # Mutates by randomly changing popsize*mutation_rate genes of a given chromosome by a small random value
    def mutate(self, chromosome, init=False):
        if not init:
            genes = np.random.choice(self.nb_genes, int(self.nb_genes * self.mutation_rate), replace=False)
            for index in genes:
                chromosome[index] = chromosome[index] + np.random.uniform(-0.001, 0.001)
        else:
            genes = np.random.choice(self.nb_genes, int(self.nb_genes * self.init_rate), replace=False)
            for index in genes:
                chromosome[index] = chromosome[index] + np.random.uniform(-0.005, 0.005)
        return chromosome


    # Mutates by randomly changing popsize*mutation_rate genes within a certain frequency range
    # of a given chromosome by a small random value
    def mutate_fourier(self, array):
        if self.fourier is None:
            self.fourier = utils.pad_fourier(array)
        for gene in range(int(self.nb_genes * self.mutation_rate)):
            xloc = np.random.randint(0, 300)
            yloc = np.random.randint(0, 31)
            self.fourier[-xloc][yloc] = self.fourier[-xloc][yloc] + np.random.normal(0.0, 0.5, 1)
        ifourier = librosa.util.fix_length(librosa.istft(self.fourier, hop_length=512), 16000)
        #self.fourier = librosa.istft(self.fourier)
        return ifourier


    # Generates a child for a pair of chromosomes
    def crossover(self, par1, par2):
        if self.crossover_method == 'spc':
            crossover_point = np.random.randint(0, self.nb_genes-1)
            ch1 = np.concatenate((par1[:crossover_point], par2[crossover_point:]))
            ch2 = np.concatenate((par2[:crossover_point], par1[crossover_point:]))
        else:
            raise NotImplementedError
        return ch1, ch2


    # TODO: implement Mutation Momentum
    # def momentum(self):

    # TODO: (Maybe) implement gradient estimation


    # Mates pairs of parents drawn from the pool based on a weighted probability
    def mate_pool(self, pool, scores):
        probs = scipy.special.softmax(scores/self.temp)
        par1 = np.random.choice(self.popsize, self.popsize/2, replace=True, p=probs)
        par2 = np.random.choice(self.popsize, self.popsize/2, replace=True, p=probs)
        parents = np.stack([par1, par2], axis=1)
        parents = [[self.population[pair[0]], self.population[pair[1]]] for pair in parents]
        offspring = [self.crossover(*pair) for pair in parents]
        offspring = np.concatenate(np.array(offspring))
        return offspring


    # Randomly selects pairs of the nb_parents fittest individuals and generates the next population
    def strongest_mate(self, pool, nb_parents=8):
        parents = [utils.random_pairs(pool[:nb_parents]) for i in range(int(self.popsize/2))]
        offspring = [self.crossover(*pair) for pair in parents]
        offspring = np.concatenate(np.array(offspring))
        return offspring


    # Sort by lowest confidence score of initial label / highest  score for target label
    def fit_sort(self, target_label=None):
        scores = np.empty(self.popsize)
        for index, elem in enumerate(self.population):
            if target_label is None:
                scores[index] = self.model.get_confidence_scores(elem)[self.og_index]
            else:
                scores[index] = self.model.get_confidence_scores(elem)[target_label]
        if target_label is not None:
            sort_scores = np.flip(scores.argsort(), 0)
            scores = np.flip(np.sort(scores), 0)
        else:
            sort_scores = scores.argsort()
        sorted_pop = self.population[sort_scores]
        return sorted_pop, scores


    # TODO: delete?
    def get_fittest(self, target_label=None):
        return self.fit_sort(target_label)


    # Tries to decrease confidence in og label
    # by applying crossover between the fittest members and mutating the offspring
    def search(self, out_dir):
        self.fourier = None
        self.population, scores = self.fit_sort()
        for epoch in range(self.epochs):
            print(self.model.get_confidence_scores(self.population[0])[self.og_index])
            offspring = self.strongest_mate(self.population)
            #offspring = self.mate_pool(self.population, scores)
            self.population = np.array([self.mutate(chromosome) for chromosome in offspring])
            self.population, scores = self.fit_sort()
            winner = self.population[0]
            winner_label, winner_index = self.model.predict_array(winner)
            if winner_index != self.og_index:
                print('Changed prediction from {} to {} in {} epochs.'.format(self.og_label, winner_label, epoch))
                utils.save_array_to_wav(out_dir, 'epoch_0_{}.wav'.format(self.og_label), self.filepath, 16000)
                utils.save_array_to_wav(out_dir, 'epoch_{}_{}.wav'.format(epoch, winner_label), winner, 16000)
                print('Aborting.')
                return None
        print('Failed to produce adversarial example with the given parameters.')


    # Maximizes the confidence in a chosen label
    def targeted_search(self, target_label, out_dir):
        self.fourier = None
        self.population, old_scores = self.fit_sort(target_label)
        for epoch in range(self.epochs):
            #print('Best Score: ', old_scores[0])
            print('Mutation Rate: ', self.mutation_rate)
            offspring = self.strongest_mate(self.population)
            #offspring = self.mate_pool(self.population, scores)
            self.population = np.array([self.mutate(chromosome) for chromosome in offspring])
            #self.population = np.array([self.mutate_fourier(chromosome) for chromosome in offspring])
            self.population, new_scores = self.fit_sort(target_label)
            self.mutation_rate = self.get_mutation_rate(old_scores[0], new_scores[0])
            print('Score diff: ', np.abs(old_scores-new_scores))
            old_scores = new_scores
            winner = self.population[0]
            winner_label, winner_index = self.model.predict_array(winner)
            if winner_index == target_label:
                print('Changed prediction from {} to {} in {} epochs.'.format(self.og_label, winner_label, epoch))
                utils.save_array_to_wav(out_dir, 'epoch_0_{}.wav'.format(self.og_label), self.filepath, 16000)
                utils.save_array_to_wav(out_dir, 'epoch_{}_{}.wav'.format(epoch, winner_label), winner, 16000)
                print('Aborting.')
                return None
        print('Failed to produce adversarial example with the given parameters.')


    def get_mutation_rate(self, old, new):
        p_new = self.alpha*self.mutation_rate+(self.beta/np.abs(old-new))
        #print(self.beta/np.abs(old-new))
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